#!/usr/bin/env python3
"""
Training script for Koopman SBI Model

This script trains a Koopman student model using a pre-trained flow matching teacher
for SBI benchmark tasks like two_moons. It follows the existing SBI training pattern
but uses the teacher model to generate training data for the Koopman approach.

Usage:
    # Step 1: Train teacher flow matching model first
    python run_sbibm.py --train_dir teacher_two_moons
    
    # Step 2: Train Koopman student
    python train_koopman_sbi.py --teacher_model_path teacher_two_moons/best_model.pt \
                                --settings_file koopman_two_moons_settings.yaml \
                                --train_dir koopman_two_moons

Author: SBI-compatible Koopman Flow Matching implementation
"""

import argparse
import csv
import math
import os
import sys
import logging
import time
from os.path import join
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sbibm.tasks
from sbibm.metrics import c2st
import torch
import numpy as np
import yaml
import wandb
from torch.utils.data import Dataset

# Import registration helper
from register_koopman_model import register_koopman_model, create_koopman_sbi_model
from koopman_sbi_model import KoopmanSBIModel

# Import existing SBI utilities  
from run_sbibm import SbiDataset, generate_dataset, load_dataset

try:
    from dingo.core.posterior_models.build_model import (
        build_model_from_kwargs,
        autocomplete_model_kwargs,
    )
    from dingo.core.utils import build_train_and_test_loaders, RuntimeLimits
    DINGO_AVAILABLE = True
except ImportError:
    DINGO_AVAILABLE = False
    print("Warning: Dingo not available. Limited functionality.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register Koopman model
register_koopman_model()


def train_koopman_model(train_dir, settings, train_loader, test_loader, use_wandb=False):
    """
    Train Koopman model using teacher-student approach
    
    Args:
        train_dir: Training output directory
        settings: Configuration dictionary
        train_loader: Training data loader
        test_loader: Test data loader  
        use_wandb: Whether to use weights & biases logging
        
    Returns:
        Trained KoopmanSBIModel
    """
    logger.info("Starting Koopman model training")
    
    # Autocomplete model kwargs (like standard flow matching)
    autocomplete_model_kwargs(
        settings["model"],
        input_dim=settings["task"]["dim_theta"],  # input = theta dimension
        context_dim=settings["task"]["dim_x"],  # context dim = observation dimension
    )
    
    # Create Koopman model using factory function (simpler approach)
    model = create_koopman_sbi_model(settings, device=settings["training"].get("device", "cpu"))
    
    # Count parameters from individual networks - all components must exist
    assert hasattr(model, 'encoder'), "Model must have encoder component"
    assert hasattr(model, 'decoder'), "Model must have decoder component"  
    assert hasattr(model, 'affine_generator'), "Model must have affine_generator component"
    
    total_params = 0
    total_params += sum(p.numel() for p in model.encoder.parameters())
    total_params += sum(p.numel() for p in model.decoder.parameters())
    # Linear evolution parameters
    total_params += sum(p.numel() for p in model.linear_evolution.parameters())
    # Conditional control network
    total_params += sum(p.numel() for p in model.control_linear.parameters())
    
    logger.info(f"Created Koopman model with {total_params} parameters")
    
    # Setup optimizer and scheduler
    model.optimizer_kwargs = settings["training"]["optimizer"]
    model.scheduler_kwargs = settings["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()
    
    # Load pre-generated Koopman dataset
    logger.info("Loading pre-generated Koopman dataset")
    
    dataset_path = os.path.join(train_dir, "koopman_dataset.pt")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Pre-generated Koopman dataset not found at {dataset_path}")
    
    koopman_data = torch.load(dataset_path)
    logger.info(f"Loaded Koopman dataset with {len(koopman_data['theta_0'])} triplets")
    logger.info(f"Dataset parameters: buffer_size={koopman_data['buffer_size']}, samples_per_observation={koopman_data['samples_per_observation']}")
    
    # Initialize model with pre-generated dataset
    model.load_koopman_dataset(koopman_data)
    
    # Training loop
    runtime_limits = RuntimeLimits(
        epoch_start=0,
        max_epochs_total=settings["training"]["epochs"],
    )
    
    # Use KoopmanSBIModel's built-in training method
    logger.info("Using KoopmanSBIModel training infrastructure")
    model.train(
        train_loader,
        test_loader,
        train_dir=train_dir,
        runtime_limits=runtime_limits,
        early_stopping=settings["training"].get("early_stopping", True),
        use_wandb=use_wandb,
    )
    
    # Return the trained model directly (Koopman models can't use build_model_from_kwargs)
    logger.info("Returning trained Koopman model")
    return model



def save_model(model, train_dir, epoch, loss, is_best=False):
    """Save model checkpoint"""
    os.makedirs(train_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }
    
    if is_best:
        torch.save(checkpoint, join(train_dir, "best_model.pt"))
        logger.info(f"Saved best model at epoch {epoch} with loss {loss:.6f}")
    else:
        torch.save(checkpoint, join(train_dir, f"checkpoint_epoch_{epoch}.pt"))


def evaluate_koopman_model(train_dir, settings, dataset, model, use_wandb=False, skip_c2st=False):
    """
    Evaluate Koopman model using SBI metrics
    
    Args:
        train_dir: Training directory for output
        settings: Configuration
        dataset: SBI dataset
        model: Trained model
        use_wandb: Whether to use wandb
    """
    logger.info("Starting Koopman model evaluation")
    
    task = sbibm.get_task(settings["task"]["name"])
    c2st_scores = {}
    
    # Time comparison
    koopman_times = []
    
    for obs in range(1, 11):
        logger.info(f"Evaluating observation {obs}")
        
        reference_samples = task.get_reference_posterior_samples(num_observation=obs)
        num_samples = len(reference_samples)  # Use full reference dataset size
        
        observation = dataset.standardize(
            task.get_observation(num_observation=obs), label="x"
        )
        
        # Time Koopman sampling
        start_time = time.time()
        
        # Generate samples using Koopman (should be much faster)
        observation_batch = observation.repeat((num_samples * 2, 1))
        posterior_samples = model.sample_batch(observation_batch).detach()
        
        koopman_time = time.time() - start_time
        koopman_times.append(koopman_time)
        
        # Unstandardize samples
        posterior_samples = dataset.standardize(
            posterior_samples, label="theta", inverse=True
        )
        
        # Discard samples outside prior
        prior_mask = torch.isfinite(task.prior_dist.log_prob(posterior_samples))
        logger.info(f"{(1 - torch.sum(prior_mask) / len(prior_mask)) * 100:.2f}% "
                   f"samples outside prior")
        posterior_samples = posterior_samples[prior_mask]
        
        # Compute C2ST if not skipped
        n = min(len(reference_samples), len(posterior_samples))
        if not skip_c2st:
            c2st_score = c2st(posterior_samples[:n], reference_samples[:n])
            c2st_scores[f"C2ST {obs}"] = c2st_score.item()
        else:
            c2st_score = None
            logger.info(f"Skipping C2ST computation for observation {obs}")
        
        # Plot comparison - reference first (underneath), then Koopman on top
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(
            reference_samples[:n, 0].cpu().numpy(),
            reference_samples[:n, 1].cpu().numpy(),
            s=0.5, alpha=0.6, label=f"Reference ({n} samples)", color='red'
        )
        plt.scatter(
            posterior_samples[:n, 0].cpu().numpy(),
            posterior_samples[:n, 1].cpu().numpy(),
            s=0.5, alpha=0.6, label=f"Koopman ({n} samples)", color='blue'
        )
        if c2st_score is not None:
            plt.title(f"Observation {obs}: C2ST = {c2st_score.item():.3f}, "
                     f"Time = {koopman_time:.3f}s ({num_samples*2} generated)")
        else:
            plt.title(f"Observation {obs}: Time = {koopman_time:.3f}s ({num_samples*2} generated)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(join(train_dir, f"posterior_{obs}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
    # Save results
    with open(join(train_dir, "c2st_koopman.csv"), "w") as f:
        w = csv.DictWriter(f, c2st_scores.keys())
        w.writeheader()
        w.writerow(c2st_scores)
    
    # Speed summary
    avg_time = np.mean(koopman_times)
    logger.info(f"Average Koopman sampling time: {avg_time:.3f} ± {np.std(koopman_times):.3f} seconds")
    
    # Speed comparison plot with sample information
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), koopman_times, 'bo-', label='Koopman Sampling Time')
    
    # Add sample count annotations
    for i, (obs, time_val) in enumerate(zip(range(1, 11), koopman_times)):
        # Get reference samples count for this observation
        ref_samples = task.get_reference_posterior_samples(num_observation=obs)
        sample_count = len(ref_samples) * 2  # We generate 2x samples
        ax.annotate(f'{sample_count}', (obs, time_val), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax.set_xlabel('Observation Number')
    ax.set_ylabel('Sampling Time (seconds)')
    ax.set_title('Koopman Sampling Speed (numbers show sample count generated)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(join(train_dir, "sampling_speed.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    if use_wandb:
        wandb.log(c2st_scores)
        wandb.log({"avg_sampling_time": avg_time})
        
    logger.info(f"Evaluation complete. Average C2ST: {np.mean(list(c2st_scores.values())):.3f}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Koopman SBI model')
    parser.add_argument('--train_dir', required=True,
                       help='Directory to save training outputs')
    parser.add_argument('--teacher_model_path', required=True,
                       help='Path to pre-trained teacher model')
    parser.add_argument('--settings_file', default=None,
                       help='Settings YAML file (if not in train_dir)')
    parser.add_argument('--device', default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--skip-c2st', action='store_true',
                       help='Skip C2ST computation during evaluation')
    
    args = parser.parse_args()
    
    # Load settings
    if args.settings_file:
        settings_path = args.settings_file
    else:
        settings_path = join(args.train_dir, "settings.yaml")
        
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    
    # Override teacher model path
    settings["model"]["teacher_model_path"] = args.teacher_model_path
    
    # Override device if specified
    if args.device:
        settings["training"]["device"] = args.device
    
    # Verify teacher model exists
    if not os.path.exists(args.teacher_model_path):
        logger.error(f"Teacher model not found: {args.teacher_model_path}")
        sys.exit(1)
    
    logger.info(f"Teacher model: {args.teacher_model_path}")
    logger.info(f"Settings: {settings_path}")
    logger.info(f"Output directory: {args.train_dir}")
    
    # Setup wandb if configured
    use_wandb = settings["training"].get("wandb", False)
    if use_wandb:
        wandb.init(
            config=settings, 
            dir=args.train_dir, 
            project="koopman-sbi",
            name=f"koopman-{settings['task']['name']}"
        )
    
    # Load or generate dataset (same as flow training)
    dataset_dir = os.path.join(args.train_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Try to load existing dataset first
    try:
        if os.path.exists(os.path.join(dataset_dir, 'x.npy')) and os.path.exists(os.path.join(dataset_dir, 'theta.npy')):
            logger.info(f"Loading existing dataset from {dataset_dir}")
            dataset = load_dataset(dataset_dir, settings)
        else:
            logger.info(f"Generating new dataset and saving to {dataset_dir}")
            dataset = generate_dataset(settings, directory_save=dataset_dir)
    except Exception as e:
        logger.warning(f"Failed to load dataset: {e}, generating new one")
        dataset = generate_dataset(settings, directory_save=dataset_dir)
    
    logger.info(f"Using dataset with {len(dataset)} samples")
    
    # Create data loaders
    train_loader, test_loader = build_train_and_test_loaders(
        dataset,
        settings["training"]["train_fraction"],
        settings["training"]["batch_size"],
        settings["training"]["num_workers"],
    )
    
    # Train model
    try:
        model = train_koopman_model(
            args.train_dir,
            settings=settings,
            train_loader=train_loader,
            test_loader=test_loader,
            use_wandb=use_wandb,
        )
        
        # Evaluate model
        evaluate_koopman_model(
            args.train_dir, 
            settings, 
            dataset, 
            model, 
            use_wandb=use_wandb,
            skip_c2st=args.skip_c2st
        )
        
        logger.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
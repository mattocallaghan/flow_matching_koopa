#!/usr/bin/env python3
"""
Generate Koopman Dataset from Pre-trained Teacher Model

This script generates a dataset of (θ₀, θ₁, x) triplets for Koopman training:
- θ₀: Random noise samples  
- θ₁: Samples from teacher posterior p(θ|x)
- x: Observation context

Usage:
    python generate_koopman_dataset.py --teacher_model_path teacher/best_model.pt --output_path koopman_dataset.pt --settings_file settings.yaml
"""
from torchdiffeq import odeint
import argparse
import logging
import os
import time
from pathlib import Path
import torch
import numpy as np
import yaml
from typing import Tuple, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from run_sbibm import generate_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_koopman_dataset(
    teacher_model_path: str,
    observations: torch.Tensor,
    buffer_size: int,
    samples_per_observation: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate Koopman training dataset using pre-trained teacher model
    
    Args:
        teacher_model_path: Path to trained teacher model
        observations: Observation samples [n_obs, x_dim]
        buffer_size: Total number of triplets to generate
        samples_per_observation: Number of (noise, flow_sample) pairs per observation
        device: Device to run on
        
    Returns:
        Tuple of (theta_0, theta_1, x_repeated) tensors
    """
    logger.info(f"Loading teacher model from {teacher_model_path}")
    model_load_start = time.time()
    
    # Load teacher model
    teacher_model = build_model_from_kwargs(
        filename=teacher_model_path,
        device=device
    )
    teacher_model.network.eval()
    
    model_load_time = time.time() - model_load_start
    logger.info(f"Teacher model loaded in {model_load_time:.2f} seconds")
    
    logger.info(f"Generating {buffer_size} triplets with {samples_per_observation} samples per observation")
    
    # Use ALL observations with samples_per_observation each
    n_observations = len(observations)
    expected_total = n_observations * samples_per_observation
    
    if expected_total != buffer_size:
        logger.warning(f"Expected {expected_total} samples ({n_observations} obs × {samples_per_observation} samples) but buffer_size={buffer_size}")
        logger.warning(f"Will generate {expected_total} samples total (ignoring buffer_size)")
    
    total_samples_needed = expected_total
    logger.info(f"Using ALL {n_observations} observations with {samples_per_observation} samples each = {total_samples_needed} total triplets")
    
    with torch.no_grad():
        observations_device = observations.to(device)
        
        # Simple and fast approach: create one big batch
        logger.info("Creating single large batch for maximum efficiency...")
        
        # Create all observation-sample pairs at once
        obs_repeated = observations_device.unsqueeze(1).repeat(1, samples_per_observation, 1)  # [n_obs, samples_per_obs, x_dim]
        obs_flat = obs_repeated.view(-1, obs_repeated.shape[-1])  # [n_obs * samples_per_obs, x_dim]
        logger.info(f"Generated observation batch of shape {obs_flat.shape}")

        # Single teacher model call for ALL samples
        start_time = time.time()

        # Process in smaller chunks if no batch method



        chunk_size = 10000
        theta_1_chunks = []
        theta_0_chunks = []
        total_chunks = (len(obs_flat) + chunk_size - 1) // chunk_size
        logger.info(f"Processing {len(obs_flat):,} samples in {total_chunks} chunks of {chunk_size:,}")
        
        for chunk_idx, i in enumerate(range(0, len(obs_flat), chunk_size)):
            chunk_start = time.time()
            chunk = obs_flat[i:i+chunk_size]
            
            # Generate noise samples
            noise_start = time.time()
            theta_0 = teacher_model.sample_theta_0(len(chunk))
            noise_time = time.time() - noise_start
            
            # Flow integration
            flow_start = time.time()
            _, theta_1 = odeint(
                lambda t, theta_t: teacher_model.evaluate_vectorfield(t, theta_t, chunk),
                theta_0,
                teacher_model.integration_range.to(device),
                atol=1e-7,
                rtol=1e-7,
                method="dopri5",
            )
            flow_time = time.time() - flow_start

            theta_1_chunks.append(theta_1)
            theta_0_chunks.append(theta_0)

            chunk_time = time.time() - chunk_start
            samples_processed = i + len(chunk)
            
            logger.info(f"Chunk {chunk_idx+1}/{total_chunks}: {len(chunk):,} samples, "
                       f"noise={noise_time:.2f}s, flow={flow_time:.2f}s, total={chunk_time:.2f}s "
                       f"({samples_processed:,}/{len(obs_flat):,} completed)")
        # Concatenate results
        concat_start = time.time()
        theta_1_all = torch.cat(theta_1_chunks, dim=0)
        theta_0_all = torch.cat(theta_0_chunks, dim=0)
        concat_time = time.time() - concat_start
        
        total_sampling_time = time.time() - start_time
        samples_per_sec = len(obs_flat) / total_sampling_time
        
        logger.info(f"Teacher sampling completed: {total_sampling_time:.2f}s total, "
                   f"concat={concat_time:.2f}s, {samples_per_sec:.1f} samples/sec")
        
        x_all = obs_flat

        # Move to CPU
        theta_0_all = theta_0_all.cpu()
        theta_1_all = theta_1_all.cpu()
        x_all = x_all.cpu()
    
    logger.info(f"Generated dataset shapes: theta_0={theta_0_all.shape}, theta_1={theta_1_all.shape}, x={x_all.shape}")
    
    return theta_0_all, theta_1_all, x_all


def plot_koopman_dataset(theta_0, theta_1, x, output_dir, original_dataset=None, n_samples_plot=10000):
    """
    Plot samples from the generated Koopman dataset to verify quality
    
    Args:
        theta_0: Noise samples [N, theta_dim]
        theta_1: Flow samples [N, theta_dim] 
        x: Observations [N, x_dim]
        output_dir: Directory to save plots
        original_dataset: Original SBIBM dataset for comparison
        n_samples_plot: Number of samples to plot
    """
    logger.info(f"Plotting Koopman dataset samples...")
    
    # Use ALL samples for plotting
    n_plot = len(theta_0)
    logger.info(f"Plotting ALL {n_plot} samples from the dataset")
    
    theta_0_plot = theta_0
    theta_1_plot = theta_1
    x_plot = x
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original training data for comparison (both standardized and unstandardized)
    original_theta = None
    original_x = None
    original_theta_unstd = None
    original_x_unstd = None
    
    if original_dataset is not None:
        original_theta_list = []
        original_x_list = []
        n_orig_plot = min(n_plot, len(original_dataset))
        for i in range(n_orig_plot):
            theta, x_obs = original_dataset[i]
            original_theta_list.append(theta)
            original_x_list.append(x_obs)
        original_theta = torch.stack(original_theta_list)  # Standardized
        original_x = torch.stack(original_x_list)  # Standardized
        
        # Also get unstandardized versions for comparison
        original_theta_unstd = original_dataset.standardize(original_theta, "theta", inverse=True)
        original_x_unstd = original_dataset.standardize(original_x, "x", inverse=True)
        
        # De-standardize the teacher output theta_1 for fair comparison
        theta_1_unstd = original_dataset.standardize(theta_1_plot, "theta", inverse=True)
        theta_0_unstd = original_dataset.standardize(theta_0_plot, "theta", inverse=True)
        x_unstd = original_dataset.standardize(x_plot, "x", inverse=True)
        
        # Debug: print ranges to understand the data
        logger.info("Data ranges for debugging:")
        logger.info(f"  original_theta (std): min={original_theta.min(dim=0)[0].tolist()}, max={original_theta.max(dim=0)[0].tolist()}")
        logger.info(f"  theta_1_plot (std): min={theta_1_plot.min(dim=0)[0].tolist()}, max={theta_1_plot.max(dim=0)[0].tolist()}")
        logger.info(f"  original_theta_unstd: min={original_theta_unstd.min(dim=0)[0].tolist()}, max={original_theta_unstd.max(dim=0)[0].tolist()}")
        logger.info(f"  theta_1_unstd: min={theta_1_unstd.min(dim=0)[0].tolist()}, max={theta_1_unstd.max(dim=0)[0].tolist()}")

    # 1. Plot theta_0 vs theta_1 distribution comparison (with original data)
    n_cols = 4 if original_dataset is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Theta_0 (noise) distribution
    axes[0].scatter(theta_0_plot[:, 0], theta_0_plot[:, 1], alpha=0.6, s=1, c='blue')
    axes[0].set_title(f'θ₀ (Noise) Distribution\nALL {n_plot:,} samples')
    axes[0].set_xlabel('θ₀[0]')
    axes[0].set_ylabel('θ₀[1]')
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].grid(True, alpha=0.3)
    
    # Theta_1 (flow output) distribution  
    axes[1].scatter(theta_1_plot[:, 0], theta_1_plot[:, 1], alpha=0.6, s=1, c='red')
    axes[1].set_title(f'θ₁ (Flow Output) Distribution\nALL {n_plot:,} samples')
    axes[1].set_xlabel('θ₁[0]')
    axes[1].set_ylabel('θ₁[1]')
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].grid(True, alpha=0.3)
    
    # Overlay comparison
    axes[2].scatter(theta_0_plot[:, 0], theta_0_plot[:, 1], alpha=0.4, s=1, c='blue', label='θ₀ (noise)')
    axes[2].scatter(theta_1_plot[:, 0], theta_1_plot[:, 1], alpha=0.4, s=1, c='red', label='θ₁ (flow)')
    axes[2].set_title(f'θ₀ vs θ₁ Comparison\nALL {n_plot:,} samples')
    axes[2].set_xlabel('θ[0]')
    axes[2].set_ylabel('θ[1]')
    axes[2].set_xlim(-5, 5)
    axes[2].set_ylim(-5, 5)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Original training data comparison (if available) - UNSTANDARDIZED  
    if original_dataset is not None:
        # Plot red (teacher) first so green (original) is on top
        axes[3].scatter(theta_1_unstd[:, 0], theta_1_unstd[:, 1], alpha=0.4, s=1, c='red', label='θ₁ (teacher output)')
        axes[3].scatter(original_theta_unstd[:, 0], original_theta_unstd[:, 1], alpha=0.4, s=1, c='green', label='Original θ (training)')
        axes[3].set_title(f'Original vs Teacher Output (Unstandardized)\nALL {n_plot:,} samples')
        axes[3].set_xlabel('θ[0]')
        axes[3].set_ylabel('θ[1]')
        # Adaptive limits based on data range
        all_theta_unstd = torch.cat([original_theta_unstd, theta_1_unstd], dim=0)
        margin = 0.1
        x_min, x_max = all_theta_unstd[:, 0].min().item(), all_theta_unstd[:, 0].max().item()
        y_min, y_max = all_theta_unstd[:, 1].min().item(), all_theta_unstd[:, 1].max().item()
        x_range = x_max - x_min
        y_range = y_max - y_min
        axes[3].set_xlim(x_min - margin*x_range, x_max + margin*x_range)
        axes[3].set_ylim(y_min - margin*y_range, y_max + margin*y_range)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "koopman_theta_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Plot observation distribution
    n_obs_cols = 2 if original_dataset is not None else 1
    fig, axes = plt.subplots(1, n_obs_cols, figsize=(8*n_obs_cols, 6))
    if n_obs_cols == 1:
        axes = [axes]  # Make it a list for consistent indexing
    
    axes[0].scatter(x_plot[:, 0], x_plot[:, 1], alpha=0.6, s=1, c='green')
    axes[0].set_title(f'Koopman Observation Distribution\nALL {n_plot:,} samples')
    axes[0].set_xlabel('x[0]')
    axes[0].set_ylabel('x[1]')
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].grid(True, alpha=0.3)
    
    # Original observations comparison (if available) - UNSTANDARDIZED
    if original_dataset is not None:
        # Plot green (koopman) first so blue (original) is on top  
        axes[1].scatter(x_unstd[:, 0], x_unstd[:, 1], alpha=0.4, s=1, c='green', label='Koopman x')
        axes[1].scatter(original_x_unstd[:, 0], original_x_unstd[:, 1], alpha=0.4, s=1, c='blue', label='Original x (training)')
        axes[1].set_title(f'Original vs Koopman Observations (Unstandardized)\nALL {n_plot:,} samples')
        axes[1].set_xlabel('x[0]')
        axes[1].set_ylabel('x[1]')
        # Adaptive limits for observations
        all_x_unstd = torch.cat([original_x_unstd, x_unstd], dim=0)
        margin = 0.1
        x_min, x_max = all_x_unstd[:, 0].min().item(), all_x_unstd[:, 0].max().item()
        y_min, y_max = all_x_unstd[:, 1].min().item(), all_x_unstd[:, 1].max().item()
        x_range = x_max - x_min
        y_range = y_max - y_min
        axes[1].set_xlim(x_min - margin*x_range, x_max + margin*x_range)
        axes[1].set_ylim(y_min - margin*y_range, y_max + margin*y_range)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "koopman_observations.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Teacher model validation plot - check if teacher samples match expected two_moons distribution
    if original_dataset is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Standardized comparison (as used by teacher model)
        # Plot red (teacher) first so green (original) is on top
        axes[0].scatter(theta_1_plot[:, 0], theta_1_plot[:, 1], alpha=0.4, s=1, c='red', label='Teacher θ₁ (standardized)')
        axes[0].scatter(original_theta[:, 0], original_theta[:, 1], alpha=0.4, s=1, c='green', label='Original θ (standardized)')
        axes[0].set_title(f'Teacher Model Check (Standardized Space)\nALL {n_plot:,} samples')
        axes[0].set_xlabel('θ[0] (standardized)')
        axes[0].set_ylabel('θ[1] (standardized)')
        axes[0].set_xlim(-3, 3)  # Standardized space typically ±3
        axes[0].set_ylim(-3, 3)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Unstandardized comparison (physical space)
        # Plot red (teacher) first so green (original) is on top
        axes[1].scatter(theta_1_unstd[:, 0], theta_1_unstd[:, 1], alpha=0.4, s=1, c='red', label='Teacher θ₁ (physical)')
        axes[1].scatter(original_theta_unstd[:, 0], original_theta_unstd[:, 1], alpha=0.4, s=1, c='green', label='Original θ (physical)')
        axes[1].set_title(f'Teacher Model Check (Physical Space)\nALL {n_plot:,} samples')
        axes[1].set_xlabel('θ[0] (physical)')
        axes[1].set_ylabel('θ[1] (physical)')
        # Adaptive limits for physical space
        all_phys = torch.cat([original_theta_unstd, theta_1_unstd], dim=0)
        margin = 0.1
        x_min, x_max = all_phys[:, 0].min().item(), all_phys[:, 0].max().item()
        y_min, y_max = all_phys[:, 1].min().item(), all_phys[:, 1].max().item()
        x_range = x_max - x_min
        y_range = y_max - y_min
        axes[1].set_xlim(x_min - margin*x_range, x_max + margin*x_range)
        axes[1].set_ylim(y_min - margin*y_range, y_max + margin*y_range)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "teacher_model_validation.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics for debugging
        logger.info("Teacher Model Validation:")
        logger.info(f"  Original θ (std): mean={original_theta.mean(dim=0).tolist()}, std={original_theta.std(dim=0).tolist()}")
        logger.info(f"  Teacher θ₁ (std): mean={theta_1_plot.mean(dim=0).tolist()}, std={theta_1_plot.std(dim=0).tolist()}")
        logger.info(f"  Original θ (phys): mean={original_theta_unstd.mean(dim=0).tolist()}, std={original_theta_unstd.std(dim=0).tolist()}")
        logger.info(f"  Teacher θ₁ (phys): mean={theta_1_unstd.mean(dim=0).tolist()}, std={theta_1_unstd.std(dim=0).tolist()}")
    
    # 4. Statistics summary
    logger.info("Koopman Dataset Statistics:")
    logger.info(f"  θ₀ mean: {theta_0.mean(dim=0).tolist()}")
    logger.info(f"  θ₀ std:  {theta_0.std(dim=0).tolist()}")
    logger.info(f"  θ₁ mean: {theta_1.mean(dim=0).tolist()}")
    logger.info(f"  θ₁ std:  {theta_1.std(dim=0).tolist()}")
    logger.info(f"  x mean:  {x.mean(dim=0).tolist()}")
    logger.info(f"  x std:   {x.std(dim=0).tolist()}")
    
    # Check for NaN/Inf values
    theta_0_bad = torch.any(~torch.isfinite(theta_0))
    theta_1_bad = torch.any(~torch.isfinite(theta_1))
    x_bad = torch.any(~torch.isfinite(x))
    
    if theta_0_bad or theta_1_bad or x_bad:
        logger.warning("⚠️  Found non-finite values in dataset!")
        logger.warning(f"  θ₀ has NaN/Inf: {theta_0_bad}")
        logger.warning(f"  θ₁ has NaN/Inf: {theta_1_bad}")
        logger.warning(f"  x has NaN/Inf: {x_bad}")
    else:
        logger.info("✓ All values are finite")
    
    logger.info(f"Dataset plots saved to {output_dir}/")
    
    return {
        'theta_0_stats': {'mean': theta_0.mean(dim=0), 'std': theta_0.std(dim=0)},
        'theta_1_stats': {'mean': theta_1.mean(dim=0), 'std': theta_1.std(dim=0)},
        'x_stats': {'mean': x.mean(dim=0), 'std': x.std(dim=0)},
        'has_nan': {'theta_0': theta_0_bad, 'theta_1': theta_1_bad, 'x': x_bad}
    }


def main():
    """Main function"""
    script_start = time.time()
    logger.info("=== KOOPMAN DATASET GENERATION STARTED ===")
    
    parser = argparse.ArgumentParser(description='Generate Koopman training dataset')
    parser.add_argument('--teacher_model_path', required=True,
                       help='Path to trained teacher model')
    parser.add_argument('--output_path', required=True,
                       help='Path to save generated dataset')
    parser.add_argument('--settings_file', required=True,
                       help='Settings file for dataset generation')
    parser.add_argument('--device', default='cpu',
                       help='Device to run on (cpu/cuda)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots of the dataset')
    
    args = parser.parse_args()
    
    # Load settings
    with open(args.settings_file, 'r') as f:
        settings = yaml.safe_load(f)
    
    # Extract parameters
    buffer_size = settings.get('koopman', {}).get('buffer_size', 10000)
    samples_per_observation = settings.get('koopman', {}).get('samples_per_observation', 20)
    
    logger.info(f"Dataset generation parameters:")
    logger.info(f"  Buffer size: {buffer_size}")
    logger.info(f"  Samples per observation: {samples_per_observation}")
    logger.info(f"  Device: {args.device}")
    
    # Generate original dataset to get observations
    logger.info("Generating original SBIBM dataset for observations")
    dataset_gen_start = time.time()
    original_dataset = generate_dataset(settings)
    dataset_gen_time = time.time() - dataset_gen_start
    logger.info(f"Original dataset generated in {dataset_gen_time:.2f} seconds")
    
    # Extract observations (standardized)
    observations = []
    for i in range(len(original_dataset)):
        _, x = original_dataset[i]
        observations.append(x)
    observations = torch.stack(observations)
    
    logger.info(f"Using {len(observations)} observations from SBIBM dataset")
    
    # Generate Koopman dataset
    logger.info("Starting Koopman dataset generation...")
    koopman_gen_start = time.time()
    theta_0, theta_1, x = generate_koopman_dataset(
        teacher_model_path=args.teacher_model_path,
        observations=observations,
        buffer_size=buffer_size,
        samples_per_observation=samples_per_observation,
        device=args.device
    )
    koopman_gen_time = time.time() - koopman_gen_start
    logger.info(f"Koopman dataset generation completed in {koopman_gen_time:.2f} seconds")
    
    # Plot dataset for inspection if requested
    if args.plot:
        logger.info("Generating diagnostic plots...")
        plot_start = time.time()
        output_dir = Path(args.output_path).parent
        plot_stats = plot_koopman_dataset(theta_0, theta_1, x, output_dir / "dataset_plots", original_dataset)
        plot_time = time.time() - plot_start
        logger.info(f"Diagnostic plots generated in {plot_time:.2f} seconds")
    
    print("Generated Koopman dataset:\n")
    print(f"  theta_0 shape: {theta_0.shape}")
    print(f"  theta_1 shape: {theta_1.shape}")
    print(f"  x shape: {x.shape}")

    # Save dataset
    dataset = {
        'theta_0': theta_0,
        'theta_1': theta_1, 
        'x': x,
        'buffer_size': buffer_size,
        'samples_per_observation': samples_per_observation,
        'standardization': original_dataset.standardization  # Include standardization info
    }
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    save_start = time.time()
    torch.save(dataset, output_path)
    save_time = time.time() - save_start
    
    total_time = time.time() - script_start
    
    # Final summary
    logger.info(f"Saved Koopman dataset to {output_path} in {save_time:.2f} seconds")
    logger.info(f"Dataset contains {len(theta_0):,} triplets")
    logger.info("=== TIMING SUMMARY ===")
    logger.info(f"  Dataset generation: {dataset_gen_time:.2f}s")
    logger.info(f"  Koopman generation: {koopman_gen_time:.2f}s")
    if args.plot:
        logger.info(f"  Plot generation:    {plot_time:.2f}s")
    logger.info(f"  Dataset saving:     {save_time:.2f}s")
    logger.info(f"  TOTAL TIME:         {total_time:.2f}s")
    logger.info(f"  Generation rate:    {len(theta_0)/koopman_gen_time:.1f} triplets/sec")
    logger.info("=== KOOPMAN DATASET GENERATION COMPLETED ===") 


if __name__ == "__main__":
    main()
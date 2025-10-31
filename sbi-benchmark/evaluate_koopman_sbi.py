#!/usr/bin/env python3
"""
Evaluation Script for Koopman SBI Models

This script provides comprehensive evaluation and comparison of Koopman models
against their teacher models using SBI benchmark metrics. It evaluates both
sample quality (C2ST scores) and sampling speed.

Usage:
    # Evaluate trained Koopman model
    python evaluate_koopman_sbi.py --koopman_model_path koopman_two_moons/best_model.pt \
                                   --teacher_model_path teacher_two_moons/best_model.pt \
                                   --output_dir evaluation_results
                                   
    # Quick evaluation (fewer samples)
    python evaluate_koopman_sbi.py --koopman_model_path koopman_two_moons/best_model.pt \
                                   --quick

Author: SBI-compatible Koopman evaluation
"""

import argparse
import csv
import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sbibm.tasks
from sbibm.metrics import c2st, mmd, posterior_mean_error
import torch
import numpy as np
import pandas as pd

# Import our components
from register_koopman_model import register_koopman_model
from run_sbibm import generate_dataset, SbiDataset

try:
    from dingo.core.posterior_models.build_model import build_model_from_kwargs
    DINGO_AVAILABLE = True
except ImportError:
    DINGO_AVAILABLE = False
    print("Warning: Dingo not available. Limited functionality.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register Koopman model
register_koopman_model()


class KoopmanSBIEvaluator:
    """
    Comprehensive evaluator for Koopman SBI models
    """
    
    def __init__(self,
                 koopman_model_path: str,
                 teacher_model_path: Optional[str] = None,
                 task_name: str = "two_moons",
                 device: str = None):
        """
        Initialize evaluator
        
        Args:
            koopman_model_path: Path to trained Koopman model
            teacher_model_path: Path to teacher model (for comparison)
            task_name: SBI task name
            device: Device to run on
        """
        self.koopman_model_path = koopman_model_path
        self.teacher_model_path = teacher_model_path
        self.task_name = task_name
        
        # Device setup
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load SBI task
        self.task = sbibm.get_task(task_name)
        logger.info(f"Loaded SBI task: {task_name}")
        
        # Load models
        self.koopman_model = self.load_koopman_model()
        self.teacher_model = self.load_teacher_model() if teacher_model_path else None
        
        # Create dataset for standardization
        self.dataset = self.create_evaluation_dataset()
        
    def load_koopman_model(self):
        """Load trained Koopman model"""
        logger.info(f"Loading Koopman model from {self.koopman_model_path}")
        
        if not os.path.exists(self.koopman_model_path):
            raise FileNotFoundError(f"Koopman model not found: {self.koopman_model_path}")
            
        # Load checkpoint
        checkpoint = torch.load(self.koopman_model_path, map_location=self.device)
        
        # Extract model configuration from metadata
        if 'metadata' in checkpoint and 'train_settings' in checkpoint['metadata']:
            model_config = checkpoint['metadata']['train_settings']['model']
        else:
            # Fallback configuration
            logger.warning("No metadata found, using default configuration")
            model_config = {
                'type': 'koopman_sbi',
                'lifted_dim': 64,
                'lambda_phase': 1.0,
                'lambda_target': 1.0,
                'lambda_recon': 1.0,
                'lambda_cons': 0.1,
                'buffer_size': 5000,
                'input_dim': 2,  # two_moons
                'context_dim': 2
            }
        
        # Create model using factory function
        from register_koopman_model import create_koopman_sbi_model
        
        settings = {'model': model_config}
        model = create_koopman_sbi_model(settings, device=self.device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info("Loaded Koopman model via factory function")
        return model
            
    def load_teacher_model(self):
        """Load teacher model for comparison"""
        if not self.teacher_model_path:
            return None
            
        logger.info(f"Loading teacher model from {self.teacher_model_path}")
        
        if not os.path.exists(self.teacher_model_path):
            logger.warning(f"Teacher model not found: {self.teacher_model_path}")
            return None
            
        if DINGO_AVAILABLE:
            try:
                model = build_model_from_kwargs(
                    filename=self.teacher_model_path,
                    device=self.device
                )
                model.eval()
                logger.info("Loaded teacher model")
                return model
            except Exception as e:
                logger.warning(f"Failed to load teacher model: {e}")
                return None
        else:
            logger.warning("Dingo not available, cannot load teacher model")
            return None
            
    def create_evaluation_dataset(self) -> SbiDataset:
        """Create small dataset for standardization"""
        prior = self.task.get_prior()
        simulator = self.task.get_simulator()
        
        # Generate small dataset for standardization
        theta_samples = prior(1000)
        x_samples = simulator(theta_samples)
        
        theta_tensor = torch.tensor(theta_samples, dtype=torch.float)
        x_tensor = torch.tensor(x_samples, dtype=torch.float)
        
        return SbiDataset(theta_tensor, x_tensor)
        
    def benchmark_sampling_speed(self, 
                                n_samples: int = 1000,
                                n_observations: int = 5,
                                n_trials: int = 3) -> Dict[str, Any]:
        """
        Benchmark sampling speed for both models
        
        Args:
            n_samples: Number of samples per observation
            n_observations: Number of different observations to test
            n_trials: Number of timing trials per observation
            
        Returns:
            Dictionary with timing results
        """
        logger.info(f"Benchmarking sampling speed: {n_samples} samples, {n_observations} obs, {n_trials} trials")
        
        results = {
            'koopman_times': [],
            'teacher_times': [],
            'observations': []
        }
        
        for obs_idx in range(1, n_observations + 1):
            observation = self.task.get_observation(num_observation=obs_idx)
            observation_std = self.dataset.standardize(observation, label="x")
            
            results['observations'].append(obs_idx)
            
            # Benchmark Koopman model
            koopman_times = []
            for trial in range(n_trials):
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start_time = time.time()
                
                with torch.no_grad():
                    observation_batch = observation_std.repeat((n_samples, 1))
                    samples = self.koopman_model.sample_batch(observation_batch)
                    
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end_time = time.time()
                
                koopman_times.append(end_time - start_time)
                
            koopman_avg = np.mean(koopman_times)
            results['koopman_times'].append(koopman_avg)
            
            # Benchmark teacher model (if available)
            if self.teacher_model is not None:
                teacher_times = []
                for trial in range(n_trials):
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        observation_batch = observation_std.repeat((n_samples, 1))
                        samples = self.teacher_model.sample_batch(observation_batch)
                        
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    end_time = time.time()
                    
                    teacher_times.append(end_time - start_time)
                    
                teacher_avg = np.mean(teacher_times)
                results['teacher_times'].append(teacher_avg)
            else:
                results['teacher_times'].append(None)
                
        # Compute summary statistics
        results['koopman_mean'] = np.mean(results['koopman_times'])
        results['koopman_std'] = np.std(results['koopman_times'])
        
        if self.teacher_model is not None and all(t is not None for t in results['teacher_times']):
            results['teacher_mean'] = np.mean(results['teacher_times'])
            results['teacher_std'] = np.std(results['teacher_times'])
            results['speedup'] = results['teacher_mean'] / results['koopman_mean']
        else:
            results['teacher_mean'] = None
            results['teacher_std'] = None
            results['speedup'] = None
            
        return results
        
    def evaluate_sample_quality(self, 
                               n_samples: int = 2000,
                               n_observations: int = 10) -> Dict[str, Any]:
        """
        Evaluate sample quality using SBI metrics
        
        Args:
            n_samples: Number of samples to generate per observation
            n_observations: Number of observations to evaluate
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info(f"Evaluating sample quality: {n_samples} samples, {n_observations} observations")
        
        results = {
            'c2st_scores': [],
            'mmd_scores': [],
            'mean_errors': [],
            'observations': [],
            'koopman_samples': [],
            'reference_samples': [],
            'teacher_samples': []
        }
        
        for obs_idx in range(1, n_observations + 1):
            logger.info(f"Evaluating observation {obs_idx}")
            
            # Get reference samples
            reference_samples = self.task.get_reference_posterior_samples(num_observation=obs_idx)
            
            # Get observation
            observation = self.task.get_observation(num_observation=obs_idx)
            observation_std = self.dataset.standardize(observation, label="x")
            
            # Generate Koopman samples
            with torch.no_grad():
                observation_batch = observation_std.repeat((n_samples * 2, 1))  # *2 for prior filtering
                koopman_samples = self.koopman_model.sample_batch(observation_batch)
                koopman_samples = self.dataset.standardize(koopman_samples, label="theta", inverse=True)
            
            # Filter samples within prior
            prior_mask = torch.isfinite(self.task.prior_dist.log_prob(koopman_samples))
            koopman_samples_filtered = koopman_samples[prior_mask]
            
            # Generate teacher samples (if available)
            teacher_samples_filtered = None
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_samples = self.teacher_model.sample_batch(observation_batch)
                    teacher_samples = self.dataset.standardize(teacher_samples, label="theta", inverse=True)
                    
                teacher_prior_mask = torch.isfinite(self.task.prior_dist.log_prob(teacher_samples))
                teacher_samples_filtered = teacher_samples[teacher_prior_mask]
            
            # Ensure we have enough samples
            n_eval = min(len(reference_samples), len(koopman_samples_filtered))
            if teacher_samples_filtered is not None:
                n_eval = min(n_eval, len(teacher_samples_filtered))
                
            if n_eval < 100:
                logger.warning(f"Only {n_eval} valid samples for observation {obs_idx}")
                continue
                
            # Truncate to same size
            ref_samples = reference_samples[:n_eval]
            koop_samples = koopman_samples_filtered[:n_eval]
            
            # Compute metrics
            c2st_score = c2st(koop_samples, ref_samples)
            mmd_score = mmd(koop_samples, ref_samples)
            mean_error = posterior_mean_error(koop_samples, ref_samples)
            
            results['c2st_scores'].append(c2st_score.item())
            results['mmd_scores'].append(mmd_score.item())
            results['mean_errors'].append(mean_error.item())
            results['observations'].append(obs_idx)
            
            # Store samples for plotting (first 3 observations only)
            if obs_idx <= 3:
                results['koopman_samples'].append(koop_samples.cpu().numpy())
                results['reference_samples'].append(ref_samples.cpu().numpy())
                if teacher_samples_filtered is not None:
                    teacher_eval = teacher_samples_filtered[:n_eval]
                    results['teacher_samples'].append(teacher_eval.cpu().numpy())
                else:
                    results['teacher_samples'].append(None)
                    
        # Compute summary statistics
        if results['c2st_scores']:
            results['c2st_mean'] = np.mean(results['c2st_scores'])
            results['c2st_std'] = np.std(results['c2st_scores'])
            results['mmd_mean'] = np.mean(results['mmd_scores'])
            results['mmd_std'] = np.std(results['mmd_scores'])
            results['mean_error_mean'] = np.mean(results['mean_errors'])
            results['mean_error_std'] = np.std(results['mean_errors'])
        else:
            logger.warning("No valid quality evaluations completed")
            
        return results
        
    def create_plots(self, speed_results: Dict, quality_results: Dict, output_dir: Path):
        """Create evaluation plots"""
        
        # 1. Speed comparison plot
        if speed_results['speedup'] is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Sampling times
            obs_nums = speed_results['observations']
            ax1.plot(obs_nums, speed_results['koopman_times'], 'bo-', label='Koopman')
            ax1.plot(obs_nums, speed_results['teacher_times'], 'ro-', label='Teacher')
            ax1.set_xlabel('Observation Number')
            ax1.set_ylabel('Sampling Time (seconds)')
            ax1.set_title('Sampling Speed Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Speedup
            speedups = [t/k for t, k in zip(speed_results['teacher_times'], speed_results['koopman_times'])]
            ax2.bar(obs_nums, speedups, color='green', alpha=0.7)
            ax2.set_xlabel('Observation Number')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title(f'Speedup (Avg: {speed_results["speedup"]:.1f}x)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'speed_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        # 2. Quality metrics plot
        if quality_results.get('c2st_scores'):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            obs_nums = quality_results['observations']
            
            # C2ST scores
            axes[0].plot(obs_nums, quality_results['c2st_scores'], 'bo-')
            axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random')
            axes[0].set_xlabel('Observation Number')
            axes[0].set_ylabel('C2ST Score')
            axes[0].set_title('Sample Quality (C2ST)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # MMD scores
            axes[1].plot(obs_nums, quality_results['mmd_scores'], 'go-')
            axes[1].set_xlabel('Observation Number')
            axes[1].set_ylabel('MMD Score')
            axes[1].set_title('Maximum Mean Discrepancy')
            axes[1].grid(True, alpha=0.3)
            
            # Mean errors
            axes[2].plot(obs_nums, quality_results['mean_errors'], 'mo-')
            axes[2].set_xlabel('Observation Number')
            axes[2].set_ylabel('Posterior Mean Error')
            axes[2].set_title('Mean Error')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'quality_metrics.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        # 3. Posterior comparison plots
        for i, (koop_samples, ref_samples, teacher_samples) in enumerate(zip(
            quality_results['koopman_samples'],
            quality_results['reference_samples'], 
            quality_results['teacher_samples']
        )):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot samples
            ax.scatter(koop_samples[:, 0], koop_samples[:, 1], 
                      s=1, alpha=0.6, label='Koopman', color='blue')
            ax.scatter(ref_samples[:, 0], ref_samples[:, 1],
                      s=1, alpha=0.6, label='Reference', color='red')
            
            if teacher_samples is not None:
                ax.scatter(teacher_samples[:, 0], teacher_samples[:, 1],
                          s=1, alpha=0.4, label='Teacher', color='orange')
                          
            ax.set_xlabel('θ₁')
            ax.set_ylabel('θ₂')
            ax.set_title(f'Posterior Samples - Observation {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(output_dir / f'posterior_comparison_{i+1}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        logger.info(f"Saved plots to {output_dir}")
        
    def run_full_evaluation(self, 
                           output_dir: str,
                           n_samples: int = 2000,
                           quick: bool = False) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            output_dir: Output directory for results
            n_samples: Number of samples for evaluation
            quick: Whether to run quick evaluation
            
        Returns:
            Complete evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting comprehensive Koopman evaluation")
        
        if quick:
            n_samples = min(n_samples, 500)
            n_obs_speed = 3
            n_obs_quality = 5
            n_trials = 1
        else:
            n_obs_speed = 5
            n_obs_quality = 10
            n_trials = 3
            
        # 1. Speed benchmark
        logger.info("Running speed benchmark...")
        speed_results = self.benchmark_sampling_speed(
            n_samples=n_samples, 
            n_observations=n_obs_speed,
            n_trials=n_trials
        )
        
        # 2. Quality evaluation
        logger.info("Evaluating sample quality...")
        quality_results = self.evaluate_sample_quality(
            n_samples=n_samples,
            n_observations=n_obs_quality
        )
        
        # 3. Create plots
        logger.info("Creating evaluation plots...")
        self.create_plots(speed_results, quality_results, output_path)
        
        # 4. Save results
        results_summary = {
            'task': self.task_name,
            'koopman_model': self.koopman_model_path,
            'teacher_model': self.teacher_model_path,
            'speed_results': speed_results,
            'quality_results': {
                'c2st_mean': quality_results.get('c2st_mean'),
                'c2st_std': quality_results.get('c2st_std'),
                'mmd_mean': quality_results.get('mmd_mean'),
                'mmd_std': quality_results.get('mmd_std'),
                'mean_error_mean': quality_results.get('mean_error_mean'),
                'mean_error_std': quality_results.get('mean_error_std')
            },
            'evaluation_params': {
                'n_samples': n_samples,
                'n_observations_speed': n_obs_speed,
                'n_observations_quality': n_obs_quality,
                'quick_mode': quick
            }
        }
        
        # Save JSON summary
        with open(output_path / 'evaluation_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
            
        # Save CSV with detailed results
        if quality_results.get('c2st_scores'):
            df = pd.DataFrame({
                'observation': quality_results['observations'],
                'c2st_score': quality_results['c2st_scores'],
                'mmd_score': quality_results['mmd_scores'],
                'mean_error': quality_results['mean_errors']
            })
            df.to_csv(output_path / 'detailed_quality_results.csv', index=False)
            
        # Print summary
        self.print_evaluation_summary(results_summary)
        
        return results_summary
        
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary to console"""
        
        print("\n" + "="*60)
        print("KOOPMAN MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Task: {results['task']}")
        print(f"Koopman Model: {results['koopman_model']}")
        if results['teacher_model']:
            print(f"Teacher Model: {results['teacher_model']}")
            
        # Speed results
        speed = results['speed_results']
        print(f"\nSAMPLING SPEED:")
        print(f"  Koopman: {speed['koopman_mean']:.4f} ± {speed['koopman_std']:.4f} seconds")
        if speed['teacher_mean'] is not None:
            print(f"  Teacher:  {speed['teacher_mean']:.4f} ± {speed['teacher_std']:.4f} seconds")
            print(f"  Speedup:  {speed['speedup']:.1f}x faster")
        else:
            print("  Teacher: Not available")
            
        # Quality results
        quality = results['quality_results']
        if quality['c2st_mean'] is not None:
            print(f"\nSAMPLE QUALITY:")
            print(f"  C2ST Score: {quality['c2st_mean']:.3f} ± {quality['c2st_std']:.3f}")
            print(f"  MMD Score:  {quality['mmd_mean']:.4f} ± {quality['mmd_std']:.4f}")
            print(f"  Mean Error: {quality['mean_error_mean']:.4f} ± {quality['mean_error_std']:.4f}")
            
            # Quality interpretation
            if quality['c2st_mean'] < 0.6:
                print("  → Excellent sample quality!")
            elif quality['c2st_mean'] < 0.7:
                print("  → Good sample quality")
            else:
                print("  → Sample quality could be improved")
        else:
            print("\nSAMPLE QUALITY: Evaluation failed")
            
        print("="*60)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Koopman SBI model')
    parser.add_argument('--koopman_model_path', required=True,
                       help='Path to trained Koopman model')
    parser.add_argument('--teacher_model_path', default=None,
                       help='Path to teacher model (for comparison)')
    parser.add_argument('--output_dir', default='koopman_evaluation',
                       help='Output directory for results')
    parser.add_argument('--task', default='two_moons',
                       help='SBI task name')
    parser.add_argument('--device', default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='Number of samples for evaluation')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick evaluation with fewer samples')
    
    args = parser.parse_args()
    
    # Verify Koopman model exists
    if not os.path.exists(args.koopman_model_path):
        logger.error(f"Koopman model not found: {args.koopman_model_path}")
        sys.exit(1)
        
    # Create evaluator
    evaluator = KoopmanSBIEvaluator(
        koopman_model_path=args.koopman_model_path,
        teacher_model_path=args.teacher_model_path,
        task_name=args.task,
        device=args.device
    )
    
    # Run evaluation
    try:
        results = evaluator.run_full_evaluation(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            quick=args.quick
        )
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
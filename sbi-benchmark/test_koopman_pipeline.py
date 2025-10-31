#!/usr/bin/env python3
"""
Complete Pipeline Test for Koopman SBI Implementation

This script tests the complete Koopman pipeline:
1. Train teacher flow matching model on two_moons
2. Train Koopman student model using teacher
3. Evaluate and compare both models
4. Generate comprehensive report

Usage:
    python test_koopman_pipeline.py --output_base test_results --quick

Author: Complete Koopman SBI pipeline test
"""

import argparse
import os
import sys
import subprocess
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, Any

import yaml
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KoopmanPipelineTester:
    """
    Complete pipeline tester for Koopman SBI implementation
    """
    
    def __init__(self, output_base: str, quick: bool = False):
        """
        Initialize pipeline tester
        
        Args:
            output_base: Base directory for all outputs
            quick: Whether to run in quick mode (smaller datasets, fewer epochs)
        """
        self.output_base = Path(output_base)
        self.quick = quick
        self.use_pretrained_teacher = None
        self.regenerate_koopman_dataset = False
        
        # Create output directories
        self.teacher_dir = self.output_base / "teacher_two_moons"
        self.koopman_dir = self.output_base / "koopman_two_moons"
        self.eval_dir = self.output_base / "evaluation"
        
        for dir_path in [self.teacher_dir, self.koopman_dir, self.eval_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized pipeline tester with output base: {self.output_base}")
        if quick:
            logger.info("Running in QUICK mode (reduced datasets and epochs)")
            
    def create_teacher_settings(self) -> str:
        """
        Create settings file for teacher flow matching model
        
        Returns:
            Path to created settings file
        """
        # Load settings from existing settings.yaml file
        with open('settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
            
        settings_path = self.teacher_dir / "settings.yaml"
        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)
            
        logger.info(f"Created teacher settings: {settings_path}")
        return str(settings_path)
        
    def create_koopman_settings(self, teacher_model_path: str) -> str:
        """
        Create settings file for Koopman student model
        
        Args:
            teacher_model_path: Path to trained teacher model
            
        Returns:
            Path to created settings file
        """
        # Load settings from existing koopman_two_moons_settings.yaml file
        with open('koopman_two_moons_settings.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        
        # Set teacher model path
        settings['model']['teacher_model_path'] = teacher_model_path
            
        settings_path = self.koopman_dir / "settings.yaml"
        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False)
            
        logger.info(f"Created Koopman settings: {settings_path}")
        return str(settings_path)
        
    def run_command(self, cmd: list, description: str, cwd: str = None) -> bool:
        """
        Run a command and handle errors
        
        Args:
            cmd: Command to run as list
            description: Description for logging
            cwd: Working directory
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running {description}...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=cwd
            )
            logger.info(f"{description} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"{description} failed with return code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"{description} failed with error: {e}")
            return False
            
    def step1_train_teacher(self) -> bool:
        """
        Step 1: Train teacher flow matching model
        
        Returns:
            True if successful
        """
        logger.info("\n" + "="*50)
        logger.info("STEP 1: Training Teacher Flow Matching Model")
        logger.info("="*50)
        
        # Create settings
        settings_path = self.create_teacher_settings()
        
        # Run training
        cmd = [
            sys.executable, 'run_sbibm.py',
            '--train_dir', str(self.teacher_dir)
        ]
        
        success = self.run_command(cmd, "teacher training", cwd='.')
        
        if success:
            # Verify model was created
            model_path = self.teacher_dir / "best_model.pt"
            if model_path.exists():
                logger.info(f"Teacher model saved to: {model_path}")
                return True
            else:
                logger.error("Teacher model file not found after training")
                return False
        else:
            return False
            
    def step2_generate_koopman_dataset(self, teacher_model_path: str) -> bool:
        """
        Step 2: Generate Koopman dataset from teacher model
        
        Args:
            teacher_model_path: Path to trained teacher model
            
        Returns:
            True if successful
        """
        logger.info("\\n" + "="*50)
        logger.info("STEP 2: Generating Koopman Dataset")
        logger.info("="*50)
        
        # Check if dataset already exists
        dataset_path = self.koopman_dir / "koopman_dataset.pt"
        if dataset_path.exists() and not self.regenerate_koopman_dataset:
            logger.info(f"Koopman dataset already exists at {dataset_path}")
            logger.info("Use --regenerate-koopman-dataset to force regeneration")
            return True
        
        # Create Koopman settings for dataset generation
        koopman_settings_path = self.create_koopman_settings(teacher_model_path)
        
        # Run dataset generation
        cmd = [
            sys.executable, 'generate_koopman_dataset.py',
            '--teacher_model_path', teacher_model_path,
            '--output_path', str(dataset_path),
            '--settings_file', koopman_settings_path,
            '--plot'  # Generate diagnostic plots
        ]
        
        success = self.run_command(cmd, "Koopman dataset generation", cwd='.')
        
        if success:
            if dataset_path.exists():
                logger.info(f"Koopman dataset saved to: {dataset_path}")
                return True
            else:
                logger.error("Dataset file not found after generation")
                return False
        else:
            return False
            
    def step3_train_koopman(self, teacher_model_path: str) -> bool:
        """
        Step 3: Train Koopman student model
        
        Args:
            teacher_model_path: Path to trained teacher model
            
        Returns:
            True if successful
        """
        logger.info("\n" + "="*50)
        logger.info("STEP 3: Training Koopman Student Model")
        logger.info("="*50)
        
        # Create settings
        settings_path = self.create_koopman_settings(teacher_model_path)
        
        # Run training
        cmd = [
            sys.executable, 'train_koopman_sbi.py',
            '--train_dir', str(self.koopman_dir),
            '--teacher_model_path', teacher_model_path,
            '--settings_file', settings_path,
            '--skip-c2st'  # Skip C2ST to avoid NaN errors
        ]
        
        success = self.run_command(cmd, "Koopman training", cwd='.')
        
        if success:
            # Verify model was created
            model_path = self.koopman_dir / "best_model.pt"
            if model_path.exists():
                logger.info(f"Koopman model saved to: {model_path}")
                return True
            else:
                logger.error("Koopman model file not found after training")
                return False
        else:
            return False
            
    def step3_evaluate(self, koopman_model_path: str, teacher_model_path: str) -> bool:
        """
        Step 3: Evaluate and compare models
        
        Args:
            koopman_model_path: Path to trained Koopman model
            teacher_model_path: Path to teacher model
            
        Returns:
            True if successful
        """
        logger.info("\n" + "="*50)
        logger.info("STEP 3: Evaluating and Comparing Models")
        logger.info("="*50)
        
        # Run evaluation
        cmd = [
            sys.executable, 'evaluate_koopman_sbi.py',
            '--koopman_model_path', koopman_model_path,
            '--teacher_model_path', teacher_model_path,
            '--output_dir', str(self.eval_dir),
            '--task', 'two_moons'
        ]
        
        if self.quick:
            cmd.append('--quick')
            
        success = self.run_command(cmd, "model evaluation", cwd='.')
        
        if success:
            # Verify evaluation results exist
            results_path = self.eval_dir / "evaluation_summary.json"
            if results_path.exists():
                logger.info(f"Evaluation results saved to: {results_path}")
                return True
            else:
                logger.error("Evaluation results not found")
                return False
        else:
            return False
            
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate final pipeline test report
        
        Returns:
            Report dictionary
        """
        logger.info("\n" + "="*50)
        logger.info("GENERATING PIPELINE TEST REPORT")
        logger.info("="*50)
        
        report = {
            'pipeline_test': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'quick_mode': self.quick,
                'output_base': str(self.output_base)
            },
            'steps': {
                'teacher_training': {},
                'koopman_training': {},
                'evaluation': {}
            },
            'summary': {}
        }
        
        # Check if all outputs exist
        teacher_model = self.teacher_dir / "best_model.pt"
        koopman_model = self.koopman_dir / "best_model.pt"
        eval_results = self.eval_dir / "evaluation_summary.json"
        
        report['steps']['teacher_training']['completed'] = teacher_model.exists()
        report['steps']['koopman_training']['completed'] = koopman_model.exists()
        report['steps']['evaluation']['completed'] = eval_results.exists()
        
        # Load evaluation results if available
        if eval_results.exists():
            try:
                with open(eval_results, 'r') as f:
                    eval_data = json.load(f)
                report['evaluation_results'] = eval_data
                
                # Extract key metrics
                speed_results = eval_data.get('speed_results', {})
                quality_results = eval_data.get('quality_results', {})
                
                report['summary']['speedup'] = speed_results.get('speedup')
                report['summary']['koopman_time'] = speed_results.get('koopman_mean')
                report['summary']['teacher_time'] = speed_results.get('teacher_mean')
                report['summary']['c2st_score'] = quality_results.get('c2st_mean')
                report['summary']['mmd_score'] = quality_results.get('mmd_mean')
                
            except Exception as e:
                logger.error(f"Failed to load evaluation results: {e}")
                
        # Save report
        report_path = self.output_base / "pipeline_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Pipeline test report saved to: {report_path}")
        
        return report
        
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final summary to console"""
        
        print("\n" + "="*70)
        print("KOOPMAN SBI PIPELINE TEST SUMMARY")
        print("="*70)
        
        print(f"Test Mode: {'QUICK' if self.quick else 'FULL'}")
        print(f"Output Directory: {self.output_base}")
        
        # Step completion
        steps = report['steps']
        print(f"\nSTEP COMPLETION:")
        print(f"  Teacher Training: {'✓' if steps['teacher_training']['completed'] else '✗'}")
        print(f"  Koopman Training: {'✓' if steps['koopman_training']['completed'] else '✗'}")
        print(f"  Evaluation:       {'✓' if steps['evaluation']['completed'] else '✗'}")
        
        # Performance summary
        summary = report.get('summary', {})
        if summary:
            print(f"\nPERFORMANCE SUMMARY:")
            
            if summary.get('speedup'):
                print(f"  Speedup: {summary['speedup']:.1f}x faster than teacher")
                print(f"  Koopman Time: {summary['koopman_time']:.4f} seconds")
                print(f"  Teacher Time:  {summary['teacher_time']:.4f} seconds")
            else:
                print(f"  Speedup: Not available")
                
            if summary.get('c2st_score'):
                print(f"  Sample Quality (C2ST): {summary['c2st_score']:.3f}")
                if summary['c2st_score'] < 0.6:
                    print(f"    → Excellent quality!")
                elif summary['c2st_score'] < 0.7:
                    print(f"    → Good quality")
                else:
                    print(f"    → Needs improvement")
                    
        all_completed = all(step['completed'] for step in steps.values())
        
        if all_completed:
            print(f"\n🎉 PIPELINE TEST SUCCESSFUL! 🎉")
            speedup = summary.get('speedup')
            if speedup is not None and speedup > 5:
                print(f"   Koopman model achieves significant speedup!")
        else:
            print(f"\n❌ PIPELINE TEST INCOMPLETE")
            print(f"   Check logs for details on failed steps")
            
        print("="*70)
        
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete pipeline test
        
        Returns:
            True if all steps completed successfully
        """
        start_time = time.time()
        
        logger.info("Starting complete Koopman SBI pipeline test")
        
        try:
            # Step 1: Train teacher (or use pre-trained)
            logger.info(f"DEBUG: self.use_pretrained_teacher = {self.use_pretrained_teacher}")
            if self.use_pretrained_teacher:
                logger.info(f"Using pre-trained teacher model: {self.use_pretrained_teacher}")
                teacher_model_path = self.use_pretrained_teacher
                # Verify the pre-trained model exists
                if not os.path.exists(teacher_model_path):
                    logger.error(f"Pre-trained teacher model not found: {teacher_model_path}")
                    return False
                logger.info("SKIPPING teacher training - using pre-trained model")
            else:
                logger.info("No pre-trained teacher specified - training from scratch")
                if not self.step1_train_teacher():
                    logger.error("Teacher training failed")
                    return False
                teacher_model_path = str(self.teacher_dir / "best_model.pt")
            
            # Step 2: Generate Koopman dataset
            if not self.step2_generate_koopman_dataset(teacher_model_path):
                logger.error("Koopman dataset generation failed")
                return False
            
            # Step 3: Train Koopman student
            if not self.step3_train_koopman(teacher_model_path):
                logger.error("Koopman training failed")
                return False
                
            koopman_model_path = str(self.koopman_dir / "best_model.pt")
            
            # Step 3: Evaluate
            if not self.step3_evaluate(koopman_model_path, teacher_model_path):
                logger.error("Evaluation failed")
                return False
                
            # Generate report
            report = self.generate_report()
            self.print_final_summary(report)
            
            total_time = time.time() - start_time
            logger.info(f"Complete pipeline test finished in {total_time:.1f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline test failed with error: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test complete Koopman SBI pipeline')
    parser.add_argument('--output_base', default='test_results',
                       help='Base directory for all outputs')
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode (smaller datasets, fewer epochs)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean output directory before starting')
    parser.add_argument('--use-pretrained-teacher', type=str, default=None,
                       help='Path to pre-trained teacher model (skips teacher training)')
    parser.add_argument('--regenerate-koopman-dataset', action='store_true',
                       help='Regenerate Koopman dataset even if it exists')
    
    args = parser.parse_args()
    
    # Clean output directory if requested
    if args.clean and os.path.exists(args.output_base):
        if args.use_pretrained_teacher:
            # Only clean non-teacher directories when using pre-trained teacher
            logger.info(f"Cleaning non-teacher directories in {args.output_base}")
            logger.info(f"PRESERVING teacher_two_moons directory")
            for subdir in ['koopman_two_moons', 'evaluation']:
                subdir_path = os.path.join(args.output_base, subdir)
                if os.path.exists(subdir_path):
                    logger.info(f"Deleting {subdir_path}")
                    shutil.rmtree(subdir_path)
                else:
                    logger.info(f"Directory {subdir_path} does not exist, skipping")
        else:
            logger.info(f"Cleaning entire output directory: {args.output_base}")
            shutil.rmtree(args.output_base)
        
    # Create tester and run pipeline
    tester = KoopmanPipelineTester(args.output_base, args.quick)
    tester.use_pretrained_teacher = args.use_pretrained_teacher
    tester.regenerate_koopman_dataset = args.regenerate_koopman_dataset
    
    success = tester.run_complete_pipeline()
    
    if success:
        logger.info("Pipeline test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline test failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
"""
Batch Processing Script
Process multiple datasets or run multiple experiments with different configurations
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import subprocess


class ExperimentRunner:
    """Run multiple experiments with different configurations"""
    
    def __init__(self, base_config: Dict):
        """
        Initialize experiment runner
        
        Args:
            base_config: Base configuration to use
        """
        self.base_config = base_config
        self.results = []
    
    def run_experiment(self, 
                      exp_name: str,
                      dataset_dir: str,
                      config_overrides: Dict = None) -> Dict:
        """
        Run a single experiment
        
        Args:
            exp_name: Experiment name
            dataset_dir: Dataset directory
            config_overrides: Configuration overrides
            
        Returns:
            Experiment results dictionary
        """
        print("\n" + "="*60)
        print(f"RUNNING EXPERIMENT: {exp_name}")
        print("="*60)
        
        # Create output directory
        output_dir = Path("experiments") / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            sys.executable, "inference_pipeline.py",
            "--dataset_dir", dataset_dir,
            "--output", str(output_dir / "predictions.json")
        ]
        
        # Add configuration overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if key == "feature_model":
                    cmd.extend(["--feature_model", value])
                elif key == "detector_model":
                    cmd.extend(["--detector_model", value])
                elif key == "similarity_threshold":
                    cmd.extend(["--similarity_threshold", str(value)])
                elif key == "detection_conf":
                    cmd.extend(["--detection_conf", str(value)])
                elif key == "frame_skip":
                    cmd.extend(["--frame_skip", str(value)])
                elif key == "no_tracking":
                    if value:
                        cmd.append("--no_tracking")
                elif key == "no_temporal":
                    if value:
                        cmd.append("--no_temporal")
        
        # Run experiment
        start_time = datetime.now()
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save logs
        with open(output_dir / "stdout.log", 'w') as f:
            f.write(result.stdout)
        with open(output_dir / "stderr.log", 'w') as f:
            f.write(result.stderr)
        
        # Save configuration
        config_info = {
            'experiment_name': exp_name,
            'base_config': self.base_config,
            'overrides': config_overrides,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration
        }
        
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print(f"Experiment completed in {duration:.2f} seconds")
        print(f"Results saved to: {output_dir}")
        
        return {
            'name': exp_name,
            'config': config_info,
            'success': result.returncode == 0,
            'duration': duration,
            'output_dir': str(output_dir)
        }
    
    def run_grid_search(self, 
                       dataset_dir: str,
                       param_grid: Dict[str, List]) -> List[Dict]:
        """
        Run grid search over parameter combinations
        
        Args:
            dataset_dir: Dataset directory
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of experiment results
        """
        # Generate all combinations
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        experiments = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            exp_name = "_".join([f"{k}={v}" for k, v in config.items()])
            exp_name = exp_name.replace(".", "p")  # Replace dots for filesystem
            experiments.append((exp_name, config))
        
        print(f"\nRunning grid search with {len(experiments)} experiments")
        
        results = []
        for exp_name, config in experiments:
            result = self.run_experiment(exp_name, dataset_dir, config)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def summarize_results(self, output_path: str = "experiments/summary.json"):
        """
        Summarize all experiment results
        
        Args:
            output_path: Path to save summary
        """
        summary = {
            'total_experiments': len(self.results),
            'successful': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'total_duration': sum(r['duration'] for r in self.results),
            'experiments': self.results
        }
        
        # Save summary
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total duration: {summary['total_duration']:.2f} seconds")
        print(f"\nSummary saved to: {output_path}")
        print("="*60)


def run_presets(dataset_dir: str):
    """Run all preset configurations"""
    runner = ExperimentRunner({})
    
    presets = {
        'fast': {
            'feature_model': 'clip',
            'detector_model': 'yolov8n.pt',
            'frame_skip': 5,
            'detection_conf': 0.2,
            'no_temporal': True
        },
        'balanced': {
            'feature_model': 'clip',
            'detector_model': 'yolov8l.pt',
            'frame_skip': 2,
            'detection_conf': 0.2,
            'similarity_threshold': 0.5
        },
        'accurate': {
            'feature_model': 'both',
            'detector_model': 'yolov8x.pt',
            'frame_skip': 1,
            'detection_conf': 0.15,
            'similarity_threshold': 0.45
        }
    }
    
    for preset_name, config in presets.items():
        runner.run_experiment(f"preset_{preset_name}", dataset_dir, config)
    
    runner.summarize_results()


def run_similarity_search(dataset_dir: str):
    """Run experiments with different similarity thresholds"""
    runner = ExperimentRunner({})
    
    param_grid = {
        'similarity_threshold': [0.3, 0.4, 0.5, 0.6, 0.7]
    }
    
    runner.run_grid_search(dataset_dir, param_grid)
    runner.summarize_results("experiments/similarity_search_summary.json")


def run_model_comparison(dataset_dir: str):
    """Compare different model combinations"""
    runner = ExperimentRunner({})
    
    param_grid = {
        'feature_model': ['clip', 'dinov2', 'both'],
        'detector_model': ['yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    }
    
    runner.run_grid_search(dataset_dir, param_grid)
    runner.summarize_results("experiments/model_comparison_summary.json")


def main():
    parser = argparse.ArgumentParser(
        description="Batch Processing and Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all preset configurations
  python batch_process.py --mode presets --dataset_dir ./train
  
  # Run similarity threshold search
  python batch_process.py --mode similarity --dataset_dir ./train
  
  # Run model comparison
  python batch_process.py --mode models --dataset_dir ./train
  
  # Custom grid search
  python batch_process.py --mode custom --dataset_dir ./train \\
      --param similarity_threshold 0.3,0.4,0.5 \\
      --param frame_skip 1,2,3
        """
    )
    
    parser.add_argument("--mode", type=str, required=True,
                       choices=['presets', 'similarity', 'models', 'custom'],
                       help="Batch processing mode")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Dataset directory")
    parser.add_argument("--param", type=str, action='append', nargs='+',
                       help="Parameter grid (for custom mode)")
    
    args = parser.parse_args()
    
    if args.mode == 'presets':
        run_presets(args.dataset_dir)
    
    elif args.mode == 'similarity':
        run_similarity_search(args.dataset_dir)
    
    elif args.mode == 'models':
        run_model_comparison(args.dataset_dir)
    
    elif args.mode == 'custom':
        if not args.param:
            print("Error: --param required for custom mode")
            return
        
        # Parse parameter grid
        param_grid = {}
        for param_spec in args.param:
            if len(param_spec) < 2:
                continue
            param_name = param_spec[0]
            param_values = param_spec[1].split(',')
            
            # Try to convert to appropriate type
            converted_values = []
            for v in param_values:
                try:
                    # Try int
                    converted_values.append(int(v))
                except ValueError:
                    try:
                        # Try float
                        converted_values.append(float(v))
                    except ValueError:
                        # Keep as string
                        converted_values.append(v)
            
            param_grid[param_name] = converted_values
        
        print(f"Parameter grid: {param_grid}")
        
        runner = ExperimentRunner({})
        runner.run_grid_search(args.dataset_dir, param_grid)
        runner.summarize_results("experiments/custom_search_summary.json")


if __name__ == "__main__":
    main()

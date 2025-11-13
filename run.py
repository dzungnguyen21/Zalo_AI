"""
Quick Start Script for Drone Object Detection Challenge
Provides easy-to-use commands for common tasks
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_installation():
    """Check if required packages are installed"""
    print("Checking installation...")
    
    # Check NumPy version first
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"✓ NumPy version: {numpy_version}")
        
        # Warn about NumPy 2.x compatibility
        if numpy_version.startswith('2.'):
            print("⚠ WARNING: NumPy 2.x detected!")
            print("  Some packages may have compatibility issues.")
            print("  Recommended: downgrade to NumPy 1.x")
            print("  Fix: pip install 'numpy<2.0.0' --force-reinstall")
    except ImportError:
        print("✗ NumPy not installed")
    
    try:
        import torch
        import cv2
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ OpenCV installed")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        
        # Try importing CLIP with warning suppression
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import clip
        print("✓ CLIP installed")
        
        from ultralytics import YOLO
        print("✓ Ultralytics (YOLOv8) installed")
        
        print("\n✅ All required packages are installed")
        print("   (Warnings above can be ignored if all packages loaded)")
        return True
        
    except ImportError as e:
        print(f"\n✗ Missing package: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        print("\nIf you see NumPy errors, run:")
        print("  pip install 'numpy<2.0.0' --force-reinstall")
        return False


def run_inference(args):
    """Run inference on dataset"""
    if not check_installation():
        return
    
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    cmd = [
        sys.executable, "inference_pipeline.py",
        "--dataset_dir", args.dataset_dir,
        "--output", args.output
    ]
    
    if args.model:
        cmd.extend(["--feature_model", args.model])
    if args.detector:
        cmd.extend(["--detector_model", args.detector])
    if args.threshold:
        cmd.extend(["--similarity_threshold", str(args.threshold)])
    if args.conf:
        cmd.extend(["--detection_conf", str(args.conf)])
    if args.skip:
        cmd.extend(["--frame_skip", str(args.skip)])
    if args.cpu:
        cmd.extend(["--device", "cpu"])
    if args.no_tracking:
        cmd.append("--no_tracking")
    if args.no_temporal:
        cmd.append("--no_temporal")
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_evaluation(args):
    """Run evaluation on predictions"""
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    cmd = [
        sys.executable, "evaluate.py",
        "--ground_truth", args.ground_truth,
        "--predictions", args.predictions
    ]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_analysis(args):
    """Analyze predictions"""
    print("\n" + "="*60)
    print("ANALYZING PREDICTIONS")
    print("="*60)
    
    cmd = [
        sys.executable, "evaluate.py",
        "--predictions", args.predictions,
        "--analyze_only"
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_quick_test(args):
    """Run quick test on a single video"""
    if not check_installation():
        return
    
    print("\n" + "="*60)
    print("QUICK TEST")
    print("="*60)
    
    # Use faster settings for quick test
    cmd = [
        sys.executable, "inference_pipeline.py",
        "--dataset_dir", args.dataset_dir,
        "--output", "quick_test_predictions.json",
        "--feature_model", "clip",
        "--detector_model", "yolov8n.pt",
        "--frame_skip", "5",
        "--detection_conf", "0.2"
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    print("Note: Using fast settings (CLIP + YOLOv8n + frame_skip=5)")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Quick Start Script for Drone Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check installation
  python run.py check
  
  # Run inference
  python run.py infer --dataset_dir ./train --output predictions.json
  
  # Run with custom settings
  python run.py infer --dataset_dir ./train --model both --detector yolov8x.pt
  
  # Quick test with fast settings
  python run.py test --dataset_dir ./train
  
  # Evaluate predictions
  python run.py eval --ground_truth ./train/annotations/annotations.json --predictions predictions.json
  
  # Analyze predictions
  python run.py analyze --predictions predictions.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check installation')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--dataset_dir', type=str, required=True,
                             help='Path to dataset directory')
    infer_parser.add_argument('--output', type=str, default='predictions.json',
                             help='Output predictions file')
    infer_parser.add_argument('--model', type=str, choices=['clip', 'dinov2', 'both'],
                             help='Feature extraction model')
    infer_parser.add_argument('--detector', type=str,
                             help='YOLO model (e.g., yolov8x.pt)')
    infer_parser.add_argument('--threshold', type=float,
                             help='Similarity threshold')
    infer_parser.add_argument('--conf', type=float,
                             help='Detection confidence threshold')
    infer_parser.add_argument('--skip', type=int,
                             help='Frame skip (process every N-th frame)')
    infer_parser.add_argument('--cpu', action='store_true',
                             help='Use CPU instead of GPU')
    infer_parser.add_argument('--no_tracking', action='store_true',
                             help='Disable tracking')
    infer_parser.add_argument('--no_temporal', action='store_true',
                             help='Disable temporal matching')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate predictions')
    eval_parser.add_argument('--ground_truth', type=str, required=True,
                            help='Path to ground truth annotations')
    eval_parser.add_argument('--predictions', type=str, required=True,
                            help='Path to predictions file')
    eval_parser.add_argument('--output', type=str,
                            help='Output evaluation results file')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze predictions')
    analyze_parser.add_argument('--predictions', type=str, required=True,
                               help='Path to predictions file')
    
    # Quick test command
    test_parser = subparsers.add_parser('test', help='Quick test with fast settings')
    test_parser.add_argument('--dataset_dir', type=str, required=True,
                            help='Path to dataset directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'check':
        check_installation()
    elif args.command == 'infer':
        run_inference(args)
    elif args.command == 'eval':
        run_evaluation(args)
    elif args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'test':
        run_quick_test(args)


if __name__ == "__main__":
    main()

"""
Quick Fix for NumPy Compatibility Issues
Run this script to fix NumPy 2.x compatibility problems
"""

import subprocess
import sys


def fix_numpy():
    """Fix NumPy compatibility by downgrading to 1.x"""
    print("="*60)
    print("FIXING NUMPY COMPATIBILITY")
    print("="*60)
    
    print("\n📦 Current package versions:")
    subprocess.run([sys.executable, "-m", "pip", "show", "numpy"])
    
    print("\n🔧 Downgrading NumPy to 1.x for compatibility...")
    print("   (This is required for CLIP and torchvision)")
    
    # Uninstall NumPy 2.x
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall", "numpy", "-y"
    ])
    
    # Install NumPy 1.x
    subprocess.run([
        sys.executable, "-m", "pip", "install", "numpy<2.0.0"
    ])
    
    print("\n✅ NumPy fix complete!")
    print("\nVerifying installation...")
    
    # Verify
    result = subprocess.run([
        sys.executable, "-c", 
        "import numpy; print(f'NumPy version: {numpy.__version__}')"
    ])
    
    if result.returncode == 0:
        print("\n✅ Success! You can now run the pipeline.")
        print("\nNext steps:")
        print("  python run.py check")
        print("  python run.py test --dataset_dir ./train")
    else:
        print("\n⚠ There may still be issues. Try:")
        print("  pip install -r requirements.txt --force-reinstall")


def main():
    print("\n⚠️  NumPy 2.x Compatibility Fix")
    print("="*60)
    print("\nThis script will:")
    print("  1. Uninstall NumPy 2.x")
    print("  2. Install NumPy 1.x (compatible with CLIP/torchvision)")
    print("  3. Verify the installation")
    
    response = input("\nProceed? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        fix_numpy()
    else:
        print("\nAborted. To fix manually, run:")
        print("  pip uninstall numpy -y")
        print("  pip install 'numpy<2.0.0'")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
00_verify_environment.py - Pre-flight checks for ELPV SSL Benchmark

Verifies:
1. Python version and required packages
2. CUDA availability and GPU info
3. ELPV dataset accessibility
4. Directory structure
5. semilearn library imports

Run this BEFORE starting any experiments!
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_python_version():
    """Check Python version >= 3.8"""
    print("\n[1/6] Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)")
        return False


def check_required_packages():
    """Check all required packages are installed"""
    print("\n[2/6] Checking required packages...")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'scipy': 'SciPy',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'psutil': 'psutil',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            all_ok = False
    
    # Check elpv_dataset separately
    try:
        from elpv_dataset.utils import load_dataset
        print(f"  ✅ elpv-dataset")
    except ImportError:
        print(f"  ❌ elpv-dataset - NOT INSTALLED (pip install elpv-dataset)")
        all_ok = False
    
    # Check scikit-posthocs for Friedman test
    try:
        import scikit_posthocs
        print(f"  ✅ scikit-posthocs")
    except ImportError:
        print(f"  ⚠️  scikit-posthocs - NOT INSTALLED (needed for Friedman test)")
    
    return all_ok


def check_cuda():
    """Check CUDA availability and GPU info"""
    print("\n[3/6] Checking CUDA and GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.version.cuda}")
            print(f"  ✅ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024**3)
                print(f"      GPU {i}: {props.name} ({mem_gb:.1f} GB)")
            
            return True
        else:
            print(f"  ⚠️  CUDA not available - will use CPU (SLOW!)")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        return False


def check_elpv_dataset():
    """Check ELPV dataset can be loaded"""
    print("\n[4/6] Checking ELPV dataset...")
    
    try:
        from elpv_dataset.utils import load_dataset
        images, proba, types = load_dataset()
        
        n_total = len(images)
        n_functional = sum(proba < 0.5)
        n_defective = sum(proba >= 0.5)
        
        print(f"  ✅ ELPV dataset loaded successfully")
        print(f"      Total images: {n_total}")
        print(f"      Functional (class 0): {n_functional}")
        print(f"      Defective (class 1): {n_defective}")
        print(f"      Image shape: {images[0].shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error loading ELPV dataset: {e}")
        return False


def check_directory_structure():
    """Check required directories exist"""
    print("\n[5/6] Checking directory structure...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    required_dirs = [
        'configs/elpv_benchmark/base',
        'configs/elpv_benchmark/experiments',
        'data/elpv/splits/train/seed0',
        'data/elpv/splits/train/seed1',
        'data/elpv/splits/train/seed2',
        'results/elpv_benchmark/experiments',
        'results/elpv_benchmark/aggregated',
        'results/elpv_benchmark/figures',
        'scripts/elpv_benchmark/slurm',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_ok = False
    
    return all_ok


def check_semilearn_imports():
    """Check semilearn library imports work"""
    print("\n[6/6] Checking semilearn library...")
    
    try:
        from semilearn.algorithms import get_algorithm, name2alg
        from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg
        from semilearn.core.utils import get_dataset, get_data_loader, get_net_builder
        from semilearn.datasets.cv_datasets import get_elpv
        
        # Check available algorithms
        ssl_algos = list(name2alg.keys())
        imb_algos = list(name2imbalg.keys())
        
        print(f"  ✅ semilearn library imports successful")
        print(f"      SSL algorithms: {len(ssl_algos)}")
        print(f"      Imbalanced algorithms: {len(imb_algos)}")
        
        # Check our required algorithms exist
        required_algos = ['fixmatch', 'flexmatch', 'freematch', 'softmatch', 'uda', 'meanteacher']
        required_imb = ['abc', 'darp', 'daso']
        
        missing_ssl = [a for a in required_algos if a not in ssl_algos]
        missing_imb = [a for a in required_imb if a not in imb_algos]
        
        if missing_ssl:
            print(f"  ⚠️  Missing SSL algorithms: {missing_ssl}")
        if missing_imb:
            print(f"  ⚠️  Missing imbalanced algorithms: {missing_imb}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error importing semilearn: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("ELPV SSL BENCHMARK - ENVIRONMENT VERIFICATION")
    print("="*70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("CUDA/GPU", check_cuda),
        ("ELPV Dataset", check_elpv_dataset),
        ("Directory Structure", check_directory_structure),
        ("Semilearn Library", check_semilearn_imports),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to run experiments!")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before proceeding")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

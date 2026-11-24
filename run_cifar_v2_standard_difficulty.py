#!/usr/bin/env python3
"""
Run CIFAR-10 v2 experiments with standard difficulty setup
(matches MNIST and STL-10: Clean, Easy, Middle, Hard, Very Hard)
"""

import subprocess
import os
import sys

PYTHON = "/opt/anaconda3/envs/deep_rewiring/bin/python"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Standard difficulty levels (same as MNIST and STL-10)
DIFFICULTIES = [
    ("clean", 0.0, 0.0),
    ("easy", 5.0, 0.02),
    ("middle", 15.0, 0.05),
    ("hard", 30.0, 0.1),
    ("very_hard", 45.0, 0.2),
]

def run_experiment(script, tag, max_angle, noise_sigma, n_epochs=10):
    """Run a single experiment."""
    cmd = [
        PYTHON, "-u", script,
        f"--max_angle={max_angle}",
        f"--noise_sigma={noise_sigma}",
        f"--eval_angle={max_angle}",
        f"--eval_noise_sigma={noise_sigma}",
        f"--tag={tag}",
        f"--n_epochs={n_epochs}",
    ]
    
    print(f"\n{'='*70}")
    print(f"Running: {tag}")
    print(f"  max_angle={max_angle}, noise_sigma={noise_sigma}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0

def main():
    print("="*70)
    print("CIFAR-10 v2 - Standard Difficulty Experiments")
    print("="*70)
    
    completed = 0
    failed = 0
    
    # DEEP R experiments
    print("\n" + "="*70)
    print("Model: DEEP R (v2 architecture: 3072-2048-1024-512-10 + BatchNorm)")
    print("="*70)
    
    for diff_name, max_angle, noise_sigma in DIFFICULTIES:
        tag = f"cifar_v2_drnn_{diff_name}"
        script = "script_cifar_deep_rewiring_exp7_v2.py"
        
        # Check if already completed
        results_file = f"results/{tag}_results.pickle"
        if os.path.exists(results_file):
            print(f"\n[DRNN] {diff_name}: SKIPPED (already completed)")
            completed += 1
            continue
        
        success = run_experiment(script, tag, max_angle, noise_sigma)
        if success:
            completed += 1
        else:
            failed += 1
            print(f"  ✗ FAILED: {tag}")
    
    # FC experiments
    print("\n" + "="*70)
    print("Model: FC (v2 architecture: 3072-2048-1024-512-10 + BatchNorm)")
    print("="*70)
    
    for diff_name, max_angle, noise_sigma in DIFFICULTIES:
        tag = f"cifar_v2_fc_{diff_name}"
        script = "script_cifar_fully_connected_exp7_v2.py"
        
        # Check if already completed
        results_file = f"results/{tag}_results.pickle"
        if os.path.exists(results_file):
            print(f"\n[FC] {diff_name}: SKIPPED (already completed)")
            completed += 1
            continue
        
        success = run_experiment(script, tag, max_angle, noise_sigma)
        if success:
            completed += 1
        else:
            failed += 1
            print(f"  ✗ FAILED: {tag}")
    
    # Summary
    print("\n" + "="*70)
    print(f"Summary: {completed} completed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Run STL-10 experiments with different difficulty levels
Similar to CIFAR experiments: clean, easy, middle, hard, very hard
"""

import subprocess
import sys
import os

# Define difficulty levels (same as CIFAR)
difficulty_levels = [
    {
        'name': 'clean',
        'max_angle': 0.0,
        'noise_sigma': 0.0,
        'eval_angle': 0.0,
        'eval_noise_sigma': 0.0,
    },
    {
        'name': 'easy',
        'max_angle': 5.0,
        'noise_sigma': 0.02,
        'eval_angle': 5.0,
        'eval_noise_sigma': 0.02,
    },
    {
        'name': 'middle',
        'max_angle': 15.0,
        'noise_sigma': 0.05,
        'eval_angle': 15.0,
        'eval_noise_sigma': 0.05,
    },
    {
        'name': 'hard',
        'max_angle': 30.0,
        'noise_sigma': 0.10,
        'eval_angle': 30.0,
        'eval_noise_sigma': 0.10,
    },
    {
        'name': 'very_hard',
        'max_angle': 45.0,
        'noise_sigma': 0.20,
        'eval_angle': 45.0,
        'eval_noise_sigma': 0.20,
    },
]

# Models to run
models = [
    {
        'name': 'drnn',
        'script': 'script_stl10_deep_rewiring_exp7_v2.py',
        'tag_prefix': 'stl10_drnn',
    },
    {
        'name': 'fc',
        'script': 'script_stl10_fully_connected_exp7_v2.py',
        'tag_prefix': 'stl10_fc',
    },
]

python_exec = '/opt/anaconda3/envs/deep_rewiring/bin/python'

print("=" * 70)
print("STL-10 Difficulty Experiments")
print("=" * 70)
print()

for model in models:
    print(f"\n{'=' * 70}")
    print(f"Running model: {model['name'].upper()}")
    print(f"{'=' * 70}")
    
    for diff in difficulty_levels:
        tag = f"{model['tag_prefix']}_{diff['name']}"
        print(f"\n[{model['name'].upper()}] Difficulty: {diff['name']}")
        print(f"  max_angle={diff['max_angle']}, noise_sigma={diff['noise_sigma']}")
        print(f"  tag={tag}")
        
        # Build command
        cmd = [
            python_exec, '-u', model['script'],
            f"--max_angle={diff['max_angle']}",
            f"--noise_sigma={diff['noise_sigma']}",
            f"--eval_angle={diff['eval_angle']}",
            f"--eval_noise_sigma={diff['eval_noise_sigma']}",
            f"--tag={tag}",
            "--n_epochs=10",
        ]
        
        print(f"  Command: {' '.join(cmd)}")
        print()
        
        # Run experiment
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Extract final accuracy from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Final acc:' in line:
                        print(f"  ✓ {line.strip()}")
                        break
                else:
                    print("  ✓ Completed (check results for details)")
            else:
                print(f"  ✗ Failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:500]}")
        
        except subprocess.TimeoutExpired:
            print("  ✗ Timeout after 1 hour")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
        
        print()

print("=" * 70)
print("All experiments completed!")
print("=" * 70)

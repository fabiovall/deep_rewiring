# STL-10 v3: Larger Architecture for Improved Accuracy

## Overview

This document describes the STL-10 v3 experiments, which use a larger architecture to improve FC accuracy, similar to the CIFAR-10 v2 improvement.

## Architecture Comparison

### STL-10 v2 (Current)
```
27648 → 2048 → 1024 → 512 → 10 (+ BatchNorm)
```
- First layer: 27648 × 2048 = 56.6M weights
- Total parameters: ~60M

### STL-10 v3 (New)
```
27648 → 4096 → 2048 → 1024 → 512 → 10 (+ BatchNorm)
```
- First layer: 27648 × 4096 = 113.2M weights
- Total parameters: ~120M

## Expected Improvements

Based on CIFAR-10 v2 results:
- **FC accuracy**: Expected to improve from ~29.8% (v2) to ~40-45% (v3)
- **dRNN accuracy**: Expected to improve from ~32.9% (v2) to ~45-50% (v3)
- **Gap between FC and dRNN**: May narrow slightly

## Files

### New Scripts
- `script_stl10_fully_connected_exp7_v3.py` - FC baseline with 4 hidden layers
- `script_stl10_deep_rewiring_exp7_v3.py` - DEEP R with 4 hidden layers
- `run_stl10_v3_experiments.py` - Runner script for all 10 experiments

### Connectivity Fractions (dRNN)
- Layer 1 (27648→4096): 1%
- Layer 2 (4096→2048): 3%
- Layer 3 (2048→1024): 3%
- Layer 4 (1024→512): 3%
- Output (512→10): 30%

## Running the Experiments

### Run All 10 Experiments
```bash
/opt/anaconda3/envs/deep_rewiring/bin/python run_stl10_v3_experiments.py
```

### Run Individual Experiments
```bash
# FC Clean
/opt/anaconda3/envs/deep_rewiring/bin/python script_stl10_fully_connected_exp7_v3.py \
    --max_angle=0.0 --noise_sigma=0.0 --eval_angle=0.0 --eval_noise_sigma=0.0 \
    --tag=stl10_v3_fc_clean --n_epochs=10

# dRNN Clean
/opt/anaconda3/envs/deep_rewiring/bin/python script_stl10_deep_rewiring_exp7_v3.py \
    --max_angle=0.0 --noise_sigma=0.0 --eval_angle=0.0 --eval_noise_sigma=0.0 \
    --tag=stl10_v3_drnn_clean --n_epochs=10
```

## Difficulty Levels

Same as v2:
- **Clean**: No augmentation
- **Easy**: ±5° rotation, σ=0.02 noise
- **Middle**: ±15° rotation, σ=0.05 noise
- **Hard**: ±30° rotation, σ=0.10 noise
- **Very Hard**: ±45° rotation, σ=0.20 noise

## Expected Runtime

- **FC**: ~2-3 hours per experiment (larger architecture)
- **dRNN**: ~3-4 hours per experiment (rewiring overhead)
- **Total**: ~30-40 hours for all 10 experiments

## Results

Results will be saved to:
- `results/stl10_v3_fc_*_results.pickle`
- `results/stl10_v3_drnn_*_results.pickle`

## Comparison with v2

After running, compare with v2 results:
```python
import pickle

# v2 results
with open('results/stl10_fc_clean_results.pickle', 'rb') as f:
    v2_fc = pickle.load(f)['acc_list'][-1] * 100

with open('results/stl10_drnn_clean_results.pickle', 'rb') as f:
    v2_drnn = pickle.load(f)['acc_list'][-1] * 100

# v3 results
with open('results/stl10_v3_fc_clean_results.pickle', 'rb') as f:
    v3_fc = pickle.load(f)['acc_list'][-1] * 100

with open('results/stl10_v3_drnn_clean_results.pickle', 'rb') as f:
    v3_drnn = pickle.load(f)['acc_list'][-1] * 100

print(f"v2: FC={v2_fc:.1f}%, dRNN={v2_drnn:.1f}%")
print(f"v3: FC={v3_fc:.1f}%, dRNN={v3_drnn:.1f}%")
print(f"Improvement: FC +{v3_fc-v2_fc:.1f} pp, dRNN +{v3_drnn-v2_drnn:.1f} pp")
```

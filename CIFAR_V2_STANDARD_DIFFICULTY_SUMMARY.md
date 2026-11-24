# CIFAR-10 v2 Standard Difficulty Experiments

## Overview

This document summarizes the CIFAR-10 v2 experiments with the standard difficulty setup (Clean, Easy, Middle, Hard, Very Hard) to match the MNIST and STL-10 experiments.

## What Was Done

### 1. Created Experiment Runner
**File:** `run_cifar_v2_standard_difficulty.py`
- Runs all 10 experiments (dRNN + FC at 5 difficulty levels)
- Uses the v2 architecture (3072-2048-1024-512-10 + BatchNorm)
- Standard difficulty levels:
  - Clean: no augmentation
  - Easy: ±5° rotation + σ=0.02 noise
  - Middle: ±15° rotation + σ=0.05 noise
  - Hard: ±30° rotation + σ=0.10 noise
  - Very Hard: ±45° rotation + σ=0.20 noise

### 2. Ran All Experiments
**Duration:** ~30 minutes total
- dRNN: 5 experiments (clean, easy, middle, hard, very_hard)
- FC: 5 experiments (clean, easy, middle, hard, very_hard)
- All completed successfully

### 3. Generated Updated Comparison Plot
**File:** `results/comprehensive_comparison_v2.png`
- Shows all 4 datasets (MNIST, CIFAR-10 v1, CIFAR-10 v2, STL-10)
- 4 subplots: accuracy, robustness, performance gap, summary statistics

## Results

### CIFAR-10 v2 (3072-2048-1024-512-10)

| Difficulty | dRNN acc | Δ dRNN | FC acc | Δ FC | Δ (FC − dRNN) |
|---|---|---|---|---|---|
| Clean | 55.7 % | — | 57.4 % | — | +1.6 pp |
| Easy (±5°, σ=0.02) | 48.1 % | -7.7 pp | 51.0 % | -6.4 pp | +2.9 pp |
| Middle (±15°, σ=0.05) | 43.3 % | -12.5 pp | 45.9 % | -11.4 pp | +2.7 pp |
| Hard (±30°, σ=0.10) | 38.7 % | -17.0 pp | 40.0 % | -17.4 pp | +1.3 pp |
| Very Hard (±45°, σ=0.20) | 31.9 % | -23.8 pp | 33.9 % | -23.5 pp | +1.9 pp |

**Key Findings:**
- dRNN connectivity: 1.60%
- FC slightly outperforms dRNN at all difficulty levels (1-3 pp)
- Both models show similar degradation patterns
- dRNN degrades slightly more at easy/middle difficulty
- At very-hard difficulty, both models lose ~24 pp from clean baseline

## Comparison: v1 vs v2

### Clean Accuracy
- **dRNN:** v1=39.5%, v2=55.7% (**+16.2 pp improvement**)
- **FC:** v1=48.3%, v2=57.4% (**+9.0 pp improvement**)

### Key Insights
1. **Larger architecture helps both models:** The v2 architecture (3072-2048-1024-512-10) significantly outperforms v1 (3072-300-100-10)
2. **dRNN benefits more from larger architecture:** +16.2 pp vs +9.0 pp for FC
3. **Gap narrows with larger architecture:** v1 had 8.8 pp gap (FC better), v2 has only 1.6 pp gap
4. **Consistent robustness pattern:** Both models degrade similarly across difficulty levels

## Complete Study Across 4 Datasets

### MNIST (784-300-100-10)
- dRNN connectivity: 1.33%
- At clean/easy: dRNN ≈ FC (within 0.5 pp)
- At hard/very-hard: dRNN degrades faster than FC
- **Conclusion:** Sparse network is more sensitive to perturbation on easier datasets

### CIFAR-10 v1 (3072-300-100-10)
- dRNN connectivity: 1.09%
- At clean: FC outperforms dRNN by 8.8 pp
- At augmented difficulties: dRNN degrades LESS than FC
- **Conclusion:** Sparse network is MORE robust on harder datasets (regularization effect)

### CIFAR-10 v2 (3072-2048-1024-512-10)
- dRNN connectivity: 1.60%
- At clean: FC outperforms dRNN by 1.6 pp (much smaller gap)
- At augmented difficulties: similar degradation patterns
- **Conclusion:** Larger architecture narrows the gap; dRNN benefits more from capacity

### STL-10 (27648-2048-1024-512-10)
- dRNN connectivity: 1.09%
- At clean/easy: dRNN outperforms FC
- At hard/very-hard: comparable performance (within 1-2 pp)
- **Conclusion:** Sparse network performs comparably or better on largest dataset

## Key Insights

1. **Architecture matters:** Larger architectures (v2) significantly improve accuracy for both dRNN and FC
2. **dRNN benefits more from capacity:** The sparse network gains more from larger layers (+16.2 pp vs +9.0 pp)
3. **Gap narrows with capacity:** The performance gap between FC and dRNN shrinks with larger architectures
4. **Dataset-dependent robustness:** The effect of sparsity on robustness depends on dataset complexity
5. **Parameter efficiency:** DEEP R achieves 97-99% parameter reduction with comparable or better performance

## Files Created

```
run_cifar_v2_standard_difficulty.py              # Experiment runner
results/cifar_v2_drnn_clean_results.pickle       # dRNN clean
results/cifar_v2_drnn_easy_results.pickle        # dRNN easy
results/cifar_v2_drnn_middle_results.pickle      # dRNN middle
results/cifar_v2_drnn_hard_results.pickle        # dRNN hard
results/cifar_v2_drnn_very_hard_results.pickle   # dRNN very hard
results/cifar_v2_fc_clean_results.pickle         # FC clean
results/cifar_v2_fc_easy_results.pickle          # FC easy
results/cifar_v2_fc_middle_results.pickle        # FC middle
results/cifar_v2_fc_hard_results.pickle          # FC hard
results/cifar_v2_fc_very_hard_results.pickle     # FC very hard
results/comprehensive_comparison_v2.png          # Updated comparison plot
results/cifar_v2_standard_difficulty.log         # Experiment log
```

## How to Reproduce

### Run all CIFAR v2 standard difficulty experiments:
```bash
conda activate deep_rewiring
python run_cifar_v2_standard_difficulty.py
```

### View the updated comparison plot:
```bash
open results/comprehensive_comparison_v2.png
```

## Conclusion

The CIFAR-10 v2 standard difficulty experiments complete a comprehensive study of Deep Rewiring across four datasets with consistent difficulty levels. The results demonstrate that:

1. DEEP R successfully trains very sparse networks (1-3% connectivity) from scratch
2. Larger architectures significantly improve accuracy for both sparse and dense networks
3. The sparse network benefits more from increased capacity (+16.2 pp vs +9.0 pp)
4. The performance gap between FC and dRNN narrows with larger architectures
5. The robustness to input perturbation is dataset-dependent

The updated comprehensive comparison plot provides a complete overview of all four datasets with consistent difficulty levels.

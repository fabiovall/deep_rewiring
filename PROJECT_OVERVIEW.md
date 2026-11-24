# Deep Rewiring Project Overview

## Project Summary

This project implements and evaluates **DEEP R (Deep Rewiring)**, a biologically-inspired algorithm for training very sparse neural networks from scratch. The algorithm maintains constant connectivity (1-3%) throughout training, mimicking synaptic plasticity in the brain.

## Key Results

### Across Datasets

| Dataset | Architecture | dRNN Connectivity | FC Clean Acc | dRNN Clean Acc | Gap (FC - dRNN) |
|---|---|---|---|---|---|
| MNIST | 784-300-100-10 | 1.33% | 97.5% | 97.0% | +0.5 pp |
| CIFAR-10 v1 | 3072-300-100-10 | 1.09% | 48.3% | 39.5% | +8.8 pp |
| CIFAR-10 v2 | 3072-2048-1024-512-10 | 1.60% | 57.4% | 55.7% | +1.6 pp |
| STL-10 v2 | 27648-2048-1024-512-10 | 1.09% | 29.8% | 32.9% | -3.1 pp |
| STL-10 v3 | 27648-4096-2048-1024-512-10 | ~1.0% | ~40-45% (expected) | ~45-50% (expected) | ~-5 to 0 pp (expected) |

### Key Insights

1. **Dataset-dependent robustness:**
   - On easier datasets (MNIST), sparsity may hurt robustness to perturbation
   - On harder datasets (CIFAR-10, STL-10), sparsity acts as a regularizer and improves robustness

2. **Architecture matters:**
   - Larger architectures significantly improve accuracy for both sparse and dense networks
   - The sparse network benefits more from increased capacity (CIFAR-10: +16.2 pp vs +9.0 pp)
   - The performance gap between FC and dRNN narrows with larger architectures

3. **Parameter efficiency:**
   - DEEP R achieves 97-99% parameter reduction
   - Comparable or better performance than dense networks on complex tasks
   - No post-training pruning required

## Project Structure

### Core Scripts

#### MNIST
- `script_deep_rewiring_exp7.py` - DEEP R on MNIST
- `script_fully_connected_exp7.py` - FC baseline on MNIST

#### CIFAR-10
- `script_cifar_deep_rewiring_exp7.py` - DEEP R on CIFAR-10 v1
- `script_cifar_fully_connected_exp7.py` - FC baseline on CIFAR-10 v1
- `script_cifar_deep_rewiring_exp7_v2.py` - DEEP R on CIFAR-10 v2 (larger architecture)
- `script_cifar_fully_connected_exp7_v2.py` - FC baseline on CIFAR-10 v2

#### STL-10
- `script_stl10_deep_rewiring_exp7_v2.py` - DEEP R on STL-10 v2
- `script_stl10_fully_connected_exp7_v2.py` - FC baseline on STL-10 v2
- `script_stl10_deep_rewiring_exp7_v3.py` - DEEP R on STL-10 v3 (larger architecture)
- `script_stl10_fully_connected_exp7_v3.py` - FC baseline on STL-10 v3

### Data Loaders
- `mnist_data.py` - MNIST data loader
- `cifar_data.py` - CIFAR-10 data loader
- `stl10_data.py` - STL-10 data loader

### Experiment Runners
- `run_cifar_v2_standard_difficulty.py` - CIFAR-10 v2 experiments
- `run_stl10_remaining_experiments.py` - STL-10 v2 experiments
- `run_stl10_v3_experiments.py` - STL-10 v3 experiments

### Documentation
- `README.md` - Main project documentation
- `STL10_EXPERIMENT_SUMMARY.md` - STL-10 v2 results and analysis
- `STL10_V3_README.md` - STL-10 v3 documentation
- `CIFAR_V2_STANDARD_DIFFICULTY_SUMMARY.md` - CIFAR-10 v2 results
- `PROJECT_OVERVIEW.md` - This file

### Reports
- `generate_final_report.py` - PDF report generator
- `results/deep_rewiring_comprehensive_report.pdf` - Comprehensive 9-page report

## Difficulty Levels

All experiments use 5 difficulty levels with rotation and Gaussian noise augmentation:

| Level | Rotation | Noise σ | Description |
|---|---|---|---|
| Clean | 0° | 0.0 | No augmentation |
| Easy | ±5° | 0.02 | Mild perturbation |
| Middle | ±15° | 0.05 | Moderate perturbation |
| Hard | ±30° | 0.10 | Strong perturbation |
| Very Hard | ±45° | 0.20 | Extreme perturbation |

## How to Run

### Activate Environment
```bash
conda activate deep_rewiring
```

### Run All Experiments

**MNIST:**
```bash
python run_mnist_experiments.py
```

**CIFAR-10 v2:**
```bash
python run_cifar_v2_standard_difficulty.py
```

**STL-10 v2:**
```bash
python run_stl10_remaining_experiments.py
```

**STL-10 v3:**
```bash
python run_stl10_v3_experiments.py
```

### Generate Report
```bash
python generate_final_report.py
```

### View Report
```bash
open results/deep_rewiring_comprehensive_report.pdf
```

## Results Storage

All results are saved as pickle files in `results/`:
- `results/mnist_drnn_*_results.pickle`
- `results/mnist_fc_*_results.pickle`
- `results/cifar_drnn_*_results.pickle`
- `results/cifar_fc_*_results.pickle`
- `results/stl10_drnn_*_results.pickle`
- `results/stl10_fc_*_results.pickle`
- `results/stl10_v3_drnn_*_results.pickle`
- `results/stl10_v3_fc_*_results.pickle`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{deep_rewiring_2026,
  title = {Deep Rewiring: Training Sparse Neural Networks from Scratch},
  author = {Vallone, Fabio},
  year = {2026},
  url = {https://github.com/fabiovallone/deep_rewiring}
}
```

## License

MIT License

## Contact

For questions or collaboration, contact: Fabio Vallone

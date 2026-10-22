#!/usr/bin/env bash
# =============================================================================
# run_fc_baselines.sh
# Runs all fully-connected (no rewiring) baselines for Experiment #7 difficulty
# settings. Each run produces results/{tag}_results.pickle, which can be
# plotted with plot_comparison.py alongside the DEEP R runs.
#
# Usage:
#   source /opt/anaconda3/etc/profile.d/conda.sh
#   conda activate deep_rewiring
#   bash run_fc_baselines.sh
#
# The scripts run sequentially. Total wall-time per run ≈ 10-15 min on M2 Pro.
# =============================================================================

set -euo pipefail

# --- Flags shared across all runs ---
EPOCHS=10
BATCH=10
PRINT_EVERY=100

# --- (max_angle, noise_sigma, tag) triples ---
# Keep in sync with the DEEP R experiment runs.
DIFFICULTIES=(
    # name          max_angle   noise_sigma   tag
    "easy"          5.0         0.02          fc_easy
    "default"       15.0        0.05          fc
    "hard"          30.0        0.10          fc_hard
    "very_hard"     45.0        0.20          fc_vhard
)

echo "============================================================"
echo " Fully-Connected Baselines — Experiment #7"
echo " $(date)"
echo "============================================================"

for entry in "${DIFFICULTIES[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    NAME=$1
    MAX_ANGLE=$2
    SIGMA=$3
    TAG=$4

    echo ""
    echo "------------------------------------------------------------"
    echo " Starting: $NAME  (max_angle=$MAX_ANGLE, sigma=$SIGMA, tag=$TAG)"
    echo " $(date)"
    echo "------------------------------------------------------------"

    python script_mnist_fully_connected_exp7.py \
        --n_epochs=$EPOCHS \
        --batch=$BATCH \
        --print_every=$PRINT_EVERY \
        --max_angle=$MAX_ANGLE \
        --noise_sigma=$SIGMA \
        --eval_angle=$MAX_ANGLE \
        --eval_noise_sigma=$SIGMA \
        --tag=$TAG

    echo " Done: $TAG  ($(date))"
done

echo ""
echo "============================================================"
echo " All FC baselines complete.  $(date)"
echo ""
echo " To plot all runs side-by-side:"
echo "   python plot_comparison.py \\"
echo "       --clean  results/deep_r_results.pickle \\"
echo "       --aug    results/exp7_results.pickle \\"
echo "       --aug    results/exp7_hard_results.pickle \\"
echo "       --aug    results/fc_results.pickle \\"
echo "       --aug    results/fc_hard_results.pickle \\"
echo "       --aug    results/fc_easy_results.pickle \\"
echo "       --aug    results/fc_vhard_results.pickle"
echo "============================================================"

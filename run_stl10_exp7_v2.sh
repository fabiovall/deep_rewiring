#!/usr/bin/env bash
# =============================================================================
# run_stl10_exp7_v2.sh
# Runs the complete STL-10 v2 experiment battery:
#   DEEP R and dense FC at 5 difficulty levels, then produces the results table.
#
# Architecture: 27648 -> 2048 -> 1024 -> 512 -> 10  (+ BatchNorm)
#
# Conditions:
#   clean          — no augmentation
#   rot15          — ±15° rotation
#   noise05        — σ=0.05 Gaussian noise
#   noise10        — σ=0.10 Gaussian noise
#   rot15_noise05  — ±15° + σ=0.05
#
# Usage:
#   bash run_stl10_exp7_v2.sh
#
# Estimated wall-time on M2 Pro: ~30-60 min/run × 10 runs ≈ 5-10 hours
# (STL-10 images are 9x wider than CIFAR-10, so each step is slower)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

CONDA_ENV="deep_rewiring"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "Conda environment '$CONDA_ENV' is active."
elif [[ -f /opt/anaconda3/etc/profile.d/conda.sh ]]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    echo "Conda environment '$CONDA_ENV' is active (via /opt/anaconda3)."
else
    echo "ERROR: conda not found."
    exit 1
fi

echo "Checking Python dependencies..."
python -c "import tensorflow" 2>/dev/null || { echo "tensorflow not found — installing..."; pip install tensorflow; }
python -c "import numpy"     2>/dev/null || { echo "numpy not found — installing...";     pip install numpy; }
python -c "import scipy"     2>/dev/null || { echo "scipy not found — installing...";     pip install scipy; }
echo "All dependencies OK."

mkdir -p results

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

EPOCHS=10
BATCH=128
PRINT_EVERY=200

run_deep_r() {
    local tag=$1 max_angle=$2 sigma=$3
    if [[ -f "results/${tag}_results.pickle" ]]; then
        echo "  [skip] results/${tag}_results.pickle already exists"
        return
    fi
    echo "  [run]  DEEP R    tag=${tag}  max_angle=${max_angle}  sigma=${sigma}"
    python script_stl10_deep_rewiring_exp7_v2.py \
        --n_epochs=$EPOCHS \
        --batch=$BATCH \
        --print_every=$PRINT_EVERY \
        --max_angle=$max_angle \
        --noise_sigma=$sigma \
        --eval_angle=$max_angle \
        --eval_noise_sigma=$sigma \
        --tag=$tag
}

run_fc() {
    local tag=$1 max_angle=$2 sigma=$3
    if [[ -f "results/${tag}_results.pickle" ]]; then
        echo "  [skip] results/${tag}_results.pickle already exists"
        return
    fi
    echo "  [run]  Dense FC  tag=${tag}  max_angle=${max_angle}  sigma=${sigma}"
    python script_stl10_fully_connected_exp7_v2.py \
        --n_epochs=$EPOCHS \
        --batch=$BATCH \
        --print_every=$PRINT_EVERY \
        --max_angle=$max_angle \
        --noise_sigma=$sigma \
        --eval_angle=$max_angle \
        --eval_noise_sigma=$sigma \
        --tag=$tag
}

# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo " STL-10 v2 Experiment Battery"
echo " $(date)"
echo "============================================================"

# (max_angle, sigma, deep_r_tag, fc_tag)
DIFFICULTIES=(
    "0.0  0.0   stl10_exp7_v2_clean        stl10_fc_v2_clean"
    "15.0 0.0   stl10_exp7_v2_rot15        stl10_fc_v2_rot15"
    "0.0  0.05  stl10_exp7_v2_noise05      stl10_fc_v2_noise05"
    "0.0  0.10  stl10_exp7_v2_noise10      stl10_fc_v2_noise10"
    "15.0 0.05  stl10_exp7_v2_rot15_noise05 stl10_fc_v2_rot15_noise05"
)

for entry in "${DIFFICULTIES[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    MAX_ANGLE=$1
    SIGMA=$2
    DR_TAG=$3
    FC_TAG=$4

    echo ""
    echo "------------------------------------------------------------"
    echo " Condition: max_angle=${MAX_ANGLE}°  sigma=${SIGMA}   $(date)"
    echo "------------------------------------------------------------"

    run_deep_r "$DR_TAG" "$MAX_ANGLE" "$SIGMA"
    run_fc     "$FC_TAG" "$MAX_ANGLE" "$SIGMA"
done

# ---------------------------------------------------------------------------
# Generate the results table
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo " All training runs complete.  $(date)"
echo " Generating results table..."
echo "============================================================"

python make_stl10_results_table.py

echo ""
echo "============================================================"
echo " Done.  $(date)"
echo "============================================================"

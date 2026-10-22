#!/usr/bin/env bash
# =============================================================================
# run_full_comparison.sh
# Self-contained script that runs the complete Experiment #7 battery:
# DEEP R and dense FC at each difficulty level, then produces the comparison
# plot. Safe to re-run — completed experiments are skipped.
#
# Quick start:
#   bash run_full_comparison.sh
#
# Estimated wall-time on M2 Pro (~12 min/run × 9 runs ≈ 110 min):
#   fc_clean        — clean MNIST, dense FC
#   exp7_easy       — ±5°/σ=0.02,  DEEP R
#   exp7_middle     — ±15°/σ=0.05, DEEP R
#   exp7_hard       — ±30°/σ=0.10, DEEP R
#   exp7_vhard      — ±45°/σ=0.20, DEEP R
#   fc_easy         — ±5°/σ=0.02,  dense FC
#   fc_middle       — ±15°/σ=0.05, dense FC
#   fc_hard         — ±30°/σ=0.10, dense FC
#   fc_vhard        — ±45°/σ=0.20, dense FC
#
# The clean DEEP R baseline (results/deep_r_results.pickle) must already exist
# from a previous run of script_mnist_deep_rewiring.py.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

CONDA_ENV="deep_rewiring"

# Activate the conda environment (idempotent — safe even if already active)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "Conda environment '$CONDA_ENV' is active."
elif [[ -f /opt/anaconda3/etc/profile.d/conda.sh ]]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    echo "Conda environment '$CONDA_ENV' is active (via /opt/anaconda3)."
else
    echo "ERROR: conda not found. Install Miniconda or Anaconda first."
    echo "       https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Verify required Python packages; install if missing
echo "Checking Python dependencies..."
python -c "import tensorflow" 2>/dev/null || {
    echo "tensorflow not found — installing..."
    pip install tensorflow
}
python -c "import numpy"     2>/dev/null || { echo "numpy not found — installing...";     pip install numpy; }
python -c "import matplotlib" 2>/dev/null || { echo "matplotlib not found — installing..."; pip install matplotlib; }
python -c "import scipy"     2>/dev/null || { echo "scipy not found — installing...";     pip install scipy; }
echo "All dependencies OK."

# Ensure results directory exists
mkdir -p results

# Verify the clean baseline pickle exists
if [[ ! -f "results/deep_r_results.pickle" ]]; then
    echo ""
    echo "WARNING: results/deep_r_results.pickle not found."
    echo "The clean-MNIST baseline is required for the comparison plot."
    echo "Run it first:"
    echo "    python script_mnist_deep_rewiring.py --n_epochs=10 --batch=10"
    echo ""
    read -rp "Continue anyway? [y/N] " ANSWER
    if [[ "$ANSWER" != "y" && "$ANSWER" != "Y" ]]; then
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

EPOCHS=10
BATCH=10
PRINT_EVERY=100

run_deep_r() {
    local tag=$1 max_angle=$2 sigma=$3
    if [[ -f "results/${tag}_results.pickle" ]]; then
        echo "  [skip] results/${tag}_results.pickle already exists"
        return
    fi
    echo "  [run]  DEEP R    tag=${tag}  max_angle=${max_angle}  sigma=${sigma}"
    python script_mnist_deep_rewiring_exp7.py \
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
    local tag=$1 max_angle=$2 sigma=$3 lr=${4:-0.5}
    if [[ -f "results/${tag}_results.pickle" ]]; then
        echo "  [skip] results/${tag}_results.pickle already exists"
        return
    fi
    echo "  [run]  Dense FC  tag=${tag}  max_angle=${max_angle}  sigma=${sigma}  lr=${lr}"
    python script_mnist_fully_connected_exp7.py \
        --n_epochs=$EPOCHS \
        --batch=$BATCH \
        --print_every=$PRINT_EVERY \
        --max_angle=$max_angle \
        --noise_sigma=$sigma \
        --eval_angle=$max_angle \
        --eval_noise_sigma=$sigma \
        --lr=$lr \
        --tag=$tag
}

# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo " Full Experiment #7 Comparison Battery"
echo " $(date)"
echo "============================================================"

# ── Clean MNIST: dense FC baseline ────────────────────────────────────────
# The clean DEEP R baseline (results/deep_r_results.pickle) already exists
# from the base script. The clean FC baseline is run here with tag fc_clean.
echo ""
echo "------------------------------------------------------------"
echo " Clean MNIST baselines   $(date)"
echo "------------------------------------------------------------"
run_fc "fc_clean" 0.0 0.0

# ── Difficulty levels: max_angle sigma deep_r_tag fc_tag fc_lr ──────────
# fc_lr: lr=0.01 needed for fc_vhard (±45°/σ=0.20) — at lr=0.5 the dense
#        network fails to descend from the uniform-loss basin.
DIFFICULTIES=(
    "5.0  0.02  exp7_easy  fc_easy   0.5"
    "15.0 0.05  exp7_middle fc_middle 0.5"
    "30.0 0.10  exp7_hard  fc_hard   0.5"
    "45.0 0.20  exp7_vhard fc_vhard  0.01"
)

for entry in "${DIFFICULTIES[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    MAX_ANGLE=$1
    SIGMA=$2
    DR_TAG=$3
    FC_TAG=$4
    FC_LR=$5

    echo ""
    echo "------------------------------------------------------------"
    echo " Difficulty: max_angle=${MAX_ANGLE}°  sigma=${SIGMA}   $(date)"
    echo "------------------------------------------------------------"

    run_deep_r "$DR_TAG" "$MAX_ANGLE" "$SIGMA"
    run_fc     "$FC_TAG" "$MAX_ANGLE" "$SIGMA" "$FC_LR"
done

# ---------------------------------------------------------------------------
# Generate the comparison plot
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo " All training runs complete.  $(date)"
echo " Generating comparison plot..."
echo "============================================================"

# Collect all available augmented result files (excludes *_smoke*)
AUG_ARGS=()
for f in results/exp7*_results.pickle results/fc*_results.pickle; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f")
    # skip smoke-test files
    if [[ "$base" == *smoke* ]]; then continue; fi
    AUG_ARGS+=(--aug "$f")
    echo "  + $f"
done

# Build the plot command
PLOT_CMD=(python plot_comparison.py --clean results/deep_r_results.pickle)
if [[ ${#AUG_ARGS[@]} -gt 0 ]]; then
    PLOT_CMD+=("${AUG_ARGS[@]}")
fi
PLOT_CMD+=(--out results/exp7_full_comparison.png)

# shellcheck disable=SC2068
"${PLOT_CMD[@]}"

echo ""
echo "============================================================"
echo " Done.  Plot saved to: results/exp7_full_comparison.png"
echo " $(date)"
echo "============================================================"

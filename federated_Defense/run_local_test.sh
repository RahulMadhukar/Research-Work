#!/bin/bash
# Local testing script (for running on your laptop/desktop, not SLURM)

echo "========================================================================"
echo "Local Test Started: $(date)"
echo "========================================================================"

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"

# Base paths
PLOTS_DIR="${SCRIPT_DIR}/plots"

# Find or create run_id directory
EXISTING_RUN_ID=""
if [ -d "${PLOTS_DIR}" ]; then
    # Find the most recent directory with checkpoints
    EXISTING_RUN_ID=$(ls -1t "${PLOTS_DIR}" 2>/dev/null | grep -E '^[0-9]{8}_[0-9]{6}$' | head -1)
fi

if [ -n "$EXISTING_RUN_ID" ]; then
    RUN_ID="$EXISTING_RUN_ID"
    echo "[INFO] Found existing run_id: $RUN_ID"
else
    RUN_ID=$(date +"%Y%m%d_%H%M%S")
    echo "[INFO] Creating new run_id: $RUN_ID"
fi

# Create results directory for this run
RESULTS_DIR="${PLOTS_DIR}/${RUN_ID}/results"
mkdir -p "${RESULTS_DIR}"

# Define log file paths
TRAINING_LOG="${RESULTS_DIR}/training_log_local.txt"
STATUS_LOG="${RESULTS_DIR}/training_status.log"

echo "Results directory: ${RESULTS_DIR}"
echo "Training log: ${TRAINING_LOG}"
echo "========================================================================"

# Run with checkpoint resume enabled (small test)
echo "Running small test with checkpoint resume..."
echo ""

# Automatically answer "yes" to resume prompt
echo "yes" | python3 -u main.py \
  --mode comprehensive \
  --resume-checkpoint \
  --checkpoint-interval 2 \
  --clients 5 \
  --rounds 3 \
  --subset 0.01 \
  --outdir plots \
  2>&1 | tee "${TRAINING_LOG}"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Local Test Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================================================"

# Log completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully!" | tee -a "${STATUS_LOG}"
else
    echo "⚠ Test failed with exit code $EXIT_CODE" | tee -a "${STATUS_LOG}"
fi

echo "All logs saved to: ${RESULTS_DIR}"

exit $EXIT_CODE

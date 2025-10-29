#!/bin/bash
#SBATCH --job-name=prjt1_resume              # Job name
#SBATCH --output=Output_resume_%j.txt        # Output file with job ID (temp location)
#SBATCH --error=error_resume_%j.txt          # Error file with job ID (temp location)
#SBATCH --ntasks=1                           # Run a single task
#SBATCH --time=11:59:59                      # Time limit hh:mm:ss
#SBATCH --mem=120G                           # Memory limit
#SBATCH --partition=gpu-a100                 # Specify the gpu-a100 partition
#SBATCH --gres=gpu:1                         # Request one GPU
#SBATCH --nodelist=node1                     # Ensure it uses node1

# Load CUDA module
module load cuda/cuda-11.7

# Activate the Conda environment
source ~/.bashrc          # Ensure Conda is initialized
conda activate phdffl

# Force unbuffered Python output (important for SLURM logs)
export PYTHONUNBUFFERED=1

# Base paths
BASE_DIR="/export/home/siba/Rahul/project_1/federated_Defense"
PLOTS_DIR="${BASE_DIR}/plots"

echo "========================================================================"
echo "SLURM Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "========================================================================"

# Find or create run_id directory
# Look for the most recent run_id with checkpoints (for resume)
EXISTING_RUN_ID=""
if [ -d "${PLOTS_DIR}" ]; then
    # Find the most recent directory with checkpoints
    EXISTING_RUN_ID=$(ls -1t "${PLOTS_DIR}" | grep -E '^[0-9]{8}_[0-9]{6}$' | head -1)
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

# Define log file paths in results directory
TRAINING_LOG="${RESULTS_DIR}/training_log_${SLURM_JOB_ID}.txt"
STATUS_LOG="${RESULTS_DIR}/training_status.log"

echo "Results directory: ${RESULTS_DIR}"
echo "Training log: ${TRAINING_LOG}"
echo "========================================================================"

# Automatically answer "yes" to resume prompt (for non-interactive SLURM)
# This pipes "yes" to the Python script when it asks for resume confirmation
echo "yes" | python3 -u ${BASE_DIR}/main.py \
  --mode comprehensive \
  --resume-checkpoint \
  --checkpoint-interval 2 \
  --clients 10 \
  --rounds 20 \
  --subset 0.1 \
  --outdir ${PLOTS_DIR} \
  2>&1 | tee -a "${TRAINING_LOG}"

EXIT_CODE=$?

echo "========================================================================"
echo "SLURM Job Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "========================================================================"

# Move SLURM output/error files to results directory
if [ -f "Output_resume_${SLURM_JOB_ID}.txt" ]; then
    mv "Output_resume_${SLURM_JOB_ID}.txt" "${RESULTS_DIR}/"
    echo "Moved SLURM output to: ${RESULTS_DIR}/Output_resume_${SLURM_JOB_ID}.txt"
fi

if [ -f "error_resume_${SLURM_JOB_ID}.txt" ]; then
    mv "error_resume_${SLURM_JOB_ID}.txt" "${RESULTS_DIR}/"
    echo "Moved SLURM error to: ${RESULTS_DIR}/error_resume_${SLURM_JOB_ID}.txt"
fi

# Log completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!" | tee -a "${STATUS_LOG}"
else
    echo "⚠ Training interrupted (will resume on next run)" | tee -a "${STATUS_LOG}"
fi

echo "All logs saved to: ${RESULTS_DIR}"

exit $EXIT_CODE

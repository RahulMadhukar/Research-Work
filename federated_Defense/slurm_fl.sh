#!/bin/bash
#SBATCH --job-name=prjt1                     # Job name
#SBATCH --output=Output1.txt                 # Output file
#SBATCH --error=error1.txt                    # Error file
#SBATCH --ntasks=1                           # Run a single task
#SBATCH --time=11:59:59                      # Time limit hh:mm:ss
#SBATCH --mem=120G                           # Memory limit
#SBATCH --partition=gpu-h100                 # Specify the gpu-h100 partition
#SBATCH --gres=gpu:1                         # Request one GPU
#SBATCH --nodelist=node2                     # Ensure it uses node2, where gpu-h100 is available

# Load CUDA module
module load cuda/cuda-11.7

# Activate the Conda environment
source ~/.bashrc          # Ensure Conda is initialized
conda activate phdffl

# Force unbuffered Python output (important for SLURM logs)
export PYTHONUNBUFFERED=1

# Run your script with unbuffered mode (-u)
python3 -u /export/home/siba/Rahul/project_1/federated_Defense/main.py

  2>&1 | tee Output1.txt

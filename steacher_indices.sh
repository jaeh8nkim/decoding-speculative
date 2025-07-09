#!/bin/bash
#SBATCH -J steacher_indices      # Job name
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --output=steacher_%j.out # Output file (%j = job ID)
#SBATCH --error=steacher_%j.err  # Error file
#SBATCH --time 0-23:00:00        # Maximum runtime (days-hours:minutes:seconds)

# Activate conda environment
conda activate rsd

# Check if index range is provided
if [ $# -eq 0 ]; then
    echo "Usage: sbatch steacher_indices.sh <INDEX_RANGE>"
    echo "Example: sbatch steacher_indices.sh \"111:121,131,134:173\""
    exit 1
fi

INDEX_RANGE=$1

echo "Processing indices: $INDEX_RANGE"
echo "Starting steacher_indices.py..."

# Run the Python script with the provided index range
# GPU index defaults to 0 (assigned by SLURM)
python -u steacher_indices.py "$INDEX_RANGE"

echo "Task completed!"
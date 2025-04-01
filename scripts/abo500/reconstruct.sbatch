#!/bin/bash
#SBATCH --job-name=nerf_reconstruct
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX3090:10

# Set up environment variables
export HF_HOME=/n/fs/scratch/${USER}/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/${USER}/nerf2physics/torch_cache

# Activate conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Navigate to the directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Add timing information
echo "[$(date)] Starting NeRF reconstruction process"
START_TIME=$(date +%s)

# Run the script with arguments
# You can modify scene_name and other parameters as needed
python ns_reconstruction.py --data_dir /n/fs/scratch/${USER}/nerf2physics/data/abo_500/

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] NeRF reconstruction completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"

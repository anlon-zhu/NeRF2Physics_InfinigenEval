#!/bin/bash
#SBATCH --job-name=nerf_reconstruct
#SBATCH --output=/n/fs/scratch/az4244/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/az4244/nerf2physics/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Set up environment variables
export HF_HOME=/n/fs/scratch/az4244/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/az4244/nerf2physics/torch_cache

# Activate conda environment
module load anaconda3
export PATH=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy/bin:$PATH

# Navigate to the directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Run the script with arguments
# You can modify scene_name and other parameters as needed
python ns_reconstruction.py --data_dir /n/fs/scratch/az4244/nerf2physics/data/abo_500/

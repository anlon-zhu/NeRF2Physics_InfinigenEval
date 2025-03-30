#!/bin/bash
#SBATCH --job-name=clip_fusion
#SBATCH --output=/n/fs/scratch/az4244/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/az4244/nerf2physics/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Set up environment variables
export HF_HOME=/n/fs/scratch/az4244/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/az4244/nerf2physics/torch_cache

# Activate conda environment
source /n/fs/pvl-progen/anlon/anaconda3/etc/profile.d/conda.sh
conda activate /n/fs/pvl-progen/anlon/envs/nerf2phy

# Navigate to the directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Run the script with arguments
# You can modify scene_name and other parameters as needed
python feature_fusion.py --end_idx 1 --data_dir /n/fs/scratch/az4244/nerf2physics/data/abo_500/

#!/bin/bash
#SBATCH --job-name=captioning
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
module load anaconda3
export PATH=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy/bin:$PATH

# Navigate to the directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Run the script with arguments
# You can modify scene_name and other parameters as needed
python captioning.py --end_idx 1 --data_dir /n/fs/scratch/az4244/nerf2physics/data/abo_500/ --blip2_model_dir /n/fs/scratch/az4244/nerf2physics/blip2-flan-t5-xl

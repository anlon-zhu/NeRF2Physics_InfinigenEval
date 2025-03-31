#!/bin/bash
#SBATCH --job-name=nerf_reconstruct
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <data_dir> [scene_id] [training_iters] [near_plane] [far_plane]"
    echo "Example: $0 /n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data 18015bf3 10000 0.4 6.0"
    exit 1
fi

# Set up environment variables
export HF_HOME=/n/fs/scratch/${USER}/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/${USER}/nerf2physics/torch_cache

DATA_DIR=$1
SCENE_ID=$2
TRAINING_ITERS=${3:-10000}  # Default to 10000 if not provided
NEAR_PLANE=${4:-0.4}        # Default to 0.4 if not provided
FAR_PLANE=${5:-6.0}         # Default to 6.0 if not provided

# Activate conda environment
module load anaconda3
export PATH=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy/bin:$PATH

# Navigate to the project directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# If scene_id is provided, process just that scene
if [ ! -z "$SCENE_ID" ]; then
    echo "Reconstructing single scene: $SCENE_ID"
    
    # Run NeRF Studio reconstruction
    python ns_reconstruction.py \
        --data_dir ${DATA_DIR} \
        --scene_id ${SCENE_ID} \
        --training_iters ${TRAINING_ITERS} \
        --near_plane ${NEAR_PLANE} \
        --far_plane ${FAR_PLANE} \
        --vis_mode tensorboard \
        --num_points 50000 \
        --bbox_size 1.0
    
# Otherwise, process all scenes in the data directory
else
    echo "Reconstructing all scenes in: $DATA_DIR"
    
    # Run NeRF Studio reconstruction
    python ns_reconstruction.py \
        --data_dir ${DATA_DIR} \
        --training_iters ${TRAINING_ITERS} \
        --near_plane ${NEAR_PLANE} \
        --far_plane ${FAR_PLANE} \
        --vis_mode tensorboard \
        --num_points 50000 \
        --bbox_size 1.0
fi

echo "NeRF reconstruction complete."

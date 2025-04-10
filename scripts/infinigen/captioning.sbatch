#!/bin/bash
#SBATCH --job-name=infinigen_captioning
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=az4244@princeton.edu

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <data_dir> [scene_name]"
    echo "Example: $0 /n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data 18015bf3"
    exit 1
fi

DATA_DIR=$1
SCENE_NAME=$2

# Set up environment variables
export HF_HOME=/n/fs/scratch/${USER}/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/${USER}/nerf2physics/torch_cache

# Force consistent precision for BLIP-2 model
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
# Force model to use half precision (FP16) instead of mixed precision
export TORCH_DTYPE="float16"

# Set PyTorch memory management variables to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Navigate to the directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Add timing information
echo "[$(date)] Starting Infinigen captioning process"
START_TIME=$(date +%s)

# If scene_name is provided, process just that scene
if [ ! -z "$SCENE_NAME" ]; then
    echo "Captioning single scene: $SCENE_NAME"
    
    # Run captioning on specific scene
    python captioning.py \
        --data_dir ${DATA_DIR} \
        --scene_name ${SCENE_NAME} \
        --blip2_model_dir /n/fs/scratch/${USER}/nerf2physics/blip2-flan-t5-xl
    
# Otherwise, process all scenes in the data directory
else
    echo "Captioning all scenes in: $DATA_DIR"
    
    # Run captioning on all scenes
    python captioning.py \
        --data_dir ${DATA_DIR} \
        --blip2_model_dir /n/fs/scratch/${USER}/nerf2physics/blip2-flan-t5-xl
fi

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Infinigen captioning process completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"


# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Infinigen captioning process completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"

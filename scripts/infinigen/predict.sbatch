#!/bin/bash
#SBATCH --job-name=infinigen_predict
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <data_dir> <property_name> [scene_name]"
    echo "Example: $0 /n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data density 18015bf3"
    exit 1
fi

DATA_DIR=$1
PROPERTY_NAME=$2
SCENE_NAME=$3

# Set up environment variables
export HF_HOME=/n/fs/scratch/${USER}/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/${USER}/nerf2physics/torch_cache

# Activate conda environment
module load anaconda3
export PATH=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy/bin:$PATH

# Navigate to the directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Add timing information
echo "[$(date)] Starting Infinigen property prediction process"
START_TIME=$(date +%s)

# If scene_name is provided, process just that scene
if [ ! -z "$SCENE_NAME" ]; then
    echo "Predicting property ${PROPERTY_NAME} for scene: $SCENE_NAME"
    
    # Run property prediction on specific scene
    python predict_property.py \
        --property_name ${PROPERTY_NAME} \
        --data_dir ${DATA_DIR} \
        --scene_name ${SCENE_NAME}
    
# Otherwise, process all scenes in the data directory
else
    echo "Predicting property ${PROPERTY_NAME} for all scenes in: $DATA_DIR"
    
    # Run property prediction on all scenes
    python predict_property.py \
        --property_name ${PROPERTY_NAME} \
        --data_dir ${DATA_DIR}
fi

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Infinigen property prediction process completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"

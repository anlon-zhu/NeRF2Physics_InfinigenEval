#!/bin/bash
#SBATCH --job-name=infinigen_material
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:RTX3090:10

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
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Navigate to the directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Add timing information
echo "[$(date)] Starting Infinigen material proposal process"
START_TIME=$(date +%s)

# If scene_name is provided, process just that scene
if [ ! -z "$SCENE_NAME" ]; then
    echo "Running material proposal for property ${PROPERTY_NAME} on scene: $SCENE_NAME"
    
    # Run material proposal on specific scene
    python material_proposal.py \
        --property_name ${PROPERTY_NAME} \
        --data_dir ${DATA_DIR} \
        --scene_name ${SCENE_NAME}
    
# Otherwise, process all scenes in the data directory
else
    echo "Running material proposal for property ${PROPERTY_NAME} on all scenes in: $DATA_DIR"
    
    # Run material proposal on all scenes
    python material_proposal.py \
        --property_name ${PROPERTY_NAME} \
        --data_dir ${DATA_DIR}
fi

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Infinigen material proposal process completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"

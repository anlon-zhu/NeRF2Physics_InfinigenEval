#!/bin/bash
#SBATCH --job-name=density_eval
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%A_%a.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=az4244@princeton.edu

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <data_dir>"
    echo "Example: $0 /n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data 18015bf3"
    exit 1
fi

DATA_DIR=$1

# Set up environment variables
export HF_HOME=/n/fs/scratch/${USER}/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/${USER}/nerf2physics/torch_cache
export OPEN3D_HEADLESS_RENDERING=ON

# Activate conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Navigate to the directory
echo "[$(date)] Starting batch density evaluation (Array Job ${SLURM_ARRAY_TASK_ID})"
START_TIME=$(date +%s)

# Create scene list if it doesn't exist
SCENE_LIST_FILE="${DATA_DIR}/scene_list.txt"

# Read scenes into an array
SCENE_NAMES=($(cat "${SCENE_LIST_FILE}"))
TOTAL_SCENES=${#SCENE_NAMES[@]}

if [ ${TOTAL_SCENES} -eq 0 ]; then
    echo "No scenes found in ${SCENE_LIST_FILE}"
    exit 1
fi

echo "Found ${TOTAL_SCENES} scenes in scene list file"

# Calculate which scenes this node should process
SCENES_PER_NODE=$(( (TOTAL_SCENES + 4) / 5 ))  # Ceiling division to distribute across 5 nodes
START_IDX=$(( SLURM_ARRAY_TASK_ID * SCENES_PER_NODE ))
END_IDX=$(( START_IDX + SCENES_PER_NODE - 1 ))

# Make sure END_IDX doesn't exceed the total number of scenes
if [ ${END_IDX} -ge ${TOTAL_SCENES} ]; then
    END_IDX=$(( TOTAL_SCENES - 1 ))
fi

echo "Processing scenes ${START_IDX} to ${END_IDX} out of ${TOTAL_SCENES} total scenes"

# Process each scene assigned to this node
for (( i=${START_IDX}; i<=${END_IDX}; i++ )); do
    SCENE_NAME=${SCENE_NAMES[$i]}
    
    echo "\n[$(date)] Processing scene ${i}/${END_IDX}: ${SCENE_NAME}"
    
    # Run density evaluation on this scene
    python density_evaluation.py \
        --data_dir ${DATA_DIR} \
        --scene_name ${SCENE_NAME}
    echo "[$(date)] Completed scene: ${SCENE_NAME}"
done

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Batch density evaluation completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"
exit 0

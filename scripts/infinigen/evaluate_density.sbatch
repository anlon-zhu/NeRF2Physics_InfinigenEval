#!/bin/bash
#SBATCH --job-name=density_eval
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%A_%a.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%A_%a.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
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
# Set TCNN CUDA architectures for optimal performance
# Use 60;61 for P100 GPUs, or 75;86 for RTX 2080/3090 GPUs
export TCNN_CUDA_ARCHITECTURES="75;86"  # Adjust as needed
export OPEN3D_HEADLESS_RENDERING=ON

# Make sure there's enough space for shader cache
export XDG_CACHE_HOME="/tmp/${USER}_cache"
mkdir -p $XDG_CACHE_HOME

# Activate conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH


# Navigate to the directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval
echo "[$(date)] Starting batch density evaluation (Array Job ${SLURM_ARRAY_TASK_ID})"
START_TIME=$(date +%s)

# Create scene list if it doesn't exist
SCENE_LIST_FILE="${DATA_DIR}/scene_list.txt"
if [ ! -f "${SCENE_LIST_FILE}" ]; then
    echo "Scene list file not found, creating it now..."
    SCENES_DIR="${DATA_DIR}/scenes"
    if [ ! -d "${SCENES_DIR}" ]; then
        echo "Error: Scenes directory not found at ${SCENES_DIR}"
        exit 1
    fi
    
    # Find all directories in the scenes directory
    find "${SCENES_DIR}" -maxdepth 1 -type d -not -path "${SCENES_DIR}" -exec basename {} \; > "${SCENE_LIST_FILE}"
    echo "Created scene list with $(wc -l < "${SCENE_LIST_FILE}") scenes"
fi

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
    
    # Check if ground truth density directory exists
    GT_DIR="${DATA_DIR}/scenes/${SCENE_NAME}/gt_density"
    if [ ! -d "$GT_DIR" ]; then
        echo "[WARNING] Ground truth density directory not found: $GT_DIR"
        echo "Skipping this scene. Run copy_density_files.sh first to prepare ground truth data."
        continue
    fi
    
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

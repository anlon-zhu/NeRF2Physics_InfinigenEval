#!/bin/bash
#SBATCH --job-name=colmap_convert
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=az4244@princeton.edu

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <infinigen_dir> <output_dir> [scene_id]"
    echo "Example: $0 /n/fs/scratch/${USER}/2025-03-30_02-32_density_gt_1 /n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data 18015bf3"
    exit 1
fi

INFINIGEN_DIR=$1
OUTPUT_DIR=$2
SCENE_ID=$3

# Create output and logs directories if they don't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p /n/fs/scratch/${USER}/nerf2physics/logs

# Activate conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Navigate to the project directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Add timing information
echo "[$(date)] Starting conversion process"
START_TIME=$(date +%s)

# If scene_id is provided, process just that scene
if [ ! -z "$SCENE_ID" ]; then
    echo "Converting single scene: $SCENE_ID"
    
    # Check if we're pointing to the scenes directory or a parent directory
    if [[ "${INFINIGEN_DIR}" == */scenes ]]; then
        # We're already pointing to the scenes directory
        scene_path="${INFINIGEN_DIR}/${SCENE_ID}"
    else
        # We're pointing to a parent directory that contains scenes/
        scene_path="${INFINIGEN_DIR}/scenes/${SCENE_ID}"
    fi
    
    echo "Scene path: $scene_path"
    
    # Convert to COLMAP format
    python infinigen_to_colmap.py \
        --input_dir ${scene_path} \
        --output_dir ${OUTPUT_DIR} \
        --scene_id ${SCENE_ID}
    
# Otherwise, process all scenes in the Infinigen directory
else
    echo "Processing all scenes in: $INFINIGEN_DIR"
    
    # Check if we're pointing to the scenes directory or a parent directory
    if [[ "${INFINIGEN_DIR}" == */scenes ]]; then
        # We're already pointing to the scenes directory
        scenes_dir="${INFINIGEN_DIR}"
    else
        # We're pointing to a parent directory that contains scenes/
        scenes_dir="${INFINIGEN_DIR}/scenes"
    fi
    
    echo "Scenes directory: $scenes_dir"
    
    # Process each subdirectory
    for scene_dir in ${scenes_dir}/*/; do
        scene=$(basename "$scene_dir")
        echo "Converting scene: $scene"
        
        # Convert to COLMAP format
        python infinigen_to_colmap.py \
            --input_dir ${scene_dir} \
            --output_dir ${OUTPUT_DIR} \
            --scene_id ${scene}
    done
fi

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Conversion completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"

echo "COLMAP conversion complete. Data is ready for NeRF reconstruction."

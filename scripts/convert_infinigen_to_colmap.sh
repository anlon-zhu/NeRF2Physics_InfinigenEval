#!/bin/bash
#SBATCH --job-name=colmap_convert
#SBATCH --output=/n/fs/scratch/%u/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <infinigen_dir> <output_dir> [scene_id]"
    echo "Example: $0 /n/fs/scratch/USER/2025-03-30_02-32_density_gt_1 /n/fs/scratch/USER/nerf_data 18015bf3"
    exit 1
fi

INFINIGEN_DIR=$1
OUTPUT_DIR=$2
SCENE_ID=$3

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Activate conda environment
module load anaconda3
export PATH=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy/bin:$PATH

# Navigate to the project directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# If scene_id is provided, process just that scene
if [ ! -z "$SCENE_ID" ]; then
    echo "Converting single scene: $SCENE_ID"
    
    # Convert to COLMAP format
    python infinigen_to_colmap.py \
        --input_dir ${INFINIGEN_DIR}/${SCENE_ID} \
        --output_dir ${OUTPUT_DIR} \
        --scene_id ${SCENE_ID}
    
# Otherwise, process all scenes in the Infinigen directory
else
    echo "Processing all scenes in: $INFINIGEN_DIR"
    
    # Process each subdirectory
    for scene_dir in ${INFINIGEN_DIR}/*/; do
        scene=$(basename "$scene_dir")
        echo "Converting scene: $scene"
        
        # Convert to COLMAP format
        python infinigen_to_colmap.py \
            --input_dir ${scene_dir} \
            --output_dir ${OUTPUT_DIR} \
            --scene_id ${scene}
    done
fi

echo "COLMAP conversion complete. Data is ready for NeRF reconstruction."

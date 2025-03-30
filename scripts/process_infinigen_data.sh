#!/bin/bash
#SBATCH --job-name=nerf_recon
#SBATCH --account=pvl
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/n/fs/scratch/%u/nerf_recon_%j.log

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <infinigen_dir> <output_dir> [scene_id]"
    echo "Example: $0 /n/fs/scratch/USER/2025-03-30_02-32_density_gt_1 /n/fs/scratch/USER/nerf_data 18015bf3"
    exit 1
fi

INFINIGEN_DIR=$1
OUTPUT_DIR=$2
SCENE_ID=$3

# If scene_id is provided, process just that scene
if [ ! -z "$SCENE_ID" ]; then
    echo "Processing single scene: $SCENE_ID"
    
    # Convert to COLMAP format
    python infinigen_to_colmap.py \
        --input_dir ${INFINIGEN_DIR}/${SCENE_ID} \
        --output_dir ${OUTPUT_DIR} \
        --scene_id ${SCENE_ID}
        
    # Run NeRF Studio reconstruction
    python ns_reconstruction.py \
        --data_dir ${OUTPUT_DIR} \
        --training_iters 20000 \
        --near_plane 0.4 \
        --far_plane 6.0 \
        --vis_mode tensorboard \
        --num_points 100000 \
        --bbox_size 1.0
    
# Otherwise, process all scenes in the Infinigen directory
else
    echo "Processing all scenes in ${INFINIGEN_DIR}"
    
    # Find all scene directories (immediate subdirectories that are not files)
    for SCENE_DIR in ${INFINIGEN_DIR}/*/; do
        if [ -d "$SCENE_DIR" ] && [ -d "${SCENE_DIR}/frames" ]; then
            SCENE_ID=$(basename ${SCENE_DIR})
            
            # Skip directories that are part of the job management
            if [[ "$SCENE_ID" == "datagen_command.sh" || "$SCENE_ID" == "finished_seeds.txt" || "$SCENE_ID" == "index.html" || "$SCENE_ID" == "jobs.log" || "$SCENE_ID" == "scenes_db.csv" ]]; then
                continue
            fi
            
            echo "Processing scene: $SCENE_ID"
            
            # Convert to COLMAP format
            python infinigen_to_colmap.py \
                --input_dir ${SCENE_DIR} \
                --output_dir ${OUTPUT_DIR} \
                --scene_id ${SCENE_ID}
                
            # Run NeRF Studio reconstruction
            python ns_reconstruction.py \
                --data_dir ${OUTPUT_DIR} \
                --training_iters 20000 \
                --near_plane 0.4 \
                --far_plane 6.0 \
                --vis_mode tensorboard \
                --num_points 100000 \
                --bbox_size 1.0
        fi
    done
fi

echo "All processing complete"
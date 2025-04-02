#!/bin/bash

# Check arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <output_base_dir> <infinigen_dir1> [<infinigen_dir2> ...]"
    echo "Example: $0 /n/fs/scratch/${USER}/nerf2physics/data /n/fs/scratch/${USER}/2025-03-30_02-32_density_gt_1 /n/fs/scratch/${USER}/2025-03-31_01-32_density_gt_1"
    exit 1
fi

OUTPUT_BASE=$1
shift

# Remaining arguments are the infinigen directories
INFINIGEN_DIRS=("$@")

# Create output directories
mkdir -p ${OUTPUT_BASE}/scenes
mkdir -p ${OUTPUT_BASE}/logs

echo "[$(date)] Starting organization process"

# Process each infinigen output directory
for infinigen_dir in "${INFINIGEN_DIRS[@]}"; do
    if [ ! -d "$infinigen_dir" ]; then
        echo "Warning: $infinigen_dir does not exist, skipping"
        continue
    fi
    
    echo "Processing $infinigen_dir"
    
    # Get all scene directories (excluding special files/directories)
    for scene_dir in "$infinigen_dir"/*; do
        # Skip non-directories and special files
        if [ ! -d "$scene_dir" ] || [[ $(basename "$scene_dir") == *.* ]] || 
           [[ $(basename "$scene_dir") == "logs" ]] || [[ $(basename "$scene_dir") == "index.html" ]]; then
            continue
        fi
        
        scene_id=$(basename "$scene_dir")
        echo "  Processing scene: $scene_id"
        
        # Create output directories for this scene
        scene_output_dir="${OUTPUT_BASE}/scenes/${scene_id}"
        mkdir -p "${scene_output_dir}/images"
        mkdir -p "${scene_output_dir}/gt_density"
        mkdir -p "${scene_output_dir}/camview"
        
        # Check if frames directory exists
        if [ ! -d "${scene_dir}/frames" ]; then
            echo "    Warning: No frames directory found in ${scene_dir}, skipping"
            continue
        fi
        
        # Copy camera parameters for COLMAP
        if [ -d "${scene_dir}/frames/camview" ]; then
            echo "    Copying camera parameters for COLMAP conversion"
            for camera_dir in "${scene_dir}/frames/camview"/*; do
                if [ -d "$camera_dir" ]; then
                    camera_id=$(basename "$camera_dir")
                    mkdir -p "${scene_output_dir}/camview/${camera_id}"
                    
                    # Copy all NPZ files (camera parameters)
                    find "$camera_dir" -name "*.npz" -exec cp {} "${scene_output_dir}/camview/${camera_id}/" \;
                fi
            done
        else
            echo "    Warning: No camview directory found in ${scene_dir}/frames, skipping camera parameters"
        fi
        
        # Copy RGB images for COLMAP
        if [ -d "${scene_dir}/frames/Image" ]; then
            echo "    Copying RGB images for COLMAP conversion"
            for camera_dir in "${scene_dir}/frames/Image"/*; do
                if [ -d "$camera_dir" ]; then
                    camera_id=$(basename "$camera_dir")
                    mkdir -p "${scene_output_dir}/images/${camera_id}"
                    
                    # Copy all PNG images
                    find "$camera_dir" -name "*.png" -exec cp {} "${scene_output_dir}/images/${camera_id}/" \;
                fi
            done
        else
            echo "    Warning: No Image directory found in ${scene_dir}/frames, skipping RGB images"
        fi
        
        # Copy density ground truth images
        if [ -d "${scene_dir}/frames/MaterialsDensity" ]; then
            echo "    Copying density ground truth images"
            for camera_dir in "${scene_dir}/frames/MaterialsDensity"/*; do
                if [ -d "$camera_dir" ]; then
                    camera_id=$(basename "$camera_dir")
                    
                    # Copy density files
                    count=0
                    for density_file in "$camera_dir"/*.npy; do
                        if [ -f "$density_file" ]; then
                            cp "$density_file" "${scene_output_dir}/gt_density/density_$(printf "%03d" $count).npy"
                            
                            # Also copy corresponding PNG if available
                            png_file="${density_file%.npy}.png"
                            if [ -f "$png_file" ]; then
                                cp "$png_file" "${scene_output_dir}/gt_density/density_$(printf "%03d" $count).png"
                            fi
                            
                            count=$((count+1))
                        fi
                    done
                    
                    echo "    Copied $count density files for camera $camera_id"
                fi
            done
        else
            echo "    Warning: No MaterialsDensity directory found in ${scene_dir}/frames, skipping density files"
        fi
    done
done

echo "[$(date)] Organization process complete"
echo
echo "You can now run convert_infinigen_to_colmap.sh as follows:"
echo "sbatch scripts/infinigen/convert_infinigen_to_colmap.sh ${OUTPUT_BASE} ${OUTPUT_BASE}/colmap_scenes <scene_id>"
echo
echo "And later, evaluate density as follows:"
echo "sbatch scripts/infinigen/evaluate_density.sbatch ${OUTPUT_BASE} <scene_id>"
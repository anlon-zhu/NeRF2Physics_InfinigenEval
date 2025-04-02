#!/bin/bash

# Check arguments
if [ "$#" -lt 2 ]; then
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

# Initialize arrays for metadata tracking
declare -a included_seeds
declare -a skipped_seeds
declare -a source_dirs

echo "[$(date)] Starting organization process"
TOTAL_SEEDS=0
SKIPPED_SEEDS=0

# Check if metadata.json already exists and load existing seeds
METADATA_FILE="${OUTPUT_BASE}/metadata.json"
if [ -f "$METADATA_FILE" ]; then
    echo "Found existing metadata.json, will update it with new seeds"
    
    # Extract existing seeds into an associative array for quick lookup
    declare -A existing_seeds
    
    # Use jq if available, otherwise fallback to grep + sed
    if command -v jq &> /dev/null; then
        mapfile -t existing_seed_list < <(jq -r '.included_seeds[].seed' "$METADATA_FILE")
        mapfile -t existing_source_list < <(jq -r '.included_seeds[].source' "$METADATA_FILE")
        
        # Load existing included seeds
        for i in "${!existing_seed_list[@]}"; do
            seed_id="${existing_seed_list[$i]}"
            source_id="${existing_source_list[$i]}"
            existing_seeds["$seed_id"]="$source_id"
            included_seeds+=("$seed_id")
            source_dirs+=("$source_id")
        done
        
        # Load existing skipped seeds
        mapfile -t existing_skipped < <(jq -r '.skipped_seeds[]' "$METADATA_FILE" 2>/dev/null || echo "")
        for skipped in "${existing_skipped[@]}"; do
            if [ ! -z "$skipped" ]; then
                skipped_seeds+=("$skipped")
                SKIPPED_SEEDS=$((SKIPPED_SEEDS+1))
            fi
        done
    else
        echo "Warning: jq not found, using basic text processing to parse existing metadata"
        # Basic parsing with grep (this is less reliable than jq)
        while IFS=: read -r key value; do
            seed=$(echo "$key" | grep -o '"seed"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/"seed"[[:space:]]*:[[:space:]]*"\([^"]*\)"/\1/')
            source=$(echo "$value" | grep -o '"source"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/"source"[[:space:]]*:[[:space:]]*"\([^"]*\)"/\1/')
            if [ ! -z "$seed" ] && [ ! -z "$source" ]; then
                existing_seeds["$seed"]="$source"
                included_seeds+=("$seed")
                source_dirs+=("$source")
            fi
        done < "$METADATA_FILE"
    fi
    
    echo "Loaded ${#included_seeds[@]} existing seeds and ${#skipped_seeds[@]} skipped seeds from metadata"
fi

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
        
        TOTAL_SEEDS=$((TOTAL_SEEDS+1))
        
        scene_id=$(basename "$scene_dir")
        
        # Skip this scene if it already exists in metadata
        if [ -v existing_seeds["$scene_id"] ]; then
            echo "  Scene $scene_id already processed, skipping"
            continue
        fi
        
        echo "  Processing scene: $scene_id"
        
        # Create output directories for this scene
        scene_output_dir="${OUTPUT_BASE}/scenes/${scene_id}"
        mkdir -p "${scene_output_dir}/infinigen_images"
        mkdir -p "${scene_output_dir}/gt_density"
        mkdir -p "${scene_output_dir}/camview"
        
        # Check if frames directory exists
        if [ ! -d "${scene_dir}/frames" ]; then
            echo "    Warning: No frames directory found in ${scene_dir}, skipping"
            skipped_seeds+=("$scene_id")
            SKIPPED_SEEDS=$((SKIPPED_SEEDS+1))
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
                    mkdir -p "${scene_output_dir}/infinigen_images/${camera_id}"
                    
                    # Copy all PNG images
                    find "$camera_dir" -name "*.png" -exec cp {} "${scene_output_dir}/infinigen_images/${camera_id}/" \;
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
        
        # Record this seed as successfully included
        included_seeds+=("$scene_id")
        source_dirs+=("$(basename "$infinigen_dir")")
    done
done

# Create/update metadata.json file
cat > "${OUTPUT_BASE}/metadata.json.new" << EOL
{
  "included_seeds": [
$(for i in "${!included_seeds[@]}"; do
    echo "    {
      \"seed\": \"${included_seeds[$i]}\",
      \"source\": \"${source_dirs[$i]}\"
    }$([ $i -lt $((${#included_seeds[@]}-1)) ] && echo ",")"
done)
  ],
  "skipped_seeds": [
$(for i in "${!skipped_seeds[@]}"; do
    echo "    \"${skipped_seeds[$i]}\"$([ $i -lt $((${#skipped_seeds[@]}-1)) ] && echo ",")"
done)
  ],
  "total_seeds": $TOTAL_SEEDS,
  "included_count": ${#included_seeds[@]},
  "skipped_count": $SKIPPED_SEEDS,
  "creation_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOL

# Move the new metadata file into place
mv "${OUTPUT_BASE}/metadata.json.new" "${OUTPUT_BASE}/metadata.json"

echo "[$(date)] Organization process complete"
echo "Updated metadata.json with information about ${#included_seeds[@]} included and ${#skipped_seeds[@]} skipped seeds."
echo "Processed $TOTAL_SEEDS new seeds, skipped $SKIPPED_SEEDS."
echo
echo "You can now run convert_infinigen_to_colmap.sh as follows:"
echo "sbatch scripts/infinigen/convert_infinigen_to_colmap.sh ${OUTPUT_BASE} ${OUTPUT_BASE}/colmap_scenes <scene_id>"
echo
echo "And later, evaluate density as follows:"
echo "sbatch scripts/infinigen/evaluate_density.sbatch ${OUTPUT_BASE} <scene_id>"
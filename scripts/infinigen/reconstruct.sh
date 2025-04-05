#!/bin/bash
#SBATCH --job-name=nerf_reconstruct
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=az4244@princeton.edu

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <data_dir> [training_iters] [scene_name]"
    echo "Example: $0 /n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data 10000 18015bf3"
    exit 1
fi

# Set up environment variables
export HF_HOME=/n/fs/scratch/${USER}/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/${USER}/nerf2physics/torch_cache

DATA_DIR=$1
TRAINING_ITERS=${2:-10000}  # Default to 10000 if not provided
SCENE_NAME=$3

# Activate conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Navigate to the project directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Add timing information
echo "[$(date)] Starting NeRF reconstruction process"
START_TIME=$(date +%s)

# Create/initialize metadata file if it doesn't exist
METADATA_FILE="${DATA_DIR}/metadata.json"
if [ ! -f "$METADATA_FILE" ]; then
    echo "Creating new metadata file at $METADATA_FILE"
    cat > "${METADATA_FILE}" << EOL
{
  "included_seeds": [],
  "skipped_seeds": [],
  "reconstructed_seeds": [],
  "total_seeds": 0,
  "included_count": 0,
  "skipped_count": 0,
  "reconstructed_count": 0,
  "creation_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOL
fi

# Function to check if scene has been reconstructed. Checks both metadata and directory
is_scene_reconstructed() {
    local scene=$1
    
    # First check if it's marked in metadata
    if command -v jq &> /dev/null; then
        jq -e ".reconstructed_seeds | index(\"$scene\")" "$METADATA_FILE" > /dev/null
        local metadata_result=$?
        if [ $metadata_result -eq 0 ]; then
            echo "Scene $scene found in metadata as reconstructed"
            return 0  # Already marked as reconstructed
        fi
    else
        grep -q "\"$scene\"" "$METADATA_FILE"
        local metadata_result=$?
        if [ $metadata_result -eq 0 ]; then
            echo "Scene $scene found in metadata as reconstructed"
            return 0  # Already marked as reconstructed
        fi
    fi
    
    return 1  # Scene is not reconstructed
}


# Function to update metadata with reconstructed scene
update_metadata_with_reconstructed() {
    local scene=$1
    local temp_file="${METADATA_FILE}.tmp"
    
    if command -v jq &> /dev/null; then
        # Use jq if available (more reliable)
        jq --arg scene "$scene" \
           '.reconstructed_seeds += [$scene] | .reconstructed_count = (.reconstructed_seeds | length)' \
           "$METADATA_FILE" > "$temp_file"
    else
        # Fallback to basic text manipulation (less reliable)
        # Convert to properly formatted JSON array entry
        sed -E "s/\"reconstructed_seeds\": \[/\"reconstructed_seeds\": \[\"$scene\", /g" "$METADATA_FILE" > "$temp_file"
        # Update count
        count=$(grep -o '"reconstructed_seeds"' "$temp_file" | wc -l)
        sed -E "s/\"reconstructed_count\": [0-9]+/\"reconstructed_count\": $count/g" "$temp_file" > "${temp_file}.2"
        mv "${temp_file}.2" "$temp_file"
    fi
    
    # Move the updated file in place
    mv "$temp_file" "$METADATA_FILE"
    echo "Updated metadata for reconstructed scene: $scene"
}

# If scene_name is provided, process just that scene
if [ ! -z "$SCENE_NAME" ]; then
    echo "Checking reconstruction status for: $SCENE_NAME"
    
    # Check if this scene has already been reconstructed
    if is_scene_reconstructed "$SCENE_NAME"; then
        echo "Scene $SCENE_NAME has already been reconstructed, skipping"
    else
        echo "Reconstructing single scene: $SCENE_NAME"
        
        # Run NeRF Studio reconstruction
        python ns_reconstruction.py \
            --data_dir ${DATA_DIR} \
            --scene_name ${SCENE_NAME} \
            --training_iters ${TRAINING_ITERS} \
            --near_plane ${NEAR_PLANE} \
            --far_plane ${FAR_PLANE} \
            --vis_mode tensorboard \
            --num_points 50000 \
            --bbox_size 1.0
        
        # Update metadata for this scene
        if [ $? -eq 0 ]; then
            update_metadata_with_reconstructed "$SCENE_NAME"
        fi
    fi
    
# Otherwise, process all scenes in the data directory
else
    echo "Reconstructing all scenes in: $DATA_DIR"
    
    # Get list of all scenes
    scenes_dir="${DATA_DIR}/scenes"
    if [ ! -d "$scenes_dir" ]; then
        echo "Error: Scenes directory $scenes_dir not found"
        exit 1
    fi
    
    # Process each scene
    for scene_dir in "$scenes_dir"/*; do
        if [ -d "$scene_dir" ]; then
            scene=$(basename "$scene_dir")
            
            # Check if this scene has already been reconstructed
            if is_scene_reconstructed "$scene"; then
                echo "Scene $scene has already been reconstructed, skipping"
                continue
            fi
            
            echo "Reconstructing scene: $scene"
            
            # Run NeRF Studio reconstruction
            python ns_reconstruction.py \
                --data_dir ${DATA_DIR} \
                --scene_name ${scene} \
                --training_iters ${TRAINING_ITERS} \
                --num_points 50000 \
                --bbox_size 1.0
            
            # Update metadata for this scene
            if [ $? -eq 0 ]; then
                update_metadata_with_reconstructed "$scene"
            fi
        fi
    done
fi

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] NeRF reconstruction completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"

echo "NeRF reconstruction complete."

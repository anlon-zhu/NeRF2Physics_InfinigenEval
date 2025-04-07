#!/bin/bash
#SBATCH --job-name=copy_density
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=az4244@princeton.edu

# Activate conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Define the output directory
output_dir="/n/fs/scratch/az4244/nerf2physics/infinigen_nerf_data/scenes"

# Define the list of input directories
input_dirs=(
    "/n/fs/scratch/az4244/mvs_30_renders/2025-03-30_02-32_density_gt_1/"
    "/n/fs/scratch/az4244/mvs_30_renders/2025-03-31_01-32_density_gt_1/"
    "/n/fs/scratch/az4244/mvs_30_renders/2025-04-01_19-22_density_gt_1/"
    "/n/fs/scratch/az4244/mvs_30_renders/2025-04-02_01-09_density_gt_1/"
)

# Function to copy density files from a scene directory to the output directory
copy_density_files() {
    local input_dir="$1"
    local output_base_dir="$2"
    
    echo "Processing input directory: $input_dir"
    
    # Check if this is a scene directory (contains frames/)
    if [ -d "${input_dir}/frames" ]; then
        # This is a single scene directory
        process_single_scene "$input_dir" "$output_base_dir"
    else
        # This might be a directory containing multiple scenes
        echo "Checking for multiple scenes in $input_dir"
        
        # Loop through subdirectories
        for scene_dir in "$input_dir"/*; do
            if [ -d "${scene_dir}/frames" ]; then
                echo "Found scene directory: $scene_dir"
                process_single_scene "$scene_dir" "$output_base_dir"
            fi
        done
    fi
}

# Process a single scene directory
process_single_scene() {
    local scene_dir="$1"
    local output_base_dir="$2"
    
    # Get scene name from directory name
    local scene_name=$(basename "$scene_dir")
    echo "Processing scene: $scene_name"
    
    # Check for MaterialsDensity directory
    local density_dir="${scene_dir}/frames/MaterialsDensity/camera_0"
    if [ ! -d "$density_dir" ]; then
        echo "Warning: MaterialsDensity directory not found at $density_dir"
        return
    fi
    
    # Create output directory for this scene
    local scene_output_dir="${output_base_dir}/${scene_name}"
    local gt_density_dir="${scene_output_dir}/gt_density"
    mkdir -p "$gt_density_dir"
    
    echo "Copying density files to $gt_density_dir"
    
    # Find and copy all .npy files from MaterialsDensity
    local count=0
    for density_file in "$density_dir"/*.npy; do
        if [ -f "$density_file" ]; then
            # Extract view ID from filename (e.g., MaterialsDensity_10_0_0048_0.npy -> 10)
            local filename=$(basename "$density_file")
            local view_id=$(echo "$filename" | grep -oP 'MaterialsDensity_\K\d+(?=_\d+_\d+_\d+\.npy)' || echo "unknown")
            
            if [ "$view_id" != "unknown" ]; then
                # Create a standardized output filename
                local output_file="${gt_density_dir}/density_${view_id}.npy"
                
                # Copy the file
                cp "$density_file" "$output_file"
                count=$((count + 1))
            else
                echo "Warning: Could not extract view ID from filename: $filename"
            fi
        fi
    done
    
    echo "Copied $count density files for scene $scene_name"
}

# Iterate over the input directories
for input_dir in "${input_dirs[@]}"; do
    copy_density_files "$input_dir" "$output_dir"
done

echo "Finished processing all input directories."

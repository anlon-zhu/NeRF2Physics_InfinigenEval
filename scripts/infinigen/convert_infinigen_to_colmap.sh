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

# Iterate over the input directories
for input_dir in "${input_dirs[@]}"; do
    # Print the current input directory
    echo "Processing input directory: $input_dir"

    # Call the python script
    python infinigen_to_colmap.py --input_dir "$input_dir" --output_dir "$output_dir"

    # Check the exit code
    if [ $? -ne 0 ]; then
        echo "Error: infinigen_to_colmap.py failed for input directory: $input_dir"
        exit 1
    fi
done

echo "Finished processing all input directories."
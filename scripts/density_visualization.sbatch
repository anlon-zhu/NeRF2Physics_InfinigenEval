#!/bin/bash
#SBATCH --job-name=density_viz
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=az4244@princeton.edu

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <data_dir> [output_dir] [case_study_scene]"
    echo "Example: $0 /n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data /n/fs/scratch/${USER}/nerf2physics/visualization_results scene_42"
    exit 1
fi

DATA_DIR=$1
OUTPUT_DIR=${2:-"/n/fs/scratch/${USER}/nerf2physics/visualization_results"}
CASE_STUDY_SCENE=${3:-"scene_42"}

echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Case study scene: $CASE_STUDY_SCENE"

# Set up environment variables
export HF_HOME=/n/fs/scratch/${USER}/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/${USER}/nerf2physics/torch_cache
# Set TCNN CUDA architectures for optimal performance
# Use 60;61 for P100 GPUs, or 75;86 for RTX 2080/3090 GPUs
export TCNN_CUDA_ARCHITECTURES="75;86"  # For RTX 2080
export OPEN3D_HEADLESS_RENDERING=ON

# Make sure there's enough space for shader cache
export XDG_CACHE_HOME="/tmp/${USER}_cache"
mkdir -p $XDG_CACHE_HOME

# Activate conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Navigate to the project directory
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

echo "[$(date)] Starting density visualization"
START_TIME=$(date +%s)

# Run the visualization script
python density_visualization.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --case_study_scene $CASE_STUDY_SCENE \
    --view_idx 0

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Density visualization completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"
echo "Results saved to: $OUTPUT_DIR"

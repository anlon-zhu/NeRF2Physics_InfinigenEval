#!/bin/bash
#SBATCH --job-name=density_eval
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=az4244@princeton.edu

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <data_dir> <scene_name> [cmap_min] [cmap_max]"
    echo "Example: $0 /n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data 18015bf3 0 5000"
    exit 1
fi

DATA_DIR=$1
SCENE_NAME=$2
# Default color map range values if not specified
CMAP_MIN=${3:-0}
CMAP_MAX=${4:-5000}

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

# Check if the scene directory exists
SCENE_DIR="${DATA_DIR}/scenes/${SCENE_NAME}"
if [ ! -d "$SCENE_DIR" ]; then
    echo "[ERROR] Scene directory not found: $SCENE_DIR"
    exit 1
fi

# Check if point cloud file exists
PCD_FILE="${SCENE_DIR}/ns/point_cloud.ply"
if [ ! -f "$PCD_FILE" ]; then
    echo "[ERROR] Point cloud file not found: $PCD_FILE"
    echo "Make sure the NeRF reconstruction has been completed for this scene."
    exit 1
fi

# Check if our point-wise density predictions exist
DENSITY_DIR="${SCENE_DIR}/point_densities"
if [ ! -d "$DENSITY_DIR" ]; then
    echo "[WARNING] Point densities directory not found: $DENSITY_DIR"
    echo "You should run the point density prediction script first:"
    echo "sbatch scripts/infinigen/predict.sbatch $DATA_DIR density $SCENE_NAME"
    
    # Prompt to continue without density predictions
    read -p "Do you want to proceed anyway? Density predictions will be calculated on-the-fly. [y/N] " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if ground truth density directory exists
GT_DIR="${SCENE_DIR}/gt_density"
if [ ! -d "$GT_DIR" ]; then
    echo "[WARNING] Ground truth density directory not found: $GT_DIR"
    echo "Will run visualization only, no evaluation metrics will be calculated."
    echo "To enable evaluation, place ground truth density images in: $GT_DIR"
    echo "The images should be named: density_000.png, density_001.png, etc."
fi

# Check if evaluation results already exist
OUTPUT_DIR="${SCENE_DIR}/density_evaluation"
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo "Evaluation already completed, skipping..."
    echo "Results are available at: ${OUTPUT_DIR}"
    exit 0
fi

# Add timing information
echo "[$(date)] Starting density evaluation for scene: $SCENE_NAME"
START_TIME=$(date +%s)

# Run the density evaluation script
python density_evaluation.py \
    --data_dir ${DATA_DIR} \
    --scene_name ${SCENE_NAME} \
    --property_name density \
    --cmap_min ${CMAP_MIN} \
    --cmap_max ${CMAP_MAX}

# Record end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Density evaluation completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"

# Show output directory location
OUTPUT_DIR="${SCENE_DIR}/density_evaluation"
echo "Evaluation results saved to: $OUTPUT_DIR"

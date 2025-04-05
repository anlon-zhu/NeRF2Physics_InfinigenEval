#!/bin/bash
#SBATCH --job-name=nerf_reconstruct
#SBATCH --output=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.out
#SBATCH --error=/n/fs/scratch/%u/nerf2physics/logs/%x_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_2080:1
#SBATCH --array=0-4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=az4244@princeton.edu

# Set up environment
export HF_HOME=/n/fs/scratch/${USER}/nerf2physics/hf_cache
export TORCH_HOME=/n/fs/scratch/${USER}/nerf2physics/torch_cache

module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH
cd /n/fs/pvl-progen/anlon/NeRF2Physics_InfinigenEval

# Input args
DATA_DIR=/n/fs/scratch/${USER}/nerf2physics/infinigen_nerf_data
TRAINING_ITERS=10000
SCENE_LIST=${DATA_DIR}/scene_list.txt

# Scene range for this task
# Each job processes 20 scenes
START_INDEX=$((SLURM_ARRAY_TASK_ID * 20))
END_INDEX=$((START_INDEX + 20))

# Load scene names into array
mapfile -t ALL_SCENES < "$SCENE_LIST"

# Bounds check
TOTAL_SCENES=${#ALL_SCENES[@]}
if [ $END_INDEX -gt $TOTAL_SCENES ]; then
    END_INDEX=$TOTAL_SCENES
fi

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} processing scenes ${START_INDEX} to $(($END_INDEX - 1))"

# Timing start
START_TIME=$(date +%s)

# Loop through assigned scenes
for ((i = START_INDEX; i < END_INDEX; i++)); do
    SCENE_NAME=${ALL_SCENES[$i]}
    echo "[$(date)] Starting reconstruction for scene ${SCENE_NAME} (index $i)"

    python ns_reconstruction.py \
        --data_dir ${DATA_DIR} \
        --training_iters ${TRAINING_ITERS} \
        --scene_name ${SCENE_NAME} \
        --num_points 50000 \
        --bbox_size 1.0

    echo "[$(date)] Finished reconstruction for scene ${SCENE_NAME}"
done

# Timing end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} completed in ${ELAPSED_TIME} seconds ($(($ELAPSED_TIME / 60)) minutes)"

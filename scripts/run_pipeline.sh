#!/bin/bash
# Script to run the entire NeRF2Physics pipeline as a sequence of sbatch jobs

# Create logs directory if it doesn't exist
mkdir -p /n/fs/scratch/az4244/nerf2physics/logs

# Submit first job - 3D Reconstruction
echo "Submitting 3D Reconstruction job..."
job1=$(sbatch reconstruct.sbatch | awk '{print $4}')
echo "Submitted reconstruction job: $job1"

# Submit second job - CLIP Feature Fusion (to run after first job completes)
echo "Submitting CLIP Feature Fusion job (will run after reconstruction)..."
job2=$(sbatch --dependency=afterok:$job1 fusion.sbatch | awk '{print $4}')
echo "Submitted feature fusion job: $job2"

# Submit third job - Captioning (to run after second job completes)
echo "Submitting Captioning job (will run after feature fusion)..."
job3=$(sbatch --dependency=afterok:$job2 captioning.sbatch | awk '{print $4}')
echo "Submitted captioning job: $job3"

# Submit fourth job - Material Proposal (to run after third job completes)
echo "Submitting Material Proposal job (will run after captioning)..."
job4=$(sbatch --dependency=afterok:$job3 material.sbatch | awk '{print $4}')
echo "Submitted material proposal job: $job4"

# Submit fifth job - Property Prediction (to run after fourth job completes)
echo "Submitting Property Prediction job (will run after material proposal)..."
job5=$(sbatch --dependency=afterok:$job4 predict.sbatch | awk '{print $4}')
echo "Submitted property prediction job: $job5"

echo "All jobs submitted! Pipeline will run automatically in sequence."
echo "You can check job status with: squeue -u az4244"

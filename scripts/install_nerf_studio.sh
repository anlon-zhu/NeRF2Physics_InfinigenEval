#!/bin/bash
#SBATCH --job-name=install_ns
#SBATCH --output=install_ns_%j.out
#SBATCH --error=install_ns_%j.err
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --time=30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Script for installing Nerfstudio on a compute cluster
echo "Starting Nerfstudio installation on $(hostname) at $(date)"

# Check user's quota and available disk space
echo "User quota information:"
quota -s || echo "Quota command not available"

echo "Disk space before installation:"
df -h $HOME

# Load necessary modules and environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"

# Install PyTorch with CUDA 12.x (to match system detection of 12.8)
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Don't install CUDA via conda as system already has CUDA 12.8
# Instead, just verify we can access it
echo "Verifying CUDA access:"

# Check if CUDA tools like nvcc are available
which nvcc || echo "nvcc not found in PATH"

# Install tiny-cuda-nn with the correct CUDA version
# Since we detected CUDA 12.8 on the system, we need compatible versions
echo "Installing tiny-cuda-nn..."
CUDA_HOME=/usr/local/cuda-12.8 pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch


# Install Nerfstudio with minimal dependencies
echo "Installing Nerfstudio..."
pip install nerfstudio

# Final installation test
echo "Testing final installation..."
python -c "
import torch
import numpy
print('PyTorch version:', torch.__version__)
print('NumPy version:', numpy.__version__)
print('CUDA available:', torch.cuda.is_available())
print('PyTorch CUDA version:', torch.version.cuda)
print('GPU device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))

try:
    import tinycudann
    print('tiny-cuda-nn version:', tinycudann.__version__)
    print('tiny-cuda-nn is working correctly')
except ImportError:
    print('tiny-cuda-nn is not installed. Some operations may be slower.')
except Exception as e:
    print('tiny-cuda-nn is installed but encountered an error:', str(e))
"

echo "Disk space after installation:"
df -h $HOME
echo "Installation completed at $(date)"

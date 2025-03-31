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

# Clean conda and pip caches first to free up space
echo "Cleaning caches to free up disk space..."
rm -rf ~/.cache/pip
mkdir -p ~/.cache/pip
rm -rf ~/.conda/pkgs/*
mkdir -p ~/.conda/pkgs

# Check available disk space
echo "Disk space before installation:"
df -h $HOME

# Load necessary modules and environment
module load anaconda3
export PATH=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy/bin:$PATH

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"

# Explicitly set CUDA paths to ensure version compatibility
export CUDA_HOME=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Uninstall conflicting packages
echo "Removing existing packages..."
pip uninstall -y torch torchvision functorch tinycudann numpy

# Install compatible NumPy version first (1.24 is compatible with both systems)
echo "Installing NumPy 1.24..."
pip install numpy==1.24.3

# Install PyTorch 2.1.2 with CUDA 11.8
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Print CUDA info
echo "CUDA information after PyTorch installation:"
python -c "import torch; print('PyTorch CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

# Install tiny-cuda-nn and dependencies with explicit CUDA version
echo "Installing tiny-cuda-nn dependencies..."
pip install ninja --no-cache-dir

# Setting environment variables for tiny-cuda-nn compilation
export TCNN_CUDA_ARCHITECTURES="60;70;75;80;86"
echo "Installing tiny-cuda-nn from source..."
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-cache-dir

# Install Nerfstudio with minimal dependencies
echo "Installing Nerfstudio..."
pip install nerfstudio --no-cache-dir

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

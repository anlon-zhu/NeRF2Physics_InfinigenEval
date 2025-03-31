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

# Find and set proper CUDA paths
echo "Looking for CUDA installation..."
module load cuda/11.8

# Try to locate CUDA installation
for possible_cuda_path in /usr/local/cuda-11.8 /usr/local/cuda /n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy /opt/conda
do
    if [ -d "$possible_cuda_path/bin" ] && [ -f "$possible_cuda_path/bin/nvcc" ]; then
        export CUDA_HOME="$possible_cuda_path"
        echo "Found CUDA at: $CUDA_HOME"
        break
    fi
done

# Add CUDA to path and library path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify NVCC is available
if command -v nvcc >/dev/null 2>&1; then
    echo "NVCC found: $(which nvcc)"
    echo "NVCC version: $(nvcc --version | head -n 1)"
else
    echo "WARNING: NVCC not found in PATH. Will attempt to install CUDA toolkit."
    
    # Try to install CUDA toolkit
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y --no-cache-dir
    
    # Check again for NVCC
    if command -v nvcc >/dev/null 2>&1; then
        echo "NVCC now available at: $(which nvcc)"
    else
        echo "ERROR: NVCC still not available after installation attempt."
        echo "Will try to continue, but tiny-cuda-nn may fail to install."
    fi
fi

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

# Try alternative approach using pre-built wheels first
echo "Attempting to install pre-built tiny-cuda-nn wheel..."
pip install --no-cache-dir tinycudann

# If that fails, try source installation with our detected CUDA path
if [ $? -ne 0 ]; then
    echo "Pre-built wheel installation failed, trying source installation..."
    
    # Create environment file to help tiny-cuda-nn detect CUDA properly
    echo "Creating custom environment for tiny-cuda-nn build..."
    export TCNN_CUDA_HOME="$CUDA_HOME"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    
    # Try pip install with explicit path to nvcc
    pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    
    # If still failing, try the colmap version which has fewer CUDA requirements
    if [ $? -ne 0 ]; then
        echo "Standard tiny-cuda-nn installation failed, trying nerfstudio's fork..."
        pip install --no-cache-dir git+https://github.com/nerfstudio-project/tiny-cuda-nn/#subdirectory=bindings/torch
    fi
fi

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

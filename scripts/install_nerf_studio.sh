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

# Check for permission issues
ENV_PATH=$(dirname $(which python))
echo "Environment path: $ENV_PATH"
echo "Checking write permissions..."
if [ ! -w "$ENV_PATH" ]; then
    echo "WARNING: No write permissions to $ENV_PATH"
    echo "Make sure you're using a conda environment you can write to!"
    exit 1
fi

# Set up environment
module load anaconda3
export PATH=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy/bin:$PATH

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"

# Clean cache to avoid disk space issues
echo "Cleaning pip cache..."
rm -rf ~/.cache/pip
mkdir -p ~/.cache/pip

# Check disk space
echo "Checking disk space..."
df -h ~/.cache

# Check for NumPy version and fix if needed
python -c "import numpy; print('Current NumPy version:', numpy.__version__)"
if python -c "import numpy; from packaging import version; exit(0 if version.parse(numpy.__version__) < version.parse('2.0.0') else 1)"; then
    echo "NumPy version is below 2.0, which is good for compatibility"
else
    echo "NumPy version is 2.0+, downgrading to 1.24.3 for better compatibility"
    pip uninstall numpy -y
    pip install numpy==1.24.3
fi

# Check existing CUDA version on the system
echo "Checking system CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "System CUDA version: $CUDA_VERSION"
    
    # Choose appropriate PyTorch version based on CUDA version
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        echo "Detected CUDA 12.x, installing PyTorch with CUDA 12.1 support"
        TORCH_CUDA="cu121"
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        echo "Detected CUDA 11.x, installing PyTorch with CUDA 11.8 support"
        TORCH_CUDA="cu118"
    else
        echo "Unknown CUDA version, defaulting to CUDA 11.8 support"
        TORCH_CUDA="cu118"
    fi
else
    echo "nvcc not found, defaulting to CUDA 11.8 support"
    TORCH_CUDA="cu118"
fi

# Clean previous installations
echo "Removing previous PyTorch installations..."
pip uninstall torch torchvision functorch tinycudann -y

# Install PyTorch with appropriate CUDA version
echo "Installing PyTorch with $TORCH_CUDA..."
if [ "$TORCH_CUDA" == "cu121" ]; then
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    
    # Only install cuda-toolkit if using CUDA 11.8
    echo "Installing CUDA toolkit using conda..."
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
fi

# Verify torch installation before proceeding
echo "Verifying PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Install dependencies and nerfstudio
echo "Installing ninja..."
pip install ninja

echo "Installing tiny-cuda-nn..."
echo "Setting TCNN_CUDA_ARCHITECTURES environment variable to support multiple architectures"
export TCNN_CUDA_ARCHITECTURES="6.0;6.1;7.0;7.5;8.0;8.6"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo "Installing nerfstudio..."
pip install nerfstudio

# Test installation
echo "Testing installation..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__); print('GPU device count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "Testing ns-train command..."
which ns-train
ns-train --help | head -10

echo "Installation completed at $(date)"
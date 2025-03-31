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

# Load necessary modules and environment
module load anaconda3
export PATH=/n/fs/pvl-progen/anlon/envs/nerf2phy/nerf2phy/bin:$PATH

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"

# Clean pip cache
clean_cache() {
    echo "Cleaning pip cache..."
    rm -rf ~/.cache/pip
    mkdir -p ~/.cache/pip
}

clean_cache

python -m pip install --upgrade pip
pip uninstall torch torchvision functorch tinycudann -y

# Install PyTorch 2.1.2 with CUDA 11.8
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install CUDA toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

# Install tiny-cuda-nn and dependencies
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install Nerfstudio
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

echo "Installation completed at $(date)"

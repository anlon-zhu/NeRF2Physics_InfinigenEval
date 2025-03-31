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

# Define helper functions to check package versions
check_numpy_version() {
    if python -c "import numpy; print(numpy.__version__)" 2>/dev/null | grep -q "1.24.3"; then
        echo "NumPy 1.24.3 is already installed"
        return 0
    else
        echo "Required NumPy version 1.24.3 is not installed"
        return 1
    fi
}

check_pytorch_version() {
    required_version="2.1.2"
    if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "$required_version"; then
        echo "PyTorch $required_version is already installed"
        # Check CUDA compatibility
        if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
            echo "CUDA is available and working with PyTorch"
            return 0
        else
            echo "PyTorch is installed but CUDA is not available"
            return 1
        fi
    else
        echo "Required PyTorch version $required_version is not installed"
        return 1
    fi
}

check_tinycudann() {
    if python -c "import tinycudann as tcnn; print(tcnn.__version__)" 2>/dev/null; then
        echo "tiny-cuda-nn is already installed and working"
        return 0
    else
        echo "tiny-cuda-nn is not installed or not working"
        return 1
    fi
}

check_nerfstudio() {
    if command -v ns-train >/dev/null 2>&1; then
        echo "Nerfstudio is already installed"
        return 0
    else
        echo "Nerfstudio is not installed"
        return 1
    fi
}

# Clean pip cache only if performing installations
clean_cache() {
    echo "Cleaning pip cache..."
    rm -rf ~/.cache/pip
    mkdir -p ~/.cache/pip
}

# NumPy installation
if ! check_numpy_version; then
    clean_cache
    echo "Installing NumPy 1.24.3..."
    pip uninstall numpy -y || true
    pip install numpy==1.24.3
    python -c "import numpy; print('NumPy version:', numpy.__version__)"
else
    echo "Skipping NumPy installation"
fi

# Detect CUDA version for PyTorch compatibility
if ! check_pytorch_version; then
    clean_cache
    echo "Checking system CUDA version..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo "System CUDA version: $CUDA_VERSION"
        
        # Choose appropriate PyTorch version based on CUDA version
        if [[ "$CUDA_VERSION" == 12.* ]]; then
            echo "Detected CUDA 12.x, installing PyTorch with CUDA 12.1 support"
            TORCH_CUDA="cu121"
        else
            echo "Using CUDA 11.8 support"
            TORCH_CUDA="cu118"
        fi
    else
        echo "nvcc not found, defaulting to CUDA 11.8 support"
        TORCH_CUDA="cu118"
    fi

    # Install PyTorch with appropriate CUDA version
    echo "Installing PyTorch with $TORCH_CUDA..."
    pip uninstall torch torchvision -y || true
    pip install torch==2.1.2+$TORCH_CUDA torchvision==0.16.2+$TORCH_CUDA --extra-index-url https://download.pytorch.org/whl/$TORCH_CUDA
    
    # Install CUDA toolkit if using CUDA 11.8
    if [[ "$TORCH_CUDA" == "cu118" ]]; then
        echo "Installing CUDA toolkit using conda..."
        conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
    fi

    # Verify torch installation
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
else
    echo "Skipping PyTorch installation"
fi

# Install dependencies for tiny-cuda-nn
if ! check_tinycudann; then
    clean_cache
    echo "Installing prerequisites for tiny-cuda-nn..."
    pip install ninja cmake setuptools wheel
    
    echo "Installing tiny-cuda-nn..."
    # Try using pip install approach first
    pip uninstall -y tinycudann tiny-cuda-nn || true
    
    # Detect CUDA version for tiny-cuda-nn compatibility
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        echo "Detected CUDA version for tiny-cuda-nn: $CUDA_MAJOR.$CUDA_MINOR"

        # Try to find a matching wheel based on CUDA version
        if [ "$CUDA_MAJOR" = "11" ] && [ "$CUDA_MINOR" -ge "7" ]; then
            echo "Installing tiny-cuda-nn for CUDA 11.7+"
            pip install "git+https://github.com/NerfStudio-official/tinycudann-wheels.git@master#subdirectory=cuda117"
        elif [ "$CUDA_MAJOR" = "11" ] && [ "$CUDA_MINOR" -ge "3" ]; then
            echo "Installing tiny-cuda-nn for CUDA 11.3+"
            pip install "git+https://github.com/NerfStudio-official/tinycudann-wheels.git@master#subdirectory=cuda113"
        elif [ "$CUDA_MAJOR" = "12" ]; then
            echo "Installing tiny-cuda-nn for CUDA 12.x"
            pip install "git+https://github.com/NerfStudio-official/tinycudann-wheels.git@master#subdirectory=cuda118"
        else
            # Fall back to source installation
            echo "No pre-built wheel available, attempting source installation"
            export TCNN_CUDA_ARCHITECTURES="60,61,70,75,80,86"
            pip install git+https://github.com/nvlabs/tiny-cuda-nn/#subdirectory=bindings/torch || {
                echo "Pip install failed, falling back to manual compilation..."
                TMP_DIR=$(mktemp -d)
                git clone https://github.com/NVlabs/tiny-cuda-nn.git $TMP_DIR/tiny-cuda-nn
                cd $TMP_DIR/tiny-cuda-nn/bindings/torch
                python setup.py install
                cd - > /dev/null
            }
        fi
    else
        # If nvcc is not available, try source installation
        echo "NVCC not found, attempting source installation"
        export TCNN_CUDA_ARCHITECTURES="60,61,70,75,80,86"
        pip install git+https://github.com/nvlabs/tiny-cuda-nn/#subdirectory=bindings/torch || {
            echo "Pip install failed, falling back to manual compilation..."
            TMP_DIR=$(mktemp -d)
            git clone https://github.com/NVlabs/tiny-cuda-nn.git $TMP_DIR/tiny-cuda-nn
            cd $TMP_DIR/tiny-cuda-nn/bindings/torch
            python setup.py install
            cd - > /dev/null
        }
    fi
    
    # Verify the installation
    python -c "import tinycudann as tcnn; print('tiny-cuda-nn version:', tcnn.__version__)" || 
        echo "Warning: tiny-cuda-nn installation failed but continuing with nerfstudio installation"
else
    echo "Skipping tiny-cuda-nn installation"
fi

# Install Nerfstudio if not already installed
if ! check_nerfstudio; then
    clean_cache
    echo "Installing nerfstudio..."
    pip install nerfstudio
    
    # Test if installation was successful
    echo "Testing ns-train command..."
    which ns-train
    ns-train --help | head -10
else
    NERFSTUDIO_VERSION=$(pip show nerfstudio | grep Version | awk '{print $2}')
    echo "Nerfstudio is already installed, version: $NERFSTUDIO_VERSION"
fi

# Final test of the full installation
echo "Testing final installation..."
python -c "
import torch
import numpy
import tinycudann
print('PyTorch version:', torch.__version__)
print('NumPy version:', numpy.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"

echo "Installation completed at $(date)"

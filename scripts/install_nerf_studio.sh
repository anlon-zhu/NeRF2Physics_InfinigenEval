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
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
        CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
        echo "System CUDA version: $CUDA_MAJOR.$CUDA_MINOR"
        
        # Choose appropriate PyTorch version based on CUDA version
        if [[ "$CUDA_MAJOR" == "12" ]]; then
            if [[ "$CUDA_MINOR" -ge "1" ]]; then
                echo "Detected CUDA 12.x, installing PyTorch with CUDA 12.1 support"
                TORCH_CUDA="cu121"
            else
                echo "Detected CUDA 12.0, falling back to CUDA 11.8 for PyTorch"
                TORCH_CUDA="cu118"
            fi
        elif [[ "$CUDA_MAJOR" == "11" ]]; then
            if [[ "$CUDA_MINOR" -ge "8" ]]; then
                echo "Detected CUDA 11.8+, installing PyTorch with CUDA 11.8 support"
                TORCH_CUDA="cu118"
            elif [[ "$CUDA_MINOR" -ge "7" ]]; then
                echo "Detected CUDA 11.7, installing PyTorch with CUDA 11.7 support"
                TORCH_CUDA="cu117"
            elif [[ "$CUDA_MINOR" -ge "3" ]]; then
                echo "Detected CUDA 11.3-11.6, installing PyTorch with CUDA 11.3 support"
                TORCH_CUDA="cu113"
            else
                echo "Detected CUDA 11.0-11.2, falling back to CUDA 11.3 for PyTorch"
                TORCH_CUDA="cu113"
            fi
        else
            echo "CUDA version $CUDA_MAJOR.$CUDA_MINOR not directly supported, using CUDA 11.8 for PyTorch"
            TORCH_CUDA="cu118"
        fi
    else
        echo "nvcc not found, defaulting to CUDA 11.8 support"
        TORCH_CUDA="cu118"
        CUDA_MAJOR="11"
        CUDA_MINOR="8"
        CUDA_VERSION="11.8"
    fi
    
    # Save CUDA version for later use with tiny-cuda-nn
    SYSTEM_CUDA_VERSION="$CUDA_VERSION"
    SYSTEM_CUDA_MAJOR="$CUDA_MAJOR"
    SYSTEM_CUDA_MINOR="$CUDA_MINOR"

    # Install PyTorch with appropriate CUDA version
    echo "Installing PyTorch with $TORCH_CUDA..."
    pip uninstall torch torchvision -y || true
    
    # Use appropriate PyTorch version based on detected CUDA
    if [[ "$TORCH_CUDA" == "cu121" ]]; then
        echo "Installing PyTorch with CUDA 12.1 support"
        pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
    elif [[ "$TORCH_CUDA" == "cu118" ]]; then
        echo "Installing PyTorch with CUDA 11.8 support"
        pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
        
        # Install CUDA 11.8 toolkit for better compatibility
        echo "Installing CUDA 11.8 toolkit using conda..."
        conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
    elif [[ "$TORCH_CUDA" == "cu117" ]]; then
        echo "Installing PyTorch with CUDA 11.7 support"
        pip install torch==2.1.2+cu117 torchvision==0.16.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    elif [[ "$TORCH_CUDA" == "cu113" ]]; then
        echo "Installing PyTorch with CUDA 11.3 support"
        pip install torch==2.1.2+cu113 torchvision==0.16.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    else
        echo "Unknown CUDA version, falling back to CUDA 11.8"
        pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
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
    pip uninstall -y tinycudann tiny-cuda-nn || true
    
    # Get CUDA version from PyTorch (preferred) or fallback to system version
    PYTORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    if [ -n "$PYTORCH_CUDA_VERSION" ]; then
        echo "Using PyTorch's CUDA version: $PYTORCH_CUDA_VERSION"
        CUDA_VERSION="$PYTORCH_CUDA_VERSION"
    else
        echo "PyTorch CUDA version not detected, using system CUDA version"
        CUDA_VERSION="$SYSTEM_CUDA_VERSION"
    fi
    
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    echo "Using CUDA version for tiny-cuda-nn: $CUDA_MAJOR.$CUDA_MINOR"

    # Try to install pre-built wheel first
    if [ "$CUDA_MAJOR" = "12" ]; then
        echo "Installing tiny-cuda-nn for CUDA 12.x"
        pip install "git+https://github.com/NerfStudio-official/tinycudann-wheels.git@master#subdirectory=cuda118"
    elif [ "$CUDA_MAJOR" = "11" ]; then
        if [ "$CUDA_MINOR" -ge "7" ]; then
            echo "Installing tiny-cuda-nn for CUDA 11.7+"
            pip install "git+https://github.com/NerfStudio-official/tinycudann-wheels.git@master#subdirectory=cuda117"
        elif [ "$CUDA_MINOR" -ge "3" ]; then
            echo "Installing tiny-cuda-nn for CUDA 11.3+"
            pip install "git+https://github.com/NerfStudio-official/tinycudann-wheels.git@master#subdirectory=cuda113"
        fi
    fi
    
    # Check if installation worked, if not try source installation
    if ! python -c "import tinycudann as tcnn" &>/dev/null; then
        echo "Pre-built wheel installation failed, attempting source installation"
        
        # Setup compile environment
        export TCNN_CUDA_ARCHITECTURES="60,61,70,75,80,86"
        
        # Use gcc-11 if available
        if command -v gcc-11 &> /dev/null && command -v g++-11 &> /dev/null; then
            echo "Using GCC/G++ version 11 for compilation"
            export CC=gcc-11
            export CXX=g++-11
        fi
        
        # Try to set CUDA_HOME
        if [ -d "/usr/local/cuda-$CUDA_MAJOR.$CUDA_MINOR" ]; then
            export CUDA_HOME="/usr/local/cuda-$CUDA_MAJOR.$CUDA_MINOR"
        fi
        
        # Attempt installation
        if command -v nvcc &> /dev/null; then
            echo "Attempting pip install with nvcc..."
            CUDACXX=$(which nvcc) pip install git+https://github.com/nvlabs/tiny-cuda-nn/#subdirectory=bindings/torch || {
                echo "Direct installation failed, attempting manual compilation..."
                TMP_DIR=$(mktemp -d)
                git clone --depth=1 https://github.com/NVlabs/tiny-cuda-nn.git $TMP_DIR/tiny-cuda-nn
                cd $TMP_DIR/tiny-cuda-nn/bindings/torch
                python setup.py install
                cd - > /dev/null
            }
        else
            echo "NVCC not found, tiny-cuda-nn installation may not be possible"
            echo "Installing nerfstudio without tiny-cuda-nn (some features may be slower)"
        fi
    fi
    
    # Verify installation
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
print('PyTorch version:', torch.__version__)
print('NumPy version:', numpy.__version__)
print('CUDA available:', torch.cuda.is_available())
print('PyTorch CUDA version:', torch.version.cuda)
print('GPU device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))

# Try to import tinycudann, but don't fail if not available
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

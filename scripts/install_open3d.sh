#!/bin/bash
# Installation script for Open3D with headless rendering support
# Based on Open3D documentation for headless rendering

set -e  # Exit on error

echo "Installing Open3D with headless rendering support..."

# Load anaconda module and set up conda environment
module load anaconda3
export PATH=/n/fs/vl/anlon/envs/nerf2phy/bin:$PATH

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_PREFIX"

# Define installation directories (using filesystem with more space)
OSMESA_DIR="/n/fs/vl/anlon/osmesa"
BUILD_DIR="/n/fs/vl/anlon/build_tmp"
mkdir -p "$OSMESA_DIR" "$BUILD_DIR"

# Function to check if we're on a SLURM node
is_slurm_node() {
    command -v srun &> /dev/null
}

# Check if we already have a local OSMesa installation
if [ -d "$OSMESA_DIR/lib" ] && [ -f "$OSMESA_DIR/lib/libOSMesa.so" ]; then
    echo "Found existing OSMesa installation at $OSMESA_DIR"
else
    echo "OSMesa not found in $OSMESA_DIR, installing from source..."
    
    # Check for LLVM - needed for Mesa compilation
    if ! command -v llvm-config &> /dev/null && ! command -v llvm-config-8 &> /dev/null; then
        echo "Need LLVM for compiling Mesa. Check if your cluster has LLVM modules"
        echo "You might need to run: module load llvm"
        
        # If on a cluster, try loading LLVM module
        if is_slurm_node; then
            echo "Attempting to load LLVM module..."
            module load llvm 2>/dev/null || module load llvm/8 2>/dev/null || true
        fi
        
        if ! command -v llvm-config &> /dev/null && ! command -v llvm-config-8 &> /dev/null; then
            echo "LLVM not found and cannot be loaded. Please contact your system administrator."
            echo "You may need to modify this script to use available LLVM versions."
            exit 1
        fi
    fi
    
    cd "$BUILD_DIR"
    
    # Download Mesa
    echo "Downloading Mesa 19.0.8..."
    curl -O https://mesa.freedesktop.org/archive/mesa-19.0.8.tar.xz
    tar xf mesa-19.0.8.tar.xz
    cd mesa-19.0.8
    
    # Determine LLVM config path
    LLVM_CONFIG="llvm-config"
    if command -v llvm-config-8 &> /dev/null; then
        LLVM_CONFIG="llvm-config-8"
    fi
    
    echo "Using LLVM config: $LLVM_CONFIG"
    
    # Configure and build Mesa with OSMesa Gallium driver
    echo "Configuring Mesa..."
    LLVM_CONFIG="$LLVM_CONFIG" ./configure --prefix="$OSMESA_DIR" \
        --disable-osmesa --disable-driglx-direct --disable-gbm --enable-dri \
        --with-gallium-drivers=swrast --enable-autotools --enable-llvm --enable-gallium-osmesa
    
    echo "Compiling Mesa (this may take a while)..."
    make -j$(nproc)
    
    echo "Installing Mesa to $OSMESA_DIR..."
    make install
    
    echo "OSMesa installation complete."
fi

# Set environment variables for headless rendering and point to our local OSMesa
export OPEN3D_HEADLESS_RENDERING=ON
export LD_LIBRARY_PATH="$OSMESA_DIR/lib:$LD_LIBRARY_PATH"
export CMAKE_ARGS="-DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF -DOSMESA_INCLUDE_DIR=$OSMESA_DIR/include -DOSMESA_LIBRARY=$OSMESA_DIR/lib/libOSMesa.so"

# Install Open3D from source
echo "Installing Open3D from source with headless rendering enabled..."
pip install --upgrade pip
pip install open3d --no-binary open3d

# Verify installation
echo "Verifying Open3D installation..."
python -c "import open3d as o3d; print(f'Open3D version: {o3d.__version__}')"

# Create a test script to verify headless rendering
TESTDIR="$(mktemp -d)"
TEST_SCRIPT="${TESTDIR}/test_headless.py"

cat > "$TEST_SCRIPT" << 'EOF'
import open3d as o3d
import numpy as np

# Create a simple point cloud
pcd = o3d.geometry.PointCloud()
points = np.random.rand(100, 3)
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([1, 0, 0])  # Red

# Try headless rendering
vis = o3d.visualization.Visualizer()
try:
    vis.create_window(width=640, height=480, visible=False, renderer='headless')
    vis.add_geometry(pcd)
    
    # Capture an image
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    
    print("Headless rendering test: SUCCESS")
    vis.destroy_window()
except Exception as e:
    print(f"Headless rendering test: FAILED - {e}")
EOF

echo "Running headless rendering test..."
python "$TEST_SCRIPT"
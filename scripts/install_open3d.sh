#!/bin/bash
# Installation script for Open3D with headless rendering support
# Based on Open3D documentation for headless rendering

set -e  # Exit on error

echo "Installing Open3D with headless rendering support..."

# Function to check if we're on a SLURM node
is_slurm_node() {
    command -v srun &> /dev/null
}

# Check if OSMesa is installed
if ! dpkg -s libosmesa6-dev &> /dev/null; then
    echo "OSMesa is required for headless rendering."
    echo "Installing OSMesa..."
    
    if is_slurm_node; then
        echo "Detected SLURM environment. You'll need to ask your system administrator to install OSMesa."
        echo "Please install libosmesa6-dev system-wide or follow the instructions to build from source."
        echo "See: https://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html"
        exit 1
    else
        # Try to install on Ubuntu-like systems
        sudo apt-get update
        sudo apt-get install -y libosmesa6-dev
    fi
fi

# Set environment variables for headless rendering
export OPEN3D_HEADLESS_RENDERING=ON
export CMAKE_ARGS="-DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF"

# Install Open3D from source
echo "Installing Open3D from source with headless rendering enabled..."
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

echo "Setup complete!"

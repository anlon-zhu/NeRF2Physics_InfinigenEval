import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation

# Add the parent directory to the path so we can import the InfinigenDataset class
sys.path.append(str(Path(__file__).parent.parent))
from infinigen_to_colmap import InfinigenDataset

def read_colmap_images_file(file_path):
    """Read camera positions from a COLMAP images.txt file
    
    Args:
        file_path: Path to the images.txt file
        
    Returns:
        Tuple of (positions, orientations) as numpy arrays
    """
    positions = []
    orientations = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Parse camera data line
            parts = line.split()
            if len(parts) >= 9:
                # Extract quaternion (QW, QX, QY, QZ) and translation (TX, TY, TZ)
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                
                # Convert quaternion to rotation matrix
                r = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses x, y, z, w order
                rotation_matrix = r.as_matrix()
                
                positions.append([tx, ty, tz])
                orientations.append(rotation_matrix)
            
            # Skip the next line (POINTS2D)
            i += 2
    
    return np.array(positions), orientations

def plot_camera_positions(dataset, title=None, show_axes=True, show_paths=True, colmap_images_file=None, colmap_only=False):
    """Plot camera positions from an InfinigenDataset
    
    Args:
        dataset: InfinigenDataset instance
        title: Optional title for the plot
        show_axes: Whether to show coordinate axes
        show_paths: Whether to show camera paths (connecting sequential cameras)
    """
    # Get camera positions from the dataset
    camera_positions = []
    camera_orientations = []
    
    # Get common view IDs (views that have both camera and image data)
    view_ids = sorted(dataset.camera_data.keys())
    
    for view_id in view_ids:
        # Get camera transformation matrix
        T = dataset.camera_data[view_id]['T']
        
        # Extract position (translation vector)
        position = T[:3, 3]
        
        # Extract orientation (rotation matrix)
        rotation = T[:3, :3]
        
        camera_positions.append(position)
        camera_orientations.append(rotation)
    
    # Convert to numpy arrays
    camera_positions = np.array(camera_positions)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Infinigen camera positions (unless colmap_only is True)
    if not colmap_only:
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                  c='blue', marker='o', s=50, label='Infinigen Camera Positions')
    
    # Plot COLMAP camera positions if provided
    if colmap_images_file is not None and os.path.exists(colmap_images_file):
        try:
            colmap_positions, colmap_orientations = read_colmap_images_file(colmap_images_file)
            
            # Plot COLMAP camera positions
            ax.scatter(colmap_positions[:, 0], colmap_positions[:, 1], colmap_positions[:, 2],
                      c='red', marker='^', s=50, label='COLMAP Camera Positions')
            
            # Plot COLMAP camera paths if requested
            if show_paths and len(colmap_positions) > 1:
                ax.plot(colmap_positions[:, 0], colmap_positions[:, 1], colmap_positions[:, 2],
                       'g-', alpha=0.7, label='COLMAP Camera Path')
                
            # Draw COLMAP coordinate axes if requested
            if show_axes:
                axis_scale = 0.2
                for i, (pos, rot) in enumerate(zip(colmap_positions, colmap_orientations)):
                    # Camera coordinate axes
                    x_axis = rot[:, 0] * axis_scale
                    y_axis = rot[:, 1] * axis_scale
                    z_axis = rot[:, 2] * axis_scale
                    
                    # Draw the axes (only add to legend for the first camera)
                    ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2],
                             color='darkred', alpha=0.8, label='COLMAP X axis' if i == 0 else None)
                    ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2],
                             color='darkgreen', alpha=0.8, label='COLMAP Y axis' if i == 0 else None)
                    ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2],
                             color='darkblue', alpha=0.8, label='COLMAP Z axis' if i == 0 else None)
        except Exception as e:
            print(f"Error reading COLMAP images file: {e}")
    
    # Plot camera paths if requested (unless colmap_only is True)
    if show_paths and len(camera_positions) > 1 and not colmap_only:
        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                'r-', alpha=0.7, label='Infinigen Camera Path')
    
    # Draw coordinate axes at each camera position (unless colmap_only is True)
    if show_axes and not colmap_only:
        # Scale for the axes
        axis_scale = 0.2
        
        for i, (pos, rot) in enumerate(zip(camera_positions, camera_orientations)):
            # Camera coordinate axes
            x_axis = rot[:, 0] * axis_scale
            y_axis = rot[:, 1] * axis_scale
            z_axis = rot[:, 2] * axis_scale
            
            # Draw the axes
            ax.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], 
                      color='r', alpha=0.8, label='Infinigen X axis' if i == 0 else None)
            ax.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], 
                      color='g', alpha=0.8, label='Infinigen Y axis' if i == 0 else None)
            ax.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], 
                      color='b', alpha=0.8, label='Infinigen Z axis' if i == 0 else None)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set appropriate title
    if title:
        ax.set_title(title)
    elif colmap_only and colmap_images_file is not None:
        ax.set_title(f'COLMAP Camera Positions for {dataset.scene_name}')
    else:
        ax.set_title(f'Camera Positions for {dataset.scene_name}')
    
    # Add legend
    ax.legend()
    
    # Make the plot more visually appealing
    ax.grid(True)
    
    # Determine the points to consider for setting the axis limits
    all_positions = [] if colmap_only else [camera_positions]
    if colmap_images_file is not None and os.path.exists(colmap_images_file):
        try:
            colmap_positions, _ = read_colmap_images_file(colmap_images_file)
            all_positions.append(colmap_positions)
        except Exception:
            pass
    
    # Combine all positions for determining plot limits
    combined_positions = np.vstack(all_positions) if len(all_positions) > 1 else camera_positions
    
    # Set equal aspect ratio
    max_range = np.max([
        np.max(combined_positions[:, 0]) - np.min(combined_positions[:, 0]),
        np.max(combined_positions[:, 1]) - np.min(combined_positions[:, 1]),
        np.max(combined_positions[:, 2]) - np.min(combined_positions[:, 2])
    ])
    
    mid_x = (np.max(combined_positions[:, 0]) + np.min(combined_positions[:, 0])) / 2
    mid_y = (np.max(combined_positions[:, 1]) + np.min(combined_positions[:, 1])) / 2
    mid_z = (np.max(combined_positions[:, 2]) + np.min(combined_positions[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot camera positions from Infinigen dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Infinigen scene directory")
    parser.add_argument("--scene_id", type=str, default=None, help="Scene ID to process (default: use directory name)")
    parser.add_argument("--no_axes", action="store_true", help="Don't show coordinate axes")
    parser.add_argument("--no_paths", action="store_true", help="Don't show camera paths")
    parser.add_argument("--colmap_dir", type=str, help="Path to COLMAP output directory (to compare with converted camera positions)")
    parser.add_argument("--colmap_only", action="store_true", help="Only plot COLMAP cameras (requires --colmap_dir)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Determine if input_dir is a scene directory or contains multiple scenes
    if args.scene_id is not None:
        # User specified a scene_id, look for it in input_dir
        print(f"Looking for scene {args.scene_id} in {input_dir}")
        scene_dir = input_dir / args.scene_id
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory {args.scene_id} not found in {input_dir}")
        
        # Load the dataset and plot
        dataset = InfinigenDataset(scene_dir)
        
        # Check for COLMAP images.txt file
        colmap_images_file = None
        if args.colmap_dir:
            colmap_dir = Path(args.colmap_dir)
            if args.scene_id:
                colmap_scene_dir = colmap_dir / args.scene_id
            else:
                colmap_scene_dir = colmap_dir
            
            colmap_images_file = colmap_scene_dir / 'sparse' / '0' / 'images.txt'
            if not colmap_images_file.exists():
                print(f"Warning: COLMAP images file not found at {colmap_images_file}")
                colmap_images_file = None
        
        # Validate colmap_only option
        if args.colmap_only and colmap_images_file is None:
            print("Warning: --colmap_only specified but no COLMAP images file found. Showing Infinigen cameras.")
            colmap_only = False
        else:
            colmap_only = args.colmap_only
            
        plot_camera_positions(dataset, show_axes=not args.no_axes, show_paths=not args.no_paths,
                             colmap_images_file=colmap_images_file, colmap_only=colmap_only)
        
    elif (input_dir / "frames").exists():
        # Input directory is already a scene directory
        print(f"Input directory {input_dir} appears to be a scene directory (contains frames/)")
        
        # Load the dataset and plot
        dataset = InfinigenDataset(input_dir)
        
        # Check for COLMAP images.txt file
        colmap_images_file = None
        if args.colmap_dir:
            colmap_dir = Path(args.colmap_dir)
            colmap_images_file = colmap_dir / 'sparse' / '0' / 'images.txt'
            if not colmap_images_file.exists():
                print(f"Warning: COLMAP images file not found at {colmap_images_file}")
                colmap_images_file = None
        
        # Validate colmap_only option
        if args.colmap_only and colmap_images_file is None:
            print("Warning: --colmap_only specified but no COLMAP images file found. Showing Infinigen cameras.")
            colmap_only = False
        else:
            colmap_only = args.colmap_only
            
        plot_camera_positions(dataset, show_axes=not args.no_axes, show_paths=not args.no_paths,
                             colmap_images_file=colmap_images_file, colmap_only=colmap_only)
        
    else:
        # Input directory might contain multiple scenes
        # Check for subdirectories that could be scene directories
        scene_dirs = [d for d in input_dir.iterdir() if d.is_dir() and (d / "frames").exists()]
        
        if not scene_dirs:
            raise FileNotFoundError(f"No scene directories found in {input_dir}")
        
        print(f"Found {len(scene_dirs)} scene directories in {input_dir}")
        
        # Plot each scene
        for scene_dir in scene_dirs:
            print(f"Processing scene {scene_dir.name}")
            dataset = InfinigenDataset(scene_dir)
            
            # Check for COLMAP images.txt file
            colmap_images_file = None
            if args.colmap_dir:
                colmap_dir = Path(args.colmap_dir)
                colmap_scene_dir = colmap_dir / scene_dir.name
                colmap_images_file = colmap_scene_dir / 'sparse' / '0' / 'images.txt'
                if not colmap_images_file.exists():
                    print(f"Warning: COLMAP images file not found at {colmap_images_file}")
                    colmap_images_file = None
            
            # Validate colmap_only option
            if args.colmap_only and colmap_images_file is None:
                print(f"Warning: --colmap_only specified but no COLMAP images file found for {scene_dir.name}. Showing Infinigen cameras.")
                colmap_only = False
            else:
                colmap_only = args.colmap_only
                
            plot_camera_positions(dataset, title=f"Camera Positions for {scene_dir.name}", 
                                 show_axes=not args.no_axes, show_paths=not args.no_paths,
                                 colmap_images_file=colmap_images_file, colmap_only=colmap_only)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import numpy as np
import json
import shutil
from pathlib import Path
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Infinigen camera data to COLMAP format")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing Infinigen renders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for COLMAP data")
    parser.add_argument("--scene_id", type=str, default=None, help="Scene ID to process (default: use directory name)")
    parser.add_argument("--use_depth", action="store_true", help="Enable depth supervision using depth images")
    return parser.parse_args()

def extract_view_id(filename):
    # Extract view ID from filenames like camview_10_0_0048_0.npz or Image_10_0_0048_0.png
    match = re.search(r'(?:camview|Image|Depth)_(\d+)_0_0048_0', filename)
    if match:
        return int(match.group(1))
    return None

def create_colmap_cameras_file(output_path, K, hw):
    """Create COLMAP cameras.txt file"""
    height, width = hw
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        # Using PINHOLE model with fx, fy, cx, cy parameters
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
    
    print(f"Created cameras.txt at {output_path}")

def create_colmap_images_file(output_path, image_data):
    """Create COLMAP images.txt file"""
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_data)}\n")
        
        for idx, (image_name, T) in enumerate(image_data, 1):
            # Extract rotation matrix and translation vector from T matrix
            R = T[:3, :3]
            t = T[:3, 3]
            
            # COLMAP uses quaternion for rotation
            # Convert rotation matrix to quaternion
            # This is a basic implementation - more robust methods exist
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            
            if trace > 0:
                S = np.sqrt(trace + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S
            
            # Convert to COLMAP convention where w component comes first
            quaternion = np.array([qw, qx, qy, qz])
            quaternion = quaternion / np.linalg.norm(quaternion)  # Normalize quaternion
            
            # Write image data
            f.write(f"{idx} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]} {t[0]} {t[1]} {t[2]} 1 {image_name}\n")
            # Empty line for POINTS2D (not needed for NeRF)
            f.write("\n")
    
    print(f"Created images.txt at {output_path}")
    
def create_empty_points3d_file(output_path):
    """Create empty points3D.txt file (not used by NeRF but required by COLMAP format)"""
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    
    print(f"Created empty points3D.txt at {output_path}")

def create_transforms_json(output_path, image_data, K, hw, depth_files=None):
    """Create transforms.json file required by Nerfstudio
    
    Args:
        output_path: Path to write transforms.json file
        image_data: List of tuples (image_name, T) with transform matrices
        K: Camera intrinsics matrix
        hw: Tuple of (height, width)
        depth_files: Optional dictionary mapping image names to corresponding depth file paths
    """
    height, width = hw
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    frames = []
    for idx, (image_name, T) in enumerate(image_data):
        # Extract rotation matrix and translation vector from T matrix
        c2w = np.eye(4)
        c2w[:3, :3] = T[:3, :3]
        c2w[:3, 3] = T[:3, 3]
        
        frame = {
            "file_path": f"./images/{image_name}",
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": idx + 1  # 1-based IDs for COLMAP
        }
        
        # Add depth file path if available
        if depth_files and image_name in depth_files:
            frame["depth_file_path"] = depth_files[image_name]
        
        frames.append(frame)
    
    transforms_json = {
        "camera_model": "PINHOLE",
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(width),
        "h": int(height),
        "frames": frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(transforms_json, f, indent=2)
    
    print(f"Created transforms.json at {output_path}")

def organize_for_nerfstudio(input_dir, output_dir, scene_id=None, use_depth=False):
    """Convert Infinigen data to COLMAP format for NeRF Studio
    
    This function assumes data is already organized by organize_data.sh script with structure:
    /path/to/data/scenes/scene_id/
      infinigen_images/      # Original Infinigen images
        camera_0/
          *.png
      camview/
        camera_0/
          *.npz
      images/               # New directory for COLMAP-formatted images
    """
    input_dir = Path(input_dir)
    print(f"Input directory: {input_dir}")
    
    # Auto-detect directory structure
    # If scene_id is not provided, use the input directory name
    if scene_id is None:
        scene_id = input_dir.name
        print(f"Using input directory name as scene_id: {scene_id}")
    
    # Check if input_dir is already the scene directory
    # Look for camview and infinigen_images directories
    if (input_dir / "camview").exists() and (input_dir / "infinigen_images").exists():
        # This is already the scene directory
        scene_dir = input_dir
        print(f"Input directory is already a scene directory containing camview/ and infinigen_images/")
    else:
        print(f"Input directory is not a scene directory, looking for scene {scene_id}")
        # Otherwise, it might be a parent directory
        if (input_dir / scene_id).exists():
            # Scene directory is directly in input_dir
            scene_dir = input_dir / scene_id
            print(f"Found scene directory at {scene_dir}")
        else:
            # Not found, error
            raise FileNotFoundError(f"Scene directory {scene_id} not found in {input_dir}")
    
    # Make sure output directory exists
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create COLMAP directory structure in the scene directory
    colmap_dir = scene_dir / "sparse" / "0"
    os.makedirs(colmap_dir, exist_ok=True)
    
    # Expected directories based on organize_data.sh structure
    infinigen_images_dir = scene_dir / "infinigen_images" / "camera_0"
    camview_dir = scene_dir / "camview" / "camera_0"
    depth_images_dir = scene_dir / "depth_images" / "camera_0" if use_depth else None
    
    # Create clean images directory for COLMAP sequential images
    images_dir = scene_dir / "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Create depth directory if using depth supervision
    depth_dir = None
    if use_depth and depth_images_dir and depth_images_dir.exists():
        depth_dir = scene_dir / "depth"
        os.makedirs(depth_dir, exist_ok=True)
        print(f"Using depth supervision, depth images directory: {depth_images_dir}")
    
    if not camview_dir.exists() or not infinigen_images_dir.exists():
        raise FileNotFoundError(f"Missing required directories: \nInfinigen Images: {infinigen_images_dir} (exists: {infinigen_images_dir.exists()})\nCamera: {camview_dir} (exists: {camview_dir.exists()})")
    
    # Get a list of all camera files
    cam_files = sorted([f for f in os.listdir(camview_dir) if f.endswith(".npz")])
    if not cam_files:
        raise FileNotFoundError(f"No camera files found in {camview_dir}")
    
    # Get the first camera file to extract intrinsics
    first_cam = np.load(camview_dir / cam_files[0])
    K = first_cam['K']
    hw = first_cam['HW']
    
    # Create COLMAP cameras.txt
    create_colmap_cameras_file(colmap_dir / "cameras.txt", K, hw)
    
    # Check if infinigen images exist (should already be copied by organize_data.sh)
    try:
        image_files = sorted([f for f in os.listdir(infinigen_images_dir) if f.endswith('.png') or f.endswith('.exr')])
        if not image_files:
            raise FileNotFoundError(f"No image files found in {infinigen_images_dir}")
        
        print(f"Found {len(image_files)} images in {infinigen_images_dir}")
    except Exception as e:
        print(f"Error accessing infinigen image directory: {e}")
        print(f"Directory structure: {scene_dir} contains {os.listdir(scene_dir)}")
        if (scene_dir / "infinigen_images").exists():
            print(f"Infinigen images directory exists: {scene_dir / 'infinigen_images'} contains {os.listdir(scene_dir / 'infinigen_images')}")
        raise
    
    # First collect all valid images and their transforms
    view_id_to_image = {}
    view_id_to_transform = {}
    
    for cam_file in cam_files:
        # Load camera data
        cam_data = np.load(camview_dir / cam_file)
        T = cam_data['T']
        
        # Extract view number
        view_id = extract_view_id(cam_file)
        if view_id is None:
            print(f"Warning: Could not extract view ID from camera file {cam_file}, skipping")
            continue
        
        # Look for matching image file in the infinigen_images directory
        matching_images = [img for img in image_files if img.startswith(f"{view_id}_") or img.startswith(f"{view_id:03d}_")]
        
        if not matching_images:
            # Try alternative patterns that might be in the organized data
            matching_images = [img for img in image_files if f"_{view_id}_" in img or f"_{view_id:03d}_" in img]
        
        if not matching_images:
            print(f"Warning: No matching image found for camera {cam_file} (view ID: {view_id})")
            continue
            
        # Use the first matching image
        img_file = matching_images[0]
        
        # Store the mapping
        view_id_to_image[view_id] = img_file
        view_id_to_transform[view_id] = T
    
    # Now create sequentially numbered images and build the image_data list
    image_data = []
    
    print(f"Found {len(view_id_to_image)} valid images with camera data")
    
    # Sort by view_id for consistent numbering
    sorted_view_ids = sorted(view_id_to_image.keys())
    
    # Initialize image_data list for COLMAP and depth file mapping
    image_data = []
    depth_files = {} if use_depth else None
        
    # Now create sequentially numbered images (000.png, 001.png, etc.)
    for seq_idx, view_id in enumerate(sorted_view_ids):
        img_file = view_id_to_image[view_id]
        T = view_id_to_transform[view_id]
        
        # Create sequential file name
        seq_name = f"{seq_idx:03d}{'.exr' if img_file.endswith('.exr') else '.png'}"
        source_path = infinigen_images_dir / img_file
        dest_path = images_dir / seq_name
        
        # Create a copy or symlink with the sequential name
        if seq_name != img_file and not dest_path.exists():
            try:
                # Try symlink first
                os.symlink(source_path, dest_path)
                print(f"Created symlink: {img_file} -> {seq_name}")
            except Exception as e:
                # If symlink fails, copy the file
                shutil.copy2(source_path, dest_path)
                print(f"Copied file (symlink failed): {img_file} -> {seq_name}")
        
        # Process depth image if using depth supervision
        if use_depth and depth_dir and depth_images_dir and depth_images_dir.exists():
            # Try to find matching depth image
            depth_files_list = os.listdir(depth_images_dir) if depth_images_dir.exists() else []
            
            # Look for depth image with the same view_id pattern
            matching_depth = [d for d in depth_files_list if f"Depth_{view_id}_" in d or f"Depth_{view_id:02d}_" in d]
            
            if matching_depth:
                depth_img_file = matching_depth[0]
                depth_seq_name = f"{seq_idx:03d}.png"  # Always use PNG for depth
                depth_source_path = depth_images_dir / depth_img_file
                depth_dest_path = depth_dir / depth_seq_name
                
                # Create a copy or symlink for the depth image
                if not depth_dest_path.exists():
                    try:
                        os.symlink(depth_source_path, depth_dest_path)
                        print(f"Created depth symlink: {depth_img_file} -> {depth_seq_name}")
                    except Exception as e:
                        shutil.copy2(depth_source_path, depth_dest_path)
                        print(f"Copied depth file (symlink failed): {depth_img_file} -> {depth_seq_name}")
                
                # Add depth file path to the mapping
                depth_files[seq_name] = f"./depth/{depth_seq_name}"
                print(f"Added depth image {depth_seq_name} (from {depth_img_file}) for view {view_id}")
            else:
                print(f"Warning: No matching depth file found for view ID {view_id}")
        
        # Add to image data for COLMAP - use sequential numbering for file names
        # but keep original view_id for COLMAP internals
        image_data.append((seq_name, T))
        print(f"Added image {seq_name} (from {img_file}) to COLMAP data")
    
    # Create COLMAP images.txt
    create_colmap_images_file(colmap_dir / "images.txt", image_data)
    
    # Create empty points3D.txt
    create_empty_points3d_file(colmap_dir / "points3D.txt")
    
    # Create transforms.json file required by Nerfstudio
    create_transforms_json(scene_dir / "transforms.json", image_data, K, hw, depth_files)
    
    print(f"\nSuccessfully organized data for NeRF Studio at {scene_dir}")
    print(f"Created transforms.json file for Nerfstudio")
    
    if use_depth and depth_dir:
        print(f"Included depth supervision with {len(depth_files) if depth_files else 0} depth images")
        print(f"You should run ns_reconstruction.py with a depth-enabled method like 'depth-nerfacto'")
    
    print(f"Now you can run ns_reconstruction.py with --data_dir {output_dir}")
    print(f"The data is organized in the expected directory structure: {output_dir}/scenes/{scene_id}/")

if __name__ == "__main__":
    args = parse_args()
    organize_for_nerfstudio(args.input_dir, args.output_dir, args.scene_id, args.use_depth)
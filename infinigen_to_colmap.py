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

def create_transforms_json(output_path, image_data, K, hw):
    """Create transforms.json file required by Nerfstudio"""
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

def organize_for_nerfstudio(input_dir, output_dir, scene_id=None):
    """Convert Infinigen data to COLMAP format for NeRF Studio
    
    This function assumes data is already organized by organize_data.sh script with structure:
    /path/to/data/scenes/scene_id/
      images/
        camera_0/
          *.png
      camview/
        camera_0/
          *.npz
    """
    input_dir = Path(input_dir)
    print(f"Input directory: {input_dir}")
    
    # Auto-detect directory structure
    # If scene_id is not provided, use the input directory name
    if scene_id is None:
        scene_id = input_dir.name
        print(f"Using input directory name as scene_id: {scene_id}")
    
    # Check if input_dir is already the scene directory
    # Look for camview and images directories
    if (input_dir / "camview").exists() and (input_dir / "images").exists():
        # This is already the scene directory
        scene_dir = input_dir
        print(f"Input directory is already a scene directory containing camview/ and images/")
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
    images_dir = scene_dir / "images" / "camera_0"
    camview_dir = scene_dir / "camview" / "camera_0"
    
    if not camview_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(f"Missing required directories: \nImages: {images_dir} (exists: {images_dir.exists()})\nCamera: {camview_dir} (exists: {camview_dir.exists()})")
    
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
    
    # Check if images exist (should already be copied by organize_data.sh)
    try:
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.exr')])
        if not image_files:
            raise FileNotFoundError(f"No image files found in {images_dir}")
        
        print(f"Found {len(image_files)} images in {images_dir}")
    except Exception as e:
        print(f"Error accessing image directory: {e}")
        print(f"Directory structure: {scene_dir} contains {os.listdir(scene_dir)}")
        if (scene_dir / "images").exists():
            print(f"Images directory exists: {scene_dir / 'images'} contains {os.listdir(scene_dir / 'images')}")
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
        
        # Look for matching image file in the images directory
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
    
    # Initialize image_data list for COLMAP
    image_data = []
        
    # Now create sequentially numbered images (000.png, 001.png, etc.)
    for seq_idx, view_id in enumerate(sorted_view_ids):
        img_file = view_id_to_image[view_id]
        T = view_id_to_transform[view_id]
        
        # Create sequential file name
        seq_name = f"{seq_idx:03d}{'.exr' if img_file.endswith('.exr') else '.png'}"
        source_path = images_dir / img_file
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
        
        # Add to image data for COLMAP - use sequential numbering for file names
        # but keep original view_id for COLMAP internals
        image_data.append((seq_name, T))
        print(f"Added image {seq_name} (from {img_file}) to COLMAP data")
    
    # Create COLMAP images.txt
    create_colmap_images_file(colmap_dir / "images.txt", image_data)
    
    # Create empty points3D.txt
    create_empty_points3d_file(colmap_dir / "points3D.txt")
    
    # Create transforms.json file required by Nerfstudio
    create_transforms_json(scene_dir / "transforms.json", image_data, K, hw)
    
    print(f"\nSuccessfully organized data for NeRF Studio at {scene_dir}")
    print(f"Created transforms.json file for Nerfstudio")
    print(f"Now you can run ns_reconstruction.py with --data_dir {output_dir}")
    print(f"The data is organized in the expected directory structure: {output_dir}/scenes/{scene_id}/")

if __name__ == "__main__":
    args = parse_args()
    organize_for_nerfstudio(args.input_dir, args.output_dir, args.scene_id)
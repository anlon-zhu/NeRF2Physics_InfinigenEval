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
    parser.add_argument("--use_depth", action="store_true", help="Use depth images instead of RGB images")
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

def organize_for_nerfstudio(input_dir, output_dir, scene_id=None, use_depth=False):
    """Organize Infinigen data for NeRF Studio"""
    input_dir = Path(input_dir)
    if scene_id is None:
        scene_id = input_dir.name
    
    output_dir = Path(output_dir)
    scene_dir = output_dir / scene_id
    os.makedirs(scene_dir, exist_ok=True)
    
    # Create COLMAP directories
    colmap_dir = scene_dir / "sparse" / "0"
    os.makedirs(colmap_dir, exist_ok=True)
    
    images_dir = scene_dir / "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Get camera parameters from the first camera view
    frames_dir = input_dir / "frames"
    camview_dir = frames_dir / "camview" / "camera_0"
    
    # Choose between RGB or Depth images
    if use_depth:
        image_dir = frames_dir / "Depth" / "camera_0"
        image_suffix = "Depth"
    else:
        image_dir = frames_dir / "Image" / "camera_0"
        image_suffix = "Image"
    
    if not camview_dir.exists() or not image_dir.exists():
        raise FileNotFoundError(f"Cannot find {camview_dir} or {image_dir}")
    
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
    
    # Collect all image data
    image_data = []
    
    for cam_file in cam_files:
        # Load camera data
        cam_data = np.load(camview_dir / cam_file)
        T = cam_data['T']
        
        # Extract view number
        view_id = extract_view_id(cam_file)
        if view_id is None:
            continue
        
        # Find corresponding image file
        # Convert camview_10_0_0048_0.npz to Image_10_0_0048_0.png
        img_pattern = f"{image_suffix}_{view_id}_0_0048_0.png"
        
        if not os.path.exists(image_dir / img_pattern):
            # Also check for exr if png not found
            img_pattern = img_pattern.replace(".png", ".exr")
            if not os.path.exists(image_dir / img_pattern):
                print(f"Warning: No matching image found for camera {cam_file}")
                continue
            
        # Copy image to output directory
        dest_img_name = f"{view_id:03d}{'.exr' if img_pattern.endswith('.exr') else '.png'}"
        shutil.copy(
            image_dir / img_pattern,
            images_dir / dest_img_name
        )
        
        # Add to image data for COLMAP
        image_data.append((dest_img_name, T))
        print(f"Processed image {img_pattern} -> {dest_img_name}")
    
    # Create COLMAP images.txt
    create_colmap_images_file(colmap_dir / "images.txt", image_data)
    
    # Create empty points3D.txt
    create_empty_points3d_file(colmap_dir / "points3D.txt")
    
    print(f"\nSuccessfully organized data for NeRF Studio at {scene_dir}")
    print(f"Now you can run ns_reconstruction.py with --data_dir {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    organize_for_nerfstudio(args.input_dir, args.output_dir, args.scene_id, args.use_depth)
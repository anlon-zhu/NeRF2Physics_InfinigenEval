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
    input_dir/
      scenes/
        scene_id/
          images/
            camera_0/
              *.png
          camview/
            camera_0/
              *.npz
    """
    input_dir = Path(input_dir)
    
    # If scene_id is not provided, use the base directory name
    if scene_id is None:
        if (input_dir / "scenes").exists():
            # User provided the base directory, list available scenes
            scenes = [d.name for d in (input_dir / "scenes").iterdir() if d.is_dir()]
            if not scenes:
                raise FileNotFoundError(f"No scene directories found in {input_dir / 'scenes'}")
            print(f"Available scenes: {', '.join(scenes)}")
            scene_id = scenes[0]  # Default to first scene
            print(f"Using first scene: {scene_id}")
        else:
            raise FileNotFoundError(f"No 'scenes' directory found in {input_dir}")
    
    # Navigate to the specific scene directory
    scene_dir = input_dir / "scenes" / scene_id
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory {scene_dir} not found")
    
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
        img_pattern = f"Image_{view_id}_0_0048_0.png"
        
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
    
    # Create transforms.json file required by Nerfstudio
    create_transforms_json(scene_dir / "transforms.json", image_data, K, hw)
    
    print(f"\nSuccessfully organized data for NeRF Studio at {scene_dir}")
    print(f"Created transforms.json file for Nerfstudio")
    print(f"Now you can run ns_reconstruction.py with --data_dir {output_dir}")
    print(f"The data is organized in the expected directory structure: {output_dir}/scenes/{scene_id}/")

if __name__ == "__main__":
    args = parse_args()
    organize_for_nerfstudio(args.input_dir, args.output_dir, args.scene_id)
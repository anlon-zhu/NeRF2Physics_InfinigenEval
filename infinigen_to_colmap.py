#!/usr/bin/env python3
import os
import numpy as np
import json
import shutil
from pathlib import Path
import argparse
import re
import sys

class InfinigenDataset:
    """
    Class to represent the directory structure of Infinigen outputs and 
    provide methods to convert them to NeRF-compatible formats.
    """
    
    def __init__(self, scene_dir):
        """
        Initialize the dataset from a scene directory
        
        Args:
            scene_dir: Path to the scene directory containing frames/ subdirectory
        """
        self.scene_dir = Path(scene_dir)
        self.scene_name = self.scene_dir.name
        self.frames_dir = self.scene_dir / "frames"
        
        # Validate directory structure
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found at {self.frames_dir}")
        
        # Initialize data structures
        self.camera_data = {}  # Map of view_id -> camera parameters
        self.image_data = {}   # Map of view_id -> image file path
        self.depth_data = {}   # Map of view_id -> depth file path
        self.density_data = {} # Map of view_id -> density file path
        
        # Load data
        self._load_camera_data()
        self._load_image_data()
        self._load_depth_data()
        self._load_density_data()
        
        print(f"Loaded Infinigen dataset from {self.scene_dir}")
        print(f"Found {len(self.camera_data)} camera views")
        print(f"Found {len(self.image_data)} images")
        print(f"Found {len(self.depth_data)} depth maps")
        print(f"Found {len(self.density_data)} density maps")
    
    def _load_camera_data(self):
        """Load camera parameters from camview directory"""
        camview_dir = self.frames_dir / "camview" / "camera_0"
        if not camview_dir.exists():
            print(f"Warning: Camera directory not found at {camview_dir}")
            return
        
        for cam_file in camview_dir.glob("*.npz"):
            # Extract view ID from filename (e.g., camview_10_0_0048_0.npz -> 10)
            match = re.search(r'camview_(\d+)_\d+_\d+_\d+\.npz', cam_file.name)
            if not match:
                continue
            
            view_id = int(match.group(1))
            try:
                cam_data = np.load(cam_file)
                self.camera_data[view_id] = {
                    'file': cam_file,
                    'K': cam_data['K'],
                    'T': cam_data['T'],
                    'HW': cam_data['HW']
                }
            except Exception as e:
                print(f"Error loading camera file {cam_file}: {e}")
    
    def _load_image_data(self):
        """Load image paths from Image directory"""
        image_dir = self.frames_dir / "Image" / "camera_0"
        if not image_dir.exists():
            print(f"Warning: Image directory not found at {image_dir}")
            return
        
        for img_file in image_dir.glob("*.png"):
            # Extract view ID from filename (e.g., Image_10_0_0048_0.png -> 10)
            match = re.search(r'Image_(\d+)_\d+_\d+_\d+\.png', img_file.name)
            if not match:
                continue
            
            view_id = int(match.group(1))
            self.image_data[view_id] = img_file
    
    def _load_depth_data(self):
        """Load depth map paths from Depth directory"""
        depth_dir = self.frames_dir / "Depth" / "camera_0"
        if not depth_dir.exists():
            print(f"Warning: Depth directory not found at {depth_dir}")
            return
        
        for depth_file in depth_dir.glob("*.png"):
            # Extract view ID from filename (e.g., Depth_10_0_0048_0.png -> 10)
            match = re.search(r'Depth_(\d+)_\d+_\d+_\d+\.png', depth_file.name)
            if not match:
                continue
            
            view_id = int(match.group(1))
            self.depth_data[view_id] = depth_file
    
    def _load_density_data(self):
        """Load density map paths from MaterialsDensity directory"""
        density_dir = self.frames_dir / "MaterialsDensity" / "camera_0"
        if not density_dir.exists():
            print(f"Warning: Density directory not found at {density_dir}")
            return
        
        for density_file in density_dir.glob("*.npy"):
            # Extract view ID from filename (e.g., MaterialsDensity_10_0_0048_0.npy -> 10)
            match = re.search(r'MaterialsDensity_(\d+)_\d+_\d+_\d+\.npy', density_file.name)
            if not match:
                continue
            
            view_id = int(match.group(1))
            self.density_data[view_id] = density_file
    
    def _create_directory_structure(self, output_dir):
        """Create the directory structure for NeRF export
        
        Args:
            output_dir: Base output directory
            
        Returns:
            Dictionary of created directories
        """
        output_dir = Path(output_dir)
        
        # Create output directories
        scene_dir = output_dir
        ns_dir = scene_dir / 'ns'
        img_dir = scene_dir / 'images'
        depth_dir = ns_dir / 'renders' / 'depth'
        sparse_dir = scene_dir / 'sparse' / '0'
        infinigen_images_dir = scene_dir / 'infinigen_images'
        infinigen_camview_dir = scene_dir / 'infinigen_camview'
        
        # Create all directories
        os.makedirs(ns_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(infinigen_images_dir, exist_ok=True)
        os.makedirs(infinigen_camview_dir, exist_ok=True)
        
        return {
            'scene_dir': scene_dir,
            'ns_dir': ns_dir,
            'img_dir': img_dir,
            'depth_dir': depth_dir,
            'sparse_dir': sparse_dir,
            'infinigen_images_dir': infinigen_images_dir,
            'infinigen_camview_dir': infinigen_camview_dir
        }
    
    def _copy_original_files(self, dirs):
        """Copy original Infinigen files to output directories
        
        Args:
            dirs: Dictionary of directories from _create_directory_structure
        """
        # Copy camera parameter files
        for view_id, cam_data in self.camera_data.items():
            shutil.copy2(cam_data['file'], dirs['infinigen_camview_dir'] / cam_data['file'].name)
        
        # Copy image files
        for view_id, img_file in self.image_data.items():
            shutil.copy2(img_file, dirs['infinigen_images_dir'] / img_file.name)
    
    def _get_camera_intrinsics(self, common_view_ids):
        """Get camera intrinsics from the first camera
        
        Args:
            common_view_ids: List of view IDs that have both camera and image data
            
        Returns:
            Dictionary of camera intrinsics
        """
        K = self.camera_data[common_view_ids[0]]['K']
        HW = self.camera_data[common_view_ids[0]]['HW']
        height, width = HW
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        return {
            'K': K,
            'HW': HW,
            'height': height,
            'width': width,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
    
    def _create_cameras_txt(self, sparse_dir, intrinsics):
        """Create COLMAP cameras.txt file
        
        Args:
            sparse_dir: Path to sparse directory
            intrinsics: Camera intrinsics from _get_camera_intrinsics
        """
        with open(sparse_dir / 'cameras.txt', 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")
            f.write(f"1 PINHOLE {intrinsics['width']} {intrinsics['height']} "
                   f"{intrinsics['fx']} {intrinsics['fy']} {intrinsics['cx']} {intrinsics['cy']}\n")
    
    def _process_images(self, common_view_ids, dirs, intrinsics):
        """Process images and create frame data for COLMAP and nerfstudio
        
        Args:
            common_view_ids: List of view IDs that have both camera and image data
            dirs: Dictionary of directories
            intrinsics: Camera intrinsics
            
        Returns:
            Tuple of (colmap_frames, nerfstudio_frames)
        """
        from scipy.spatial.transform import Rotation
        
        colmap_frames = []
        nerfstudio_frames = []
        
        for idx, view_id in enumerate(common_view_ids):
            # Get camera and image data
            cam_data = self.camera_data[view_id]
            img_file = self.image_data[view_id]
            T = cam_data['T']
            
            # Create sequential file name
            seq_name = f"{idx:03d}.png"
            dest_path = dirs['img_dir'] / seq_name
            
            # Create symlink to original image
            if not dest_path.exists():
                try:
                    os.symlink(img_file, dest_path)
                except Exception as e:
                    shutil.copy2(img_file, dest_path)
            
            # Process depth image if available
            depth_file_path = None
            if view_id in self.depth_data:
                depth_file = self.depth_data[view_id]
                depth_seq_name = f"{idx:03d}.exr"
                depth_dest_path = dirs['depth_dir'] / depth_seq_name
                
                if not depth_dest_path.exists():
                    try:
                        os.symlink(depth_file, depth_dest_path)
                    except Exception as e:
                        shutil.copy2(depth_file, depth_dest_path)
                
                depth_file_path = f"renders/depth/{depth_seq_name}"
            
            # Extract rotation and translation from transformation matrix
            rotation_matrix = T[:3, :3]
            t = T[:3, 3]
            
            # Convert to quaternion for COLMAP
            r = Rotation.from_matrix(rotation_matrix)
            quat = r.as_quat()  # x, y, z, w
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to w, x, y, z
            
            # Create COLMAP frame
            colmap_frame = {
                "file_path": f"images/{seq_name}",
                "height": int(intrinsics['height']),
                "width": int(intrinsics['width']),
                "camera_model": "PINHOLE",
                "camera_params": [float(intrinsics['fx']), float(intrinsics['fy']), 
                                  float(intrinsics['cx']), float(intrinsics['cy'])],
                "rotation": [float(q) for q in quat],
                "translation": [float(t) for t in t]
            }
            colmap_frames.append(colmap_frame)
            
            # Create nerfstudio frame
            nerfstudio_frame = {
                "file_path": f"../images/{seq_name}",
                "transform_matrix": T.tolist()
            }
            if depth_file_path:
                nerfstudio_frame["depth_file_path"] = depth_file_path
            
            nerfstudio_frames.append(nerfstudio_frame)
            
        return colmap_frames, nerfstudio_frames
    
    def _create_colmap_files(self, common_view_ids, dirs, colmap_frames):
        """Create COLMAP format files
        
        Args:
            common_view_ids: List of view IDs that have both camera and image data
            dirs: Dictionary of directories
            colmap_frames: List of COLMAP frame data
        """
        from scipy.spatial.transform import Rotation
        
        # Write COLMAP transforms.json
        colmap_data = {
            "frames": colmap_frames
        }
        with open(dirs['scene_dir'] / 'transforms.json', 'w') as f:
            json.dump(colmap_data, f, indent=2)
        
        # Create COLMAP images.txt
        with open(dirs['sparse_dir'] / 'images.txt', 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(common_view_ids)}\n")
            
            for idx, view_id in enumerate(common_view_ids, 1):
                # Get camera data
                cam_data = self.camera_data[view_id]
                T = cam_data['T']
                
                # Extract rotation and translation
                rotation_matrix = T[:3, :3]
                t = T[:3, 3]
                
                # Convert to quaternion
                r = Rotation.from_matrix(rotation_matrix)
                quat = r.as_quat()  # x, y, z, w
                quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to w, x, y, z
                
                # Normalize quaternion
                quat = quat / np.linalg.norm(quat)
                
                # Write image data
                f.write(f"{idx} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {t[0]} {t[1]} {t[2]} 1 {idx:03d}.png\n")
                # Empty line for POINTS2D (not needed for NeRF)
                f.write("\n")
        
        # Create empty points3D.txt
        with open(dirs['sparse_dir'] / 'points3D.txt', 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0\n")
    
    def _create_nerfstudio_files(self, dirs, nerfstudio_frames, intrinsics=None):
        """Create nerfstudio format files
        
        Args:
            dirs: Dictionary of directories
            nerfstudio_frames: List of nerfstudio frame data
        """
        # Write nerfstudio dataparser_transforms.json
        nerfstudio_data = {
            "camera_model": "OPENCV",
            "fl_x": float(intrinsics['fx']),
            "fl_y": float(intrinsics['fy']),
            "cx": float(intrinsics['cx']),
            "cy": float(intrinsics['cy']),
            "w": int(intrinsics['width']),
            "h": int(intrinsics['height']),
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "frames": nerfstudio_frames,
            "scene_scale": 1.0,
            "scene_center": [0.0, 0.0, 0.0]
        }
        with open(dirs['ns_dir'] / 'dataparser_transforms.json', 'w') as f:
            json.dump(nerfstudio_data, f, indent=2)
    
    def _create_dummy_point_cloud(self, ns_dir):
        """Create a dummy point cloud file
        
        Args:
            ns_dir: Path to nerfstudio directory
        """
        # Create a simple point cloud with a few points
        dummy_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dummy_colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255]
        ], dtype=np.uint8)
        
        # Save as PLY file
        with open(ns_dir / 'point_cloud.ply', 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(dummy_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i in range(len(dummy_points)):
                f.write(f"{dummy_points[i, 0]} {dummy_points[i, 1]} {dummy_points[i, 2]} "
                        f"{dummy_colors[i, 0]} {dummy_colors[i, 1]} {dummy_colors[i, 2]}\n")
    
    def export_to_nerf(self, output_dir, create_point_cloud=True):
        """
        Export the dataset to a NeRF-compatible format
        
        Args:
            output_dir: Path to output directory
            create_point_cloud: Whether to create a dummy point cloud file
        
        Returns:
            Dictionary with paths to the created files
        """
        # Create directory structure
        dirs = self._create_directory_structure(output_dir)
        
        # Copy original files
        self._copy_original_files(dirs)
        
        # Get common view IDs (views that have both camera and image data)
        common_view_ids = sorted(set(self.camera_data.keys()) & set(self.image_data.keys()))
        
        if not common_view_ids:
            raise ValueError("No views with both camera and image data found")
        
        # Get camera intrinsics
        intrinsics = self._get_camera_intrinsics(common_view_ids)
        
        # Create COLMAP cameras.txt
        self._create_cameras_txt(dirs['sparse_dir'], intrinsics)
        
        # Process images and create frame data
        colmap_frames, nerfstudio_frames = self._process_images(common_view_ids, dirs, intrinsics)
        
        # Create COLMAP format files
        self._create_colmap_files(common_view_ids, dirs, colmap_frames)
        
        # Create nerfstudio format files
        self._create_nerfstudio_files(dirs, nerfstudio_frames, intrinsics)
        
        # Create a dummy point cloud file if requested
        if create_point_cloud:
            self._create_dummy_point_cloud(dirs['ns_dir'])
        
        # Return paths to the created files
        return {
            "scene_name": self.scene_name,
            "pcd_file": str(dirs['ns_dir'] / 'point_cloud.ply'),
            "dt_file": str(dirs['ns_dir'] / 'dataparser_transforms.json'),
            "t_file": str(dirs['scene_dir'] / 'transforms.json'),
            "img_dir": str(dirs['img_dir']),
            "depth_dir": str(dirs['depth_dir'])
        }

def main():
    parser = argparse.ArgumentParser(description="Convert Infinigen output to NeRF-compatible format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Infinigen scene directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--scene_id", type=str, default=None, help="Scene ID to process (default: use directory name)")
    parser.add_argument("--create_point_cloud", action="store_true", help="Create a dummy point cloud file")
    args = parser.parse_args()
    
    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        # Determine if input_dir is a scene directory or contains multiple scenes
        if args.scene_id is not None:
            # User specified a scene_id, look for it in input_dir
            print(f"Looking for scene {args.scene_id} in {input_dir}")
            scene_dir = input_dir / args.scene_id
            if not scene_dir.exists():
                raise FileNotFoundError(f"Scene directory {args.scene_id} not found in {input_dir}")
            
            # Process the specific scene
            scene_output_dir = output_dir / args.scene_id
            os.makedirs(scene_output_dir, exist_ok=True)
            
            dataset = InfinigenDataset(scene_dir)
            result = dataset.export_to_nerf(scene_output_dir, args.create_point_cloud)
            
            print(f"\nSuccessfully converted scene {args.scene_id} to NeRF format:")
            print(f"Output directory: {scene_output_dir}")
            
        elif (input_dir / "frames").exists():
            # Input directory is already a scene directory
            print(f"Input directory {input_dir} appears to be a scene directory (contains frames/)")
            scene_name = input_dir.name
            
            dataset = InfinigenDataset(input_dir)
            result = dataset.export_to_nerf(output_dir, args.create_point_cloud)
            
            print(f"\nSuccessfully converted scene {scene_name} to NeRF format:")
            print(f"Output directory: {output_dir}")
            
        else:
            # Input directory might contain multiple scenes
            # Check for subdirectories that could be scene directories
            scene_dirs = [d for d in input_dir.iterdir() if d.is_dir() and (d / "frames").exists()]
            
            if not scene_dirs:
                raise FileNotFoundError(f"No valid scene directories found in {input_dir}")
            
            print(f"Found {len(scene_dirs)} scene directories in {input_dir}")
            
            results = []
            for scene_dir in scene_dirs:
                scene_name = scene_dir.name
                print(f"\nProcessing scene: {scene_name}")
                
                scene_output_dir = output_dir / scene_name
                os.makedirs(scene_output_dir, exist_ok=True)
                
                try:
                    dataset = InfinigenDataset(scene_dir)
                    result = dataset.export_to_nerf(scene_output_dir, args.create_point_cloud)
                    results.append((scene_name, result))
                    print(f"Successfully converted scene {scene_name}")
                except Exception as e:
                    print(f"Error processing scene {scene_name}: {e}")
            
            print(f"\nProcessed {len(results)} scenes successfully")
            for scene_name, result in results:
                print(f"- {scene_name}: {result['t_file']}")
        
        # Print example usage for the function
        print("\nExample usage for your function:")
        print(f"scene_dir = '{output_dir}'")
        print(f"scene_name = os.path.basename(scene_dir)")
        print(f"pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')")
        print(f"dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')")
        print(f"t_file = os.path.join(scene_dir, 'transforms.json')")
        print(f"img_dir = os.path.join(scene_dir, 'images')")
        print(f"depth_dir = os.path.join(scene_dir, 'ns', 'renders', 'depth')")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
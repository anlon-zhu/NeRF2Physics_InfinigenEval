"""
Density Evaluation Configuration
This file defines standard paths and constants for both density_evaluation.py and density_visualization.py
to ensure consistency between the evaluation and visualization processes.
"""

import os


class PathConfig:
    """Configuration class for standardized file paths and naming conventions"""
    
    # Directory structure
    @staticmethod
    def get_scenes_dir(data_dir):
        """Path to the scenes directory"""
        return os.path.join(data_dir, 'scenes')
    
    @staticmethod
    def get_scene_dir(data_dir, scene_name):
        """Path to a specific scene directory"""
        return os.path.join(PathConfig.get_scenes_dir(data_dir), scene_name)
    
    # Ground truth paths
    @staticmethod
    def get_gt_density_dir(scene_dir):
        """Path to ground truth density directory for a scene"""
        return os.path.join(scene_dir, 'gt_density')
    
    @staticmethod
    def get_gt_density_file(gt_dir, view_idx):
        """
        Path to ground truth density file for a specific view.
        Searches for different naming patterns.
        """
        # Try different naming patterns in order of preference
        patterns = [
            f'density_{view_idx:03d}.npy',  # zero-padded
            f'density_{view_idx}.npy',      # no padding
            f'density_{view_idx:03d}.png',  # zero-padded (PNG)
            f'density_{view_idx}.png'       # no padding (PNG)
        ]
        
        for pattern in patterns:
            path = os.path.join(gt_dir, pattern)
            if os.path.exists(path):
                return path
        
        return None
    
    # Output directories
    @staticmethod
    def get_evaluation_output_dir(scene_dir):
        """Path to density evaluation output directory for a scene"""
        return os.path.join(scene_dir, 'density_evaluation')
    
    # Output files for density evaluation
    @staticmethod
    def get_predicted_density_map_file(output_dir, view_idx, format='npy'):
        """Path to predicted density map file for a specific view"""
        if format.lower() == 'png':
            return os.path.join(output_dir, f'predicted_density_values_view_{view_idx}.png')
        elif format.lower() == 'npz':
            return os.path.join(output_dir, f'predicted_density_values_view_{view_idx}.npz')
        else:  # Default to npy
            return os.path.join(output_dir, f'predicted_density_values_view_{view_idx}.npy')
    
    @staticmethod
    def get_metrics_file(output_dir):
        """Path to the metrics JSON file for a scene"""
        return os.path.join(output_dir, 'density_metrics.json')
    
    @staticmethod
    def get_avg_metrics_file(output_dir):
        """Path to the average metrics JSON file for a scene"""
        return os.path.join(output_dir, 'density_metrics_avg.json')
    
    # Input/model paths
    @staticmethod
    def get_point_cloud_file(scene_dir):
        """Path to the NeRFStudio point cloud file"""
        return os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    
    @staticmethod
    def get_transforms_file(scene_dir):
        """Path to the transforms JSON file"""
        return os.path.join(scene_dir, 'transforms.json')
    
    @staticmethod
    def get_dataparser_transforms_file(scene_dir):
        """Path to the dataparser transforms JSON file"""
        return os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')


class VisualizationConfig:
    """Configuration for visualization parameters"""
    
    # Default colormap for density visualization
    DENSITY_COLORMAP = 'inferno'
    DIFF_COLORMAP = 'inferno'
    MASK_COLORMAP = 'binary'
    
    # Grid visualization config
    GRID_SIZE = (10, 10)  # 10x10 grid
    DEFAULT_VIEW_INDEX = 0  # Default view index for grid visualizations
    
    # Multi-view config
    MULTI_VIEW_INDICES = list(range(0, 30, 3))[:9]  # Views 0, 3, 6, ..., 27 (9 total)
    
    # Default case study scene
    DEFAULT_CASE_STUDY_SCENE = 'scene_42'
    
    # Output file names
    TABLE_CSV_FILENAME = 'metrics_summary_table.csv'
    TABLE_LATEX_FILENAME = 'metrics_summary_table.tex'
    METRICS_DIST_FILENAME = 'metrics_distribution.png'
    SCENEWISE_ADE_FILENAME = 'scenewise_ade.png'
    GRID_DIFF_FILENAME_TEMPLATE = 'grid_diff_density_view{}.png'
    GRID_PRED_FILENAME_TEMPLATE = 'grid_pred_density_view{}.png'
    GRID_MASK_FILENAME_TEMPLATE = 'grid_valid_mask_view{}.png'
    MULTIVIEW_FILENAME_TEMPLATE = '{}_multiview_comparison.png'


class EvaluationConfig:
    """Configuration for evaluation parameters"""
    
    # Default colormap range for density visualization
    DEFAULT_CMAP_MIN = 0
    DEFAULT_CMAP_MAX = 5000  # kg/m³
    
    # Image resolution for rendering
    DEFAULT_IMAGE_RESOLUTION = (720, 1080)  # (height, width)
    
    # Point cloud sampling
    DEFAULT_SAMPLE_VOXEL_SIZE = 0.05  # Downsample point cloud
    
    # Metric calculation parameters
    DENSITY_EPSILON = 1e-8  # Small value to avoid division by zero
    MIN_VALID_PIXELS = 10   # Minimum number of valid pixels for evaluation

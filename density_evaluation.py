import os
import json
import torch
import numpy as np
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from predict_property import predict_physical_property_query
from utils import load_ns_point_cloud, parse_transforms_json, load_images, parse_dataparser_transforms_json
from arguments import get_args
from evaluation import ADE, ALDE, APE, MnRE, show_metrics
from visualization import render_pcd_headless, values_to_colors


def predict_point_densities(args, scene_dir, clip_model, clip_tokenizer):
    """
    Get density predictions for points in the scene.
    """
    # Use point cloud from NS
    pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
    
    # Load points
    query_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.sample_voxel_size)
    query_pts = torch.Tensor(query_pts).to(args.device)
    
    # Get full prediction information
    prediction_info = predict_physical_property_query(
        args, query_pts, scene_dir, clip_model, clip_tokenizer, return_all=True
    )
    
    # Extract point coordinates and their predicted density values
    points = query_pts.cpu().numpy()
    density_values = prediction_info['query_pred_vals']
    
    return {
        'points': points,
        'density_values': density_values,
        'query_pred_probs': prediction_info['query_pred_probs'],
        'material_names': prediction_info['mat_names']
    }


def render_density_from_camera_view(points, density_values, w2c, K, hw=(1024, 1024), cmap_min=0, cmap_max=10000):
    """
    Render the density values from a specific camera view.
    Returns both the RGB visualization and a single-channel image with actual density values.
    """
    h, w = hw
    # Create point cloud with colors based on density values
    val_pcd = o3d.geometry.PointCloud()
    val_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Use mean of min-max range for visualization and data
    mean_density = np.mean(density_values, axis=1)
    
    # Create the colored point cloud for visualization
    colors = values_to_colors(mean_density, cmap_min, cmap_max)
    val_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Render from the camera view to get the RGB visualization
    rendered_rgb = render_pcd_headless(val_pcd, w2c, K, hw=hw)
    
    # Now create a density value map using the same camera parameters
    # We'll create this directly rather than extracting it from the RGB image
    
    # Create a density image filled with zeros (background)
    density_image = np.zeros(hw)
    
    # Project the 3D points to 2D pixel coordinates
    points_np = np.asarray(val_pcd.points)
    
    # Convert from world to camera coordinates
    pts_cam = (w2c[:3, :3] @ points_np.T).T + w2c[:3, 3]
    
    # Project to image coordinates
    pts_2d = np.zeros((len(pts_cam), 2))
    # Only project points in front of the camera (z > 0)
    valid_pts = pts_cam[:, 2] > 0
    if np.any(valid_pts):
        pts_2d[valid_pts, 0] = K[0, 0] * pts_cam[valid_pts, 0] / pts_cam[valid_pts, 2] + K[0, 2]
        pts_2d[valid_pts, 1] = K[1, 1] * pts_cam[valid_pts, 1] / pts_cam[valid_pts, 2] + K[1, 2]
    
    # Round to nearest pixel and check if within image bounds
    pts_2d = np.round(pts_2d).astype(int)
    in_image = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    
    # Get the points that are both valid (in front of camera) and within image bounds
    valid_indices = np.where(valid_pts & in_image)[0]
    
    if len(valid_indices) > 0:
        # Sort by depth (furthest first) so closer points overwrite further ones
        z_depths = pts_cam[valid_indices, 2]
        sorted_idxs = np.argsort(-z_depths)  # Negative for descending order
        
        # Fill in the density values (overwriting as we go from far to near)
        for idx in sorted_idxs:
            point_idx = valid_indices[idx]
            x, y = pts_2d[point_idx]
            density_image[y, x] = mean_density[point_idx]
    
    # Return both the RGB visualization and the density value image
    return rendered_rgb, density_image


def load_ground_truth_density(gt_image_path):
    """
    Load a ground truth density image and extract its colormap range.
    Returns the density values and the min/max values used for colorization.
    """
    if gt_image_path.endswith('.npy'):
        # Load raw numpy array with actual density values
        density_data = np.load(gt_image_path)
        # Filter out -1 values (if any) for min/max calculation
        valid_data = density_data[density_data != -1]
        if len(valid_data) > 0:
            vmin, vmax = np.min(valid_data), np.max(valid_data)
        else:
            vmin, vmax = 0, 1  # Default fallback
        return density_data, vmin, vmax
    else:
        # For image files, we need to estimate the original density range
        # First load as grayscale
        img = Image.open(gt_image_path).convert('L')
        density_data = np.array(img)
        
        # Check if we have a corresponding .json file with metadata
        json_path = gt_image_path.replace('.png', '.json').replace('.jpg', '.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    if 'density_min' in metadata and 'density_max' in metadata:
                        vmin = metadata['density_min']
                        vmax = metadata['density_max']
                        return density_data, vmin, vmax
            except:
                pass  # Continue with default approach if JSON parsing fails
        
        # If no metadata found, estimate from the non-zero values in the image
        # This assumes the image was created using a similar colormap approach
        valid_data = density_data[density_data > 0]  # Ignore black background
        if len(valid_data) > 0:
            # Scale back to estimated original range
            # This is an estimation - if exact values are needed, they should be stored in metadata
            vmin, vmax = 0, 5000  # Default density range estimate for normalization
        else:
            vmin, vmax = 0, 1  # Default fallback
            
        return density_data, vmin, vmax


def evaluate_density_against_gt(rendered_density, rendered_rgb, gt_density, gt_vmin, gt_vmax, is_gt_npy=False):
    """
    Compare rendered density image with ground truth density image.
    Always use single-valued density for evaluation, while RGB is used only for visualization.
    """
    # Print shapes for debugging
    print(f"Rendered density shape: {rendered_density.shape}, Rendered RGB shape: {rendered_rgb.shape}, GT density shape: {gt_density.shape}")
    
    # Always use single-channel rendered density values for evaluation
    rendered_version = rendered_density
    print("Using single-channel density values for evaluation")
    
    # If ground truth is RGB, convert to grayscale for comparison
    if len(gt_density.shape) == 3 and gt_density.shape[2] == 3:
        print("Converting RGB ground truth to grayscale for evaluation")
        gt_density = np.mean(gt_density, axis=2)
    
    # Resize ground truth if needed
    if rendered_version.shape[:2] != gt_density.shape[:2]:
        print(f"Resizing ground truth from {gt_density.shape[:2]} to {rendered_version.shape[:2]}")
        gt_density_resized = np.array(Image.fromarray(gt_density).resize(
            (rendered_version.shape[1], rendered_version.shape[0]), Image.BILINEAR))
    else:
        gt_density_resized = gt_density
    
    # Scale rendered density to match ground truth range
    rendered_version_scaled = np.copy(rendered_version)
    if rendered_version.max() > 0:  # Avoid division by zero
        # Scale rendered version to the same range as ground truth
        rendered_min = rendered_version.min()
        rendered_max = rendered_version.max()
        # Linear mapping from [rendered_min, rendered_max] to [gt_vmin, gt_vmax]
        rendered_version_scaled = ((rendered_version - rendered_min) / (rendered_max - rendered_min)) * (gt_vmax - gt_vmin) + gt_vmin
    
    # Normalize both to [0, 1] for comparison
    if gt_density_resized.max() > gt_density_resized.min():
        gt_normalized = (gt_density_resized - gt_density_resized.min()) / (gt_density_resized.max() - gt_density_resized.min())
    else:
        gt_normalized = np.zeros_like(gt_density_resized)
        
    if rendered_version_scaled.max() > rendered_version_scaled.min():
        rendered_normalized = (rendered_version_scaled - rendered_version_scaled.min()) / (rendered_version_scaled.max() - rendered_version_scaled.min())
    else:
        rendered_normalized = np.zeros_like(rendered_version_scaled)
    
    # Convert to prediction format for metrics
    # Always use single-channel processing
    pixel_values = rendered_normalized.flatten()
    gt_values = gt_normalized.flatten()
    
    # Print shapes after flattening for debugging
    print(f"Flattened pixel values shape: {pixel_values.shape}, GT values shape: {gt_values.shape}")
    
    # Filter out black background pixels
    valid_mask = gt_values > 0.01  # Threshold to avoid background
    valid_pred = pixel_values[valid_mask]
    valid_gt = gt_values[valid_mask]
    
    # Format for metrics
    valid_pred_ranges = np.stack([valid_pred * 0.8, valid_pred * 1.2], axis=1)  # Create min-max ranges
        
    # Calculate metrics
    metrics = {
        'ADE': ADE(valid_pred_ranges, valid_gt),
        'ALDE': ALDE(valid_pred_ranges, valid_gt),
        'APE': APE(valid_pred_ranges, valid_gt),
        'MnRE': MnRE(valid_pred_ranges, valid_gt)
    }
    
    return metrics


def run_density_evaluation(args):
    """
    Main function to run density evaluation.
    """
    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scene_dir = os.path.join(scenes_dir, args.scene_name)
    
    # Set up paths
    gt_density_dir = os.path.join(scene_dir, 'gt_density')
    t_file = os.path.join(scene_dir, 'transforms.json')
    output_dir = os.path.join(scene_dir, 'density_evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)
    
    # Get density predictions for all points
    print(f"Predicting densities for scene: {args.scene_name}")
    density_results = predict_point_densities(args, scene_dir, clip_model, clip_tokenizer)

    breakpoint()
    
    # Check if we have ground truth density images
    if not os.path.exists(gt_density_dir):
        print(f"Warning: Ground truth density directory not found: {gt_density_dir}")
        print("Creating visualization only, no evaluation will be performed.")
        perform_evaluation = False
    else:
        gt_files = [f for f in os.listdir(gt_density_dir) if f.endswith('.png') or f.endswith('.npy')]
        if not gt_files:
            print(f"Warning: No ground truth density images found in {gt_density_dir}")
            perform_evaluation = False
        else:
            perform_evaluation = True
    
    # Load camera transforms
    w2cs, K = parse_transforms_json(t_file, return_w2c=True)
    
    # Determine colormap range from ground truth if available
    cmap_min = args.cmap_min
    cmap_max = args.cmap_max
    
    if perform_evaluation:
        # Try to get colormap range from the first ground truth image
        for f in gt_files:
            if f.startswith('density_'):
                gt_file = os.path.join(gt_density_dir, f)
                try:
                    _, gt_vmin, gt_vmax = load_ground_truth_density(gt_file)
                    print(f"Using density range from ground truth: {gt_vmin:.2f} to {gt_vmax:.2f} kg/m³")
                    cmap_min = gt_vmin
                    cmap_max = gt_vmax
                    break
                except Exception as e:
                    print(f"Could not extract range from ground truth: {e}")
                    # Keep using the command line arguments
    
    # Render density from different camera views and compare with ground truth
    all_metrics = {}
    
    for view_idx, w2c in enumerate(w2cs):
        print(f"Processing view {view_idx}")
        
        # Convert from nerfstudio to open3d camera format
        w2c_o3d = w2c.copy()
        w2c_o3d[[1, 2]] *= -1
        
        # Render density from this view - now returns both RGB visualization and raw density values
        rendered_rgb, rendered_density = render_density_from_camera_view(
            density_results['points'],
            density_results['density_values'],
            w2c_o3d, K,
            cmap_min=cmap_min,
            cmap_max=cmap_max
        )
        
        # Save the RGB visualization 
        plt.figure(figsize=(10, 10))
        plt.imshow(rendered_rgb)
        plt.colorbar(label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
        plt.title(f'Predicted Density RGB Visualization - View {view_idx}')
        plt.savefig(os.path.join(output_dir, f'predicted_density_rgb_view_{view_idx}.png'))
        plt.close()
        
        # Save the actual density values image
        plt.figure(figsize=(10, 10))
        plt.imshow(rendered_density, cmap='inferno')
        plt.colorbar(label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
        plt.title(f'Predicted Density Values - View {view_idx}')
        plt.savefig(os.path.join(output_dir, f'predicted_density_values_view_{view_idx}.png'))
        plt.close()
        
        # Also save the raw numpy array for later use
        np.save(os.path.join(output_dir, f'predicted_density_values_view_{view_idx}.npy'), rendered_density)
        
        # Compare with ground truth if available
        if perform_evaluation:
            # First try NPY (single-point density values) with different naming patterns
            gt_file = os.path.join(gt_density_dir, f'density_{view_idx:03d}.npy')
            if not os.path.exists(gt_file):
                gt_file = os.path.join(gt_density_dir, f'density_{view_idx}.npy')
            
            # If NPY not found, fall back to PNG (3-channel visualization)
            if not os.path.exists(gt_file):
                gt_file = os.path.join(gt_density_dir, f'density_{view_idx:03d}.png')
            if not os.path.exists(gt_file):
                gt_file = os.path.join(gt_density_dir, f'density_{view_idx}.png')
            
            if os.path.exists(gt_file):
                gt_data, gt_vmin, gt_vmax = load_ground_truth_density(gt_file)
                
                # Determine if ground truth is NPY format
                is_gt_npy = gt_file.endswith('.npy')
                
                # Evaluate using the appropriate rendered version based on ground truth format
                metrics = evaluate_density_against_gt(
                    rendered_density, 
                    rendered_rgb, 
                    gt_data, 
                    gt_vmin, 
                    gt_vmax, 
                    is_gt_npy=is_gt_npy
                )
                all_metrics[f'view_{view_idx}'] = metrics
                
                # Save comparison visualization
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                # For visualization, always show RGB for better visual appeal
                plt.imshow(rendered_rgb)
                plt.title('Predicted Density (RGB)')
                plt.colorbar(label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
                
                plt.subplot(1, 3, 2)
                plt.imshow(gt_data, cmap='inferno')
                plt.title('Ground Truth Density')
                plt.colorbar(label=f'Density (kg/m³) [{gt_vmin:.1f} - {gt_vmax:.1f}]')
                
                plt.subplot(1, 3, 3)
                # Scale for difference visualization
                # Always use the single-valued density for difference calculation
                display_rendered = rendered_density
                        
                norm_rendered = (display_rendered - display_rendered.min()) / (display_rendered.max() - display_rendered.min() + 1e-8)
                norm_gt = (gt_data - gt_data.min()) / (gt_data.max() - gt_data.min() + 1e-8)
                diff = np.abs(norm_rendered - norm_gt)
                plt.imshow(diff, cmap='plasma')
                plt.title('Absolute Difference')
                plt.colorbar(label='Normalized Difference')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'density_comparison_view_{view_idx}.png'))
                plt.close()
                
                # Print metrics
                print(f"Metrics for view {view_idx}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
    
    # Save all metrics
    if all_metrics:
        with open(os.path.join(output_dir, 'density_metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Calculate and print average metrics across all views
        avg_metrics = {metric: np.mean([m[metric] for m in all_metrics.values()]) 
                      for metric in next(iter(all_metrics.values()))}
        
        print("\nAverage metrics across all views:")
        for metric_name, value in avg_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        with open(os.path.join(output_dir, 'density_metrics_avg.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=4)
    
    print(f"Density evaluation complete. Results saved to {output_dir}")


if __name__ == '__main__':
    args = get_args()
    run_density_evaluation(args)

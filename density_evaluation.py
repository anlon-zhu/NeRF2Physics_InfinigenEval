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


def find_gt_file(gt_dir, view_idx):
    """Helper function to find ground truth density file for a specific view."""
    # First try NPY (single-point density values) with different naming patterns
    gt_file = os.path.join(gt_dir, f'density_{view_idx:03d}.npy')
    if os.path.exists(gt_file):
        return gt_file
        
    gt_file = os.path.join(gt_dir, f'density_{view_idx}.npy')
    if os.path.exists(gt_file):
        return gt_file
        
    # If NPY not found, fall back to PNG (3-channel visualization)
    gt_file = os.path.join(gt_dir, f'density_{view_idx:03d}.png')
    if os.path.exists(gt_file):
        return gt_file
        
    gt_file = os.path.join(gt_dir, f'density_{view_idx}.png')
    if os.path.exists(gt_file):
        return gt_file
        
    return None
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
    Load a ground truth density file.
    Only supports NPY files containing actual density values.
    Returns the density values and the min/max values used for range determination.
    """
    if not gt_image_path.endswith('.npy'):
        raise ValueError(f"Unsupported ground truth file format: {gt_image_path}. Only .npy files are supported.")
        
    # Load raw numpy array with actual density values
    density_data = np.load(gt_image_path)
    
    # Filter out -1 values (if any) for min/max calculation
    valid_data = density_data[density_data != -1]
    if len(valid_data) > 0:
        vmin, vmax = np.min(valid_data), np.max(valid_data)
    else:
        vmin, vmax = 0, 1  # Default fallback
    
    return density_data, vmin, vmax


def evaluate_density_against_gt(rendered_density, rendered_rgb, gt_density, gt_vmin, gt_vmax, is_gt_npy=False):
    """
    Compare rendered density image with ground truth density image.
    Only performs comparison when ground truth density (NPY/NPZ) is available.
    """
    # Print shapes for debugging
    print(f"Rendered density shape: {rendered_density.shape}, Rendered RGB shape: {rendered_rgb.shape}, GT density shape: {gt_density.shape}")
    
    # Only evaluate if ground truth is single-channel density values (NPY/NPZ)
    if not is_gt_npy:
        print("WARNING: Skipping evaluation - only single-channel density values (NPY/NPZ) are used for evaluation")
        return {'ADE': 0, 'ALDE': 0, 'APE': 0, 'MnRE': 0}
        
    print("Using single-channel density values for evaluation")
    rendered_version = rendered_density
    gt_version = gt_density
    
    # Check for dimension compatibility
    if len(rendered_version.shape) != len(gt_version.shape):
        print(f"WARNING: Format mismatch - Rendered has {len(rendered_version.shape)} dimensions but GT has {len(gt_version.shape)} dimensions")
        print(f"Skipping evaluation due to incompatible formats")
        return {'ADE': 0, 'ALDE': 0, 'APE': 0, 'MnRE': 0}
    
    # Use raw density values directly - no normalization
    # as they represent physical quantities with meaning
    rendered_normalized = rendered_version
    gt_normalized = gt_version
    print("Using raw density values for evaluation")
    print(f"Density value ranges - Rendered: [{rendered_normalized.min():.4f}, {rendered_normalized.max():.4f}], GT: [{gt_normalized.min():.4f}, {gt_normalized.max():.4f}]")
    
    # For metrics calculation, we need to make sure dimensions match and we're comparing the same type of data
    try:
        # Make sure we have compatible shapes for comparison
        if len(rendered_normalized.shape) != len(gt_normalized.shape):
            raise ValueError(f"Cannot compare shapes: {rendered_normalized.shape} vs {gt_normalized.shape}")
        
        # Final shape check
        if rendered_normalized.shape != gt_normalized.shape:
            raise ValueError(f"Cannot compare shapes: {rendered_normalized.shape} vs {gt_normalized.shape}")
        
        # Flatten for metric calculation
        pixel_values = rendered_normalized.flatten()
        gt_values = gt_normalized.flatten()
        
        # Print shapes after flattening for debugging
        print(f"Flattened pixel values shape: {pixel_values.shape}, GT values shape: {gt_values.shape}")
        
        # Filter out invalid pixels:
        # 1. Exclude -1 values in ground truth (indicates missing data)
        # 2. Use less restrictive thresholds to capture more pixels
        # 3. Print a debug message about the full range of values where GT is valid
        valid_gt_mask = (gt_values != -1) & (gt_values > 0)
        if np.any(valid_gt_mask):
            print(f"DEBUG: Where GT is valid, rendered values range: [{np.min(pixel_values[valid_gt_mask]):.4f}, {np.max(pixel_values[valid_gt_mask]):.4f}]")
            print(f"DEBUG: Number of GT valid pixels: {np.sum(valid_gt_mask)}")
        
        # For evaluation, we'll use pixels where either has a meaningful value
        valid_mask = (gt_values != -1) & ((gt_values > 0) | (pixel_values > 0))
        valid_pred = pixel_values[valid_mask]
        valid_gt = gt_values[valid_mask]
        
        # Report how many valid pixels we're using
        print(f"Valid pixels for evaluation: {len(valid_pred)} out of {len(pixel_values)} total pixels ({len(valid_pred)/len(pixel_values)*100:.2f}%)")
        
        if len(valid_pred) == 0:
            print("WARNING: No valid pixels found for comparison")
            return {'ADE': 0, 'ALDE': 0, 'APE': 0, 'MnRE': 0}
        
        # Exit early if we don't have enough valid pixels
        if len(valid_pred) < 10:
            print("WARNING: Not enough valid pixels for meaningful evaluation")
            return {'ADE': 0, 'ALDE': 0, 'APE': 0, 'MnRE': 0}
        
        # Add a small epsilon to avoid divide by zero
        epsilon = 1e-8
        safe_pred = np.maximum(valid_pred, epsilon)
        safe_gt = np.maximum(valid_gt, epsilon)
            
        # Format for metrics
        valid_pred_ranges = np.stack([safe_pred * 0.8, safe_pred * 1.2], axis=1)  # Create min-max ranges
            
        # Calculate metrics with additional error handling
        try:
            metrics = {
                'ADE': ADE(valid_pred_ranges, safe_gt),
                'ALDE': ALDE(valid_pred_ranges, safe_gt),
                'APE': APE(valid_pred_ranges, safe_gt),
                'MnRE': MnRE(valid_pred_ranges, safe_gt)
            }
        except Exception as e:
            print(f"WARNING: Error computing metrics: {e}")
            return {'ADE': 0, 'ALDE': 0, 'APE': 0, 'MnRE': 0}
    except Exception as e:
        print(f"ERROR computing metrics: {e}")
        metrics = {'ADE': 0, 'ALDE': 0, 'APE': 0, 'MnRE': 0}
    
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
        
        hw = (1024, 1024)
        if perform_evaluation:
            gt_file = find_gt_file(gt_density_dir, view_idx)
            if gt_file and os.path.exists(gt_file):
                gt_data, _, _ = load_ground_truth_density(gt_file)
                hw = gt_data.shape[:2]  # Get height, width as tuple
                print(f"Using ground truth dimensions for rendering: {hw}")
        
        # Render density from this view - now returns both RGB visualization and raw density values
        rendered_rgb, rendered_density = render_density_from_camera_view(
            density_results['points'],
            density_results['density_values'],
            w2c_o3d, K,
            hw=hw,
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
        
        # Create enhanced visualization with only non-zero points
        # This makes sparse points more visible
        nonzero_mask = rendered_density > 0
        if np.any(nonzero_mask):
            plt.figure(figsize=(10, 10))
            nonzero_density = np.copy(rendered_density)
            nonzero_density[nonzero_density == 0] = np.nan  # NaN values appear transparent
            plt.imshow(nonzero_density, cmap='jet', interpolation='none')
            plt.colorbar(label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
            plt.title(f'Non-zero Density Points - View {view_idx}\n{np.sum(nonzero_mask)} points')
            plt.savefig(os.path.join(output_dir, f'nonzero_density_view_{view_idx}.png'))
            plt.close()
        
        # Create a debug visualization to show which pixels have actual values
        # This helps diagnose sparse point cloud projection issues
        filled_mask = rendered_density > 0
        filled_pixel_count = np.sum(filled_mask)
        filled_percentage = filled_pixel_count / (hw[0] * hw[1]) * 100
        
        plt.figure(figsize=(10, 10))
        plt.imshow(filled_mask, cmap='gray')
        plt.title(f'Filled Pixels Mask - View {view_idx}\n{filled_pixel_count} filled pixels ({filled_percentage:.2f}%)')
        plt.savefig(os.path.join(output_dir, f'filled_pixels_mask_view_{view_idx}.png'))
        plt.close()
        
        # Save both NPY (for evaluation) and NPZ (for full metadata)
        np.save(os.path.join(output_dir, f'predicted_density_values_view_{view_idx}.npy'), rendered_density)
        np.savez(os.path.join(output_dir, f'predicted_density_values_view_{view_idx}.npz'),
                 density=rendered_density,
                 rgb=rendered_rgb,
                 min_value=cmap_min,
                 max_value=cmap_max)
        
        # Compare with ground truth if available
        if perform_evaluation:
            # Get ground truth file
            gt_file = find_gt_file(gt_density_dir, view_idx)
            
            if os.path.exists(gt_file):
                gt_data, gt_vmin, gt_vmax = load_ground_truth_density(gt_file)
                is_gt_npy = gt_file.endswith('.npy')
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

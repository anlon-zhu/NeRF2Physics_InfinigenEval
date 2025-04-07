import os
import json
import torch
import numpy as np
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d

# Import our configuration
from density_config import PathConfig, EvaluationConfig, VisualizationConfig

from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from predict_property import predict_physical_property_query
from utils import load_ns_point_cloud, parse_transforms_json, load_images, parse_dataparser_transforms_json
from arguments import get_args
from evaluation import ADE, ALDE, APE, MnRE, show_metrics


def find_gt_file(gt_dir, view_idx):
    """Helper function to find ground truth density file for a specific view."""
    from density_config import PathConfig
    return PathConfig.get_gt_density_file(gt_dir, view_idx)
from visualization import render_pcd_headless, values_to_colors


def predict_point_densities(args, scene_dir, clip_model, clip_tokenizer):
    """
    Get density predictions for points in the scene.
    """
    # Use point cloud from NS
    pcd_file = PathConfig.get_point_cloud_file(scene_dir)
    dt_file = PathConfig.get_dataparser_transforms_file(scene_dir)
    
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


def render_density_from_camera_view(points, density_values, w2c, K, hw=None, cmap_min=None, cmap_max=None):
    # Use defaults from config if not specified
    if hw is None:
        hw = EvaluationConfig.DEFAULT_IMAGE_RESOLUTION
    if cmap_min is None:
        cmap_min = EvaluationConfig.DEFAULT_CMAP_MIN
    if cmap_max is None:
        cmap_max = EvaluationConfig.DEFAULT_CMAP_MAX
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
    # Set up paths using PathConfig
    scenes_dir = PathConfig.get_scenes_dir(args.data_dir)
    scene_dir = PathConfig.get_scene_dir(args.data_dir, args.scene_name)
    
    # Set up additional paths
    gt_density_dir = PathConfig.get_gt_density_dir(scene_dir)
    t_file = PathConfig.get_transforms_file(scene_dir)
    output_dir = PathConfig.get_evaluation_output_dir(scene_dir)
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
    
    # Store rendered data for grid visualization
    rendered_data = []
    gt_data_list = []
    predicted_density_list = []
    
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
        
        # Save the actual density values image (we're keeping this one)
        plt.figure(figsize=(10, 10))
        plt.imshow(rendered_density, cmap=VisualizationConfig.DENSITY_COLORMAP)
        plt.colorbar(label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
        plt.title(f'Predicted Density Values - View {view_idx}')
        plt.savefig(PathConfig.get_predicted_density_map_file(output_dir, view_idx, 'png'))
        plt.close()
        
        # Create enhanced visualization with only non-zero points
        # This makes sparse points more visible
        nonzero_mask = rendered_density > 0
        if np.any(nonzero_mask):
            nonzero_density = np.copy(rendered_density)
            nonzero_density[nonzero_density == 0] = np.nan  # NaN values appear transparent
            
            # Store for grid visualization
            rendered_data.append((view_idx, rendered_rgb, nonzero_density))
        
        # Save both NPY (for evaluation) and NPZ (for full metadata)
        np.save(PathConfig.get_predicted_density_map_file(output_dir, view_idx, 'npy'), rendered_density)
        np.savez(PathConfig.get_predicted_density_map_file(output_dir, view_idx, 'npz'),
                 density=rendered_density,
                 rgb=rendered_rgb,
                 min_value=cmap_min,
                 max_value=cmap_max)
        
        # Compare with ground truth if available
        if perform_evaluation:
            # Get ground truth file
            predicted_density_list.append((view_idx, rendered_density))
            gt_file = find_gt_file(gt_density_dir, view_idx)
            
            if os.path.exists(gt_file):
                gt_data, gt_vmin, gt_vmax = load_ground_truth_density(gt_file)
                gt_data_list.append((view_idx, gt_data))
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
                # Print metrics
                print(f"Metrics for view {view_idx}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
    
    # Create 3x3 grid visualization of predicted density
    if rendered_data:
        # Sort by view index and sample every 3rd view until we get 9 views
        rendered_data.sort(key=lambda x: x[0])
        sampled_views = rendered_data[::max(1, len(rendered_data) // 9)][:9]
        
        # If we have fewer than 9 views, use all available views
        if len(sampled_views) < 9:
            sampled_views = rendered_data[:min(9, len(rendered_data))]
        
        # Create the grid
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, (view_idx, rgb, nonzero_density) in enumerate(sampled_views):
            if i < 9:  # Ensure we don't exceed the grid size
                ax = axes[i]
                im = ax.imshow(nonzero_density, cmap='jet', interpolation='none')
                ax.set_title(f'View {view_idx}')
                ax.axis('off')
        
        # Hide any unused subplots
        for i in range(len(sampled_views), 9):
            axes[i].axis('off')
        
        # Add a colorbar for the entire figure
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
        
        plt.suptitle('Predicted Density - Sample Views', fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plt.savefig(os.path.join(output_dir, 'predicted_density_grid.png'))
        plt.close()
    
    # Create 3x3 grid visualization of ground truth comparison if available
    if perform_evaluation and gt_data_list and predicted_density_list:
       # Sort both lists by view index
        gt_data_list.sort(key=lambda x: x[0])
        predicted_density_list.sort(key=lambda x: x[0])
        
        # Find views that have both predicted density and GT data
        common_views = []
        for view_idx, gt_data in gt_data_list:
            for p_view_idx, pred_density in predicted_density_list:
                if view_idx == p_view_idx:
                    common_views.append((view_idx, pred_density, gt_data))
                    break
    
        # Sample views
        sampled_views = common_views[::max(1, len(common_views) // 9)][:9]
        
        # If we have fewer than 9 views, use all available views
        if len(sampled_views) < 9:
            sampled_views = common_views[:min(9, len(common_views))]
        
        # Create the grid
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        for i, (view_idx, pred, gt_data) in enumerate(sampled_views):
            if i < 9:
                ax = axes[i]
                
                # For the predicted density: mask non-points (0 values) and set them to black
                pred_mask = (pred == 0)
                # Normalize using the same colormap scale (do not perform per-image norming)
                norm_pred = (pred - cmap_min) / (cmap_max - cmap_min)
                norm_gt = (gt_data - cmap_min) / (cmap_max - cmap_min)
                norm_pred = np.clip(norm_pred, 0, 1)
                norm_gt = np.clip(norm_gt, 0, 1)
                
                pred_colors = plt.cm.jet(norm_pred)
                gt_colors = plt.cm.jet(norm_gt)
                # Set non-point areas in the predicted image
                pred_colors[pred_mask] = [1, 1, 1, 1]
                # Set the non-point areas in the gt to the same color as the gt but lower alpha
                gt_colors[pred_mask, :3] = gt_colors[pred_mask, :3]  # Keep RGB values
                gt_colors[pred_mask, 3] = 0.5  # Set alpha to 0.5
                
                # Stack the images vertically: predicted on top, GT on bottom
                combined = np.vstack([pred_colors, gt_colors])
                # Optionally, add a horizontal white dividing line between the two halves:
                line_thickness = 2
                h_pred = pred_colors.shape[0]
                combined[h_pred - line_thickness:h_pred + line_thickness, :] = 1  # White line
                
                ax.imshow(combined)
                ax.set_title(f'View {view_idx}')
                ax.axis('off')
        
        # Hide any unused subplots
        for i in range(len(sampled_views), 9):
            axes[i].axis('off')
        
        plt.suptitle('Comparison: Predicted (Top) vs Ground Truth (Bottom)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'gt_comparison_grid.png'))
        plt.close()
    
    if perform_evaluation and common_views:
        diff_images = []
        for view_idx, pred, gt_data in common_views:
            diff = pred - gt_data
            # Set non-point areas in the predicted image to transparent
            diff[pred == 0] = np.nan
            diff_images.append((view_idx, diff))
        
        # Sample up to 9 views for the grid
        sampled_diffs = diff_images[::max(1, len(diff_images) // 9)][:9]
        if len(sampled_diffs) < 9:
            sampled_diffs = diff_images[:min(9, len(diff_images))]
        
        # Determine a global maximum difference for consistent scaling
        global_max_diff = max([np.max(np.abs(diff)) for _, diff in sampled_diffs])
        if global_max_diff == 0:
            global_max_diff = 1  # Prevent divide by zero
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        for i, (view_idx, diff) in enumerate(sampled_diffs):
            if i < 9:
                ax = axes[i]
                im = ax.imshow(diff, cmap=VisualizationConfig.DENSITY_COLORMAP, vmin=0, vmax=global_max_diff)
                ax.set_title(f'View {view_idx} | Difference (Predicted - Ground Truth kg/m³)')
                ax.axis('off')
        for i in range(len(sampled_diffs), 9):
            axes[i].axis('off')
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Difference (kg/m³)')
        
        plt.suptitle('Difference Grid', fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        plt.savefig(os.path.join(output_dir, 'difference_grid.png'))
        plt.close()

    print(f"Density evaluation complete. Results saved to {output_dir}")

    # Save all metrics
    if all_metrics:
        metrics_file = PathConfig.get_metrics_file(output_dir)
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Calculate and print average metrics across all views
        avg_metrics = {metric: np.mean([m[metric] for m in all_metrics.values()]) 
                      for metric in next(iter(all_metrics.values()))}
        
        print("\nAverage metrics across all views:")
        for metric_name, value in avg_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        avg_metrics_file = PathConfig.get_avg_metrics_file(output_dir)
        with open(avg_metrics_file, 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        
        # Extract metrics for histogram generation
        metrics_by_type = {}
        for metric_type in ['ADE', 'ALDE', 'APE', 'MnRE']:
            metrics_by_type[metric_type] = [metrics[metric_type] for metrics in all_metrics.values()]
        
        # Create histograms for each metric type across all views
        plt.figure(figsize=(15, 10))
        for i, (metric_name, values) in enumerate(metrics_by_type.items(), 1):
            plt.subplot(2, 2, i)
            plt.hist(values, bins=10, alpha=0.7)
            plt.title(f'{metric_name} Distribution Across All Views')
            plt.xlabel(metric_name)
            plt.ylabel('Frequency')
            plt.axvline(np.mean(values), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(values):.4f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_histograms_all_views.png'))
        plt.close()
        
        # If we have multiple scenes, generate histograms for metrics averaged across views per scene
        # For a single scene, this is the same as the average metrics
        if len(args.scene_name.split(',')) > 1:
            # Group metrics by scene
            scene_metrics = {}
            for scene_name in args.scene_name.split(','):
                scene_metrics[scene_name] = {}
                for metric_type in ['ADE', 'ALDE', 'APE', 'MnRE']:
                    scene_views = [v for k, v in all_metrics.items() if scene_name in k]
                    if scene_views:
                        scene_metrics[scene_name][metric_type] = np.mean([v[metric_type] for v in scene_views])
            
            # Create histograms for each metric type across scenes (averaged across views)
            plt.figure(figsize=(15, 10))
            for i, metric_name in enumerate(['ADE', 'ALDE', 'APE', 'MnRE'], 1):
                values = [metrics[metric_name] for metrics in scene_metrics.values() if metric_name in metrics]
                if values:
                    plt.subplot(2, 2, i)
                    plt.bar(list(scene_metrics.keys()), values, alpha=0.7)
                    plt.title(f'Average {metric_name} Across Scenes')
                    plt.xlabel('Scene')
                    plt.ylabel(metric_name)
                    plt.xticks(rotation=45, ha='right')
                    plt.axhline(np.mean(values), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(values):.4f}')
                    plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_histograms_by_scene.png'))
            plt.close()
    
    print(f"Density evaluation complete. Results saved to {output_dir}")


if __name__ == '__main__':
    args = get_args()
    run_density_evaluation(args)
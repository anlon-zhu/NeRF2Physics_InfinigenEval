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
from visualization import render_pcd, values_to_colors


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
    """
    # Create point cloud with colors based on density values
    val_pcd = o3d.geometry.PointCloud()
    val_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Use mean of min-max range for visualization
    mean_density = np.mean(density_values, axis=1)
    colors = values_to_colors(mean_density, cmap_min, cmap_max)
    val_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Render from the camera view
    rendered_img = render_pcd(val_pcd, w2c, K, hw=hw, show=False)
    
    return rendered_img


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


def evaluate_density_against_gt(rendered_density, gt_density, gt_vmin, gt_vmax):
    """
    Compare rendered density image with ground truth density image.
    """
    # Resize ground truth if needed
    if rendered_density.shape[:2] != gt_density.shape[:2]:
        gt_density_resized = np.array(Image.fromarray(gt_density).resize(
            (rendered_density.shape[1], rendered_density.shape[0]), Image.BILINEAR))
    else:
        gt_density_resized = gt_density
    
    # Scale rendered density to match ground truth range
    rendered_density_scaled = np.copy(rendered_density)
    if rendered_density.max() > 0:  # Avoid division by zero
        # Scale rendered density to the same range as ground truth
        rendered_min = rendered_density.min()
        rendered_max = rendered_density.max()
        # Linear mapping from [rendered_min, rendered_max] to [gt_vmin, gt_vmax]
        rendered_density_scaled = ((rendered_density - rendered_min) / (rendered_max - rendered_min)) * (gt_vmax - gt_vmin) + gt_vmin
    
    # Normalize both to [0, 1] for comparison
    if gt_density_resized.max() > gt_density_resized.min():
        gt_normalized = (gt_density_resized - gt_density_resized.min()) / (gt_density_resized.max() - gt_density_resized.min())
    else:
        gt_normalized = np.zeros_like(gt_density_resized)
        
    if rendered_density_scaled.max() > rendered_density_scaled.min():
        rendered_normalized = (rendered_density_scaled - rendered_density_scaled.min()) / (rendered_density_scaled.max() - rendered_density_scaled.min())
    else:
        rendered_normalized = np.zeros_like(rendered_density_scaled)
    
    # Convert to prediction format for metrics
    pixel_values = rendered_normalized.flatten()
    gt_values = gt_normalized.flatten()
    
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
        
        # Render density from this view
        rendered_density = render_density_from_camera_view(
            density_results['points'],
            density_results['density_values'],
            w2c_o3d, K,
            cmap_min=cmap_min,
            cmap_max=cmap_max
        )
        
        # Save rendered density visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(rendered_density)
        plt.colorbar(label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
        plt.title(f'Predicted Density - View {view_idx}')
        plt.savefig(os.path.join(output_dir, f'predicted_density_view_{view_idx}.png'))
        plt.close()
        
        # Compare with ground truth if available
        if perform_evaluation:
            gt_file = os.path.join(gt_density_dir, f'density_{view_idx:03d}.png')
            if not os.path.exists(gt_file):
                gt_file = os.path.join(gt_density_dir, f'density_{view_idx}.png')
            
            if os.path.exists(gt_file):
                gt_data, gt_vmin, gt_vmax = load_ground_truth_density(gt_file)
                
                # Evaluate
                metrics = evaluate_density_against_gt(rendered_density, gt_data, gt_vmin, gt_vmax)
                all_metrics[f'view_{view_idx}'] = metrics
                
                # Save comparison visualization
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(rendered_density)
                plt.title('Predicted Density')
                plt.colorbar(label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
                
                plt.subplot(1, 3, 2)
                plt.imshow(gt_data, cmap='inferno')
                plt.title('Ground Truth Density')
                plt.colorbar(label=f'Density (kg/m³) [{gt_vmin:.1f} - {gt_vmax:.1f}]')
                
                plt.subplot(1, 3, 3)
                # Scale for difference visualization
                norm_rendered = (rendered_density - rendered_density.min()) / (rendered_density.max() - rendered_density.min() + 1e-8)
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

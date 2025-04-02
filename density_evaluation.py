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
    Load a ground truth density image.
    """
    if gt_image_path.endswith('.npy'):
        # Load numpy array
        return np.load(gt_image_path)
    else:
        # Load image and convert to grayscale (assuming density is represented as pixel intensity)
        img = Image.open(gt_image_path).convert('L')
        return np.array(img)


def evaluate_density_against_gt(rendered_density, gt_density):
    """
    Compare rendered density image with ground truth density image.
    """
    # Resize ground truth if needed
    if rendered_density.shape[:2] != gt_density.shape[:2]:
        gt_density_resized = np.array(Image.fromarray(gt_density).resize(
            (rendered_density.shape[1], rendered_density.shape[0]), Image.BILINEAR))
    else:
        gt_density_resized = gt_density
    
    # Normalize both images to same range if needed
    if rendered_density.max() > 1.0 or gt_density_resized.max() > 1.0:
        rendered_density_norm = rendered_density / rendered_density.max()
        gt_density_resized = gt_density_resized / gt_density_resized.max()
    else:
        rendered_density_norm = rendered_density
    
    # Convert to prediction format for metrics (preds should be Nx2, gts should be N)
    # For visualization we can use mean values
    pixel_values = rendered_density_norm.flatten()
    gt_values = gt_density_resized.flatten()
    
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
            cmap_min=args.cmap_min,
            cmap_max=args.cmap_max
        )
        
        # Save rendered density visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(rendered_density)
        plt.colorbar(label='Density (kg/m³)')
        plt.title(f'Predicted Density - View {view_idx}')
        plt.savefig(os.path.join(output_dir, f'predicted_density_view_{view_idx}.png'))
        plt.close()
        
        # Compare with ground truth if available
        if perform_evaluation:
            gt_file = os.path.join(gt_density_dir, f'density_{view_idx:03d}.png')
            if not os.path.exists(gt_file):
                gt_file = os.path.join(gt_density_dir, f'density_{view_idx}.png')
            
            if os.path.exists(gt_file):
                gt_density = load_ground_truth_density(gt_file)
                
                # Evaluate
                metrics = evaluate_density_against_gt(rendered_density, gt_density)
                all_metrics[f'view_{view_idx}'] = metrics
                
                # Save comparison visualization
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(rendered_density)
                plt.title('Predicted Density')
                plt.colorbar(label='Density (kg/m³)')
                
                plt.subplot(1, 3, 2)
                plt.imshow(gt_density, cmap='inferno')
                plt.title('Ground Truth Density')
                plt.colorbar(label='Density (kg/m³)')
                
                plt.subplot(1, 3, 3)
                diff = np.abs(rendered_density - gt_density)
                plt.imshow(diff, cmap='plasma')
                plt.title('Absolute Difference')
                plt.colorbar(label='Difference')
                
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

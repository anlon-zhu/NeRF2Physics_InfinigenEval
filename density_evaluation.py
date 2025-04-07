import os
import json
import torch
import numpy as np
import open_clip
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

# Configuration and dependency imports
from density_config import PathConfig, EvaluationConfig, VisualizationConfig
from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from predict_property import predict_physical_property_query
from utils import load_ns_point_cloud, parse_transforms_json
from arguments import get_args
from evaluation import ADE, ALDE, MedADE, MnRE
from visualization import render_pcd_headless, values_to_colors


##########################################
# 1. Data Retrieval and Prediction
##########################################

def find_gt_file(gt_dir, view_idx):
    """Find ground truth density file for a specific view."""
    return PathConfig.get_gt_density_file(gt_dir, view_idx)


def get_density_predictions(args, scene_dir, clip_model, clip_tokenizer):
    """
    Retrieve density predictions for scene points.
    Uses the point cloud from NS and returns points along with predicted densities.
    """
    pcd_file = PathConfig.get_point_cloud_file(scene_dir)
    dt_file = PathConfig.get_dataparser_transforms_file(scene_dir)
    
    query_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.sample_voxel_size)
    query_pts = torch.Tensor(query_pts).to(args.device)
    
    prediction_info = predict_physical_property_query(
        args, query_pts, scene_dir, clip_model, clip_tokenizer, return_all=True
    )
    
    return {
        'points': query_pts.cpu().numpy(),
        'density_values': prediction_info['query_pred_vals'],
        'query_pred_probs': prediction_info['query_pred_probs'],
        'material_names': prediction_info['mat_names']
    }


def load_ground_truth_density(gt_image_path):
    """
    Load a ground truth density file (expects .npy format).
    Returns the density array and the min/max density values.
    """
    if not gt_image_path.endswith('.npy'):
        raise ValueError(f"Unsupported ground truth file format: {gt_image_path}. Only .npy files are supported.")
    
    density_data = np.load(gt_image_path)
    valid_data = density_data[density_data != -1]
    if valid_data.size > 0:
        vmin, vmax = np.min(valid_data), np.max(valid_data)
    else:
        vmin, vmax = 0, 1
    return density_data, vmin, vmax


##########################################
# 2. Pixel-level Rendering and Metrics
##########################################

def render_density_view(points, density_values, w2c, K, hw=None, cmap_min=None, cmap_max=None):
    """
    Render density values from a camera view.
    Returns:
      - An RGB visualization image.
      - A single-channel image containing actual density values.
    """
    # Use defaults if not specified
    if hw is None:
        hw = EvaluationConfig.DEFAULT_IMAGE_RESOLUTION
    if cmap_min is None:
        cmap_min = EvaluationConfig.DEFAULT_CMAP_MIN
    if cmap_max is None:
        cmap_max = EvaluationConfig.DEFAULT_CMAP_MAX

    h, w = hw
    # Create a point cloud with colors mapped from density values
    val_pcd = o3d.geometry.PointCloud()
    val_pcd.points = o3d.utility.Vector3dVector(points)
    
    mean_density = np.mean(density_values, axis=1)
    colors = values_to_colors(mean_density, cmap_min, cmap_max)
    val_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Render RGB visualization from the camera view
    rendered_rgb = render_pcd_headless(val_pcd, w2c, K, hw=hw)
    
    # Create a density image (initialize to zeros)
    density_image = np.zeros(hw)
    points_np = np.asarray(val_pcd.points)
    
    # Convert world coordinates to camera coordinates
    pts_cam = (w2c[:3, :3] @ points_np.T).T + w2c[:3, 3]
    
    # Project valid points to image coordinates
    pts_2d = np.zeros((len(pts_cam), 2))
    valid_pts = pts_cam[:, 2] > 0
    if np.any(valid_pts):
        pts_2d[valid_pts, 0] = K[0, 0] * pts_cam[valid_pts, 0] / pts_cam[valid_pts, 2] + K[0, 2]
        pts_2d[valid_pts, 1] = K[1, 1] * pts_cam[valid_pts, 1] / pts_cam[valid_pts, 2] + K[1, 2]
    
    pts_2d = np.round(pts_2d).astype(int)
    in_image = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    valid_indices = np.where(valid_pts & in_image)[0]
    
    if valid_indices.size > 0:
        # Sort points by depth (furthest first so that closer points overwrite)
        z_depths = pts_cam[valid_indices, 2]
        sorted_idxs = np.argsort(-z_depths)
        for idx in sorted_idxs:
            point_idx = valid_indices[idx]
            x, y = pts_2d[point_idx]
            density_image[y, x] = mean_density[point_idx]
    
    return rendered_rgb, density_image


def evaluate_density(rendered_density, rendered_rgb, gt_density, gt_vmin, gt_vmax, is_gt_npy=False):
    """
    Compare a rendered density image with a ground truth density image.
    Computes metrics only if ground truth is provided as a single-channel NPY file.
    """
    print(f"Rendered density shape: {rendered_density.shape}, Rendered RGB shape: {rendered_rgb.shape}, GT density shape: {gt_density.shape}")
    
    if not is_gt_npy:
        print("WARNING: Skipping evaluation - only single-channel density values (NPY/NPZ) are used for evaluation")
        return {'ADE': 0, 'ALDE': 0, 'MedADE': 0, 'MnRE': 0}
    
    if rendered_density.shape != gt_density.shape:
        print(f"WARNING: Format mismatch - Rendered shape: {rendered_density.shape} vs GT shape: {gt_density.shape}")
        return {'ADE': 0, 'ALDE': 0, 'MedADE': 0, 'MnRE': 0}
    
    # Flatten and filter valid pixels (exclude zeros and -1 in GT)
    pixel_values = rendered_density.flatten()
    gt_values = gt_density.flatten()
    valid_gt_mask = gt_values > 0
    if np.any(valid_gt_mask):
        print(f"DEBUG: Valid GT rendered values range: [{np.min(pixel_values[valid_gt_mask]):.4f}, {np.max(pixel_values[valid_gt_mask]):.4f}]")
        print(f"DEBUG: Number of valid GT pixels: {np.sum(valid_gt_mask)}")
    
    valid_mask = valid_gt_mask & (pixel_values > 0)
    valid_pred = pixel_values[valid_mask]
    valid_gt = gt_values[valid_mask]
    print(f"Valid pixels for evaluation: {len(valid_pred)} out of {len(pixel_values)} total pixels ({len(valid_pred)/len(pixel_values)*100:.2f}%)")
    
    if len(valid_pred) < 10:
        print("WARNING: Not enough valid pixels for meaningful evaluation")
        return {'ADE': 0, 'ALDE': 0, 'MedADE': 0, 'MnRE': 0}
    
    epsilon = 1e-8
    safe_pred = np.maximum(valid_pred, epsilon)
    safe_gt = np.maximum(valid_gt, epsilon)
    
    try:
        metrics = {
            'ADE': ADE(safe_pred, safe_gt),
            'ALDE': ALDE(safe_pred, safe_gt),
            'MedADE': MedADE(safe_pred, safe_gt),
            'MnRE': MnRE(safe_pred, safe_gt)
        }
    except Exception as e:
        print(f"WARNING: Error computing metrics: {e}")
        return {'ADE': 0, 'ALDE': 0, 'MedADE': 0, 'MnRE': 0}
    
    return metrics


##########################################
# 3. Visualization Functions
##########################################

def save_density_map(rendered_density, view_idx, cmap_min, cmap_max, output_dir):
    """Save an image of the predicted density values using a colormap."""
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_density, cmap=VisualizationConfig.DENSITY_COLORMAP)
    plt.colorbar(label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
    plt.title(f'Predicted Density Values - View {view_idx}')
    save_path = PathConfig.get_predicted_density_map_file(output_dir, view_idx, 'png')
    plt.savefig(save_path)
    plt.close()


def save_density_data(rendered_density, rendered_rgb, cmap_min, cmap_max, view_idx, output_dir):
    """Save density data for further evaluation (NPY and NPZ formats)."""
    np.save(PathConfig.get_predicted_density_map_file(output_dir, view_idx, 'npy'), rendered_density)
    np.savez(PathConfig.get_predicted_density_map_file(output_dir, view_idx, 'npz'),
             density=rendered_density,
             rgb=rendered_rgb,
             min_value=cmap_min,
             max_value=cmap_max)


def create_grid_visualization(rendered_data, output_dir, cmap_min, cmap_max):
    """
    Create a 3x3 grid of predicted density maps for sample views.
    rendered_data is a list of tuples: (view_idx, rgb, nonzero_density)
    """
    rendered_data.sort(key=lambda x: x[0])
    sampled_views = rendered_data[::max(1, len(rendered_data) // 9)][:9]
    if len(sampled_views) < 9:
        sampled_views = rendered_data[:min(9, len(rendered_data))]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i, (view_idx, _, nonzero_density) in enumerate(sampled_views):
        if i < 9:
            ax = axes[i]
            im = ax.imshow(nonzero_density, cmap='jet', interpolation='none')
            ax.set_title(f'View {view_idx}')
            ax.axis('off')
    for i in range(len(sampled_views), 9):
        axes[i].axis('off')
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=f'Density (kg/m³) [{cmap_min:.1f} - {cmap_max:.1f}]')
    
    plt.suptitle('Predicted Density - Sample Views', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(os.path.join(output_dir, 'predicted_density_grid.png'))
    plt.close()


def create_comparisons(gt_data_list, predicted_density_list):
    """
    Create a grid comparing predicted (top) and ground truth (bottom) density images.
    Returns a list of common views as tuples: (view_idx, predicted, gt)
    """
    # Create dictionaries for quick lookup
    gt_dict = {view_idx: gt for view_idx, gt in gt_data_list}
    pred_dict = {view_idx: pred for view_idx, pred in predicted_density_list}
    common_views = [(view_idx, pred_dict[view_idx], gt_dict[view_idx])
                    for view_idx in sorted(set(gt_dict.keys()) & set(pred_dict.keys()))] 
    return common_views

def create_contextual_difference_grid(common_views, output_dir):
    """
    3x3 grid showing log(|pred - GT|) overlayed on GT.
    Helps visually amplify small differences without being dominated by outliers.
    """
    diff_images = []
    for view_idx, pred, gt_data in common_views:
        diff = np.log(1 + pred) - np.log(1 + gt_data)
        diff[pred == 0] = np.nan
        diff_images.append((view_idx, diff, gt_data))

    sampled_diffs = diff_images[::max(1, len(diff_images) // 9)][:9]
    if len(sampled_diffs) < 9:
        sampled_diffs = diff_images[:min(9, len(diff_images))]

    global_max_log = max(np.nanmax(diff) for _, diff, _ in sampled_diffs)
    norm = plt.Normalize(vmin=0, vmax=global_max_log)
    cmap = plt.cm.get_cmap('turbo')

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, ((view_idx, _, gt), log_diff) in enumerate(zip(sampled_diffs, log_diffs)):
        ax = axes[i]
        ax.set_title(f"View {view_idx} | log(Pred) -log(GT)")
        ax.axis('off')

        # GT background
        ax.imshow(gt, cmap='gray', alpha=0.2)

        # Overlay log difference heatmap
        im = ax.imshow(log_diff, cmap=cmap, norm=norm, alpha=1.0)

    for i in range(len(sampled_diffs), 9):
        axes[i].axis('off')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="log₁₀(|Difference| + 1)")

    plt.suptitle("Log Difference Overlayed on GT", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(os.path.join(output_dir, "contextual_difference_grid.png"))
    plt.close()

def plot_metrics_histograms(all_metrics, output_dir):
    """
    Plot histograms of each metric (ADE, ALDE, MedADE, MnRE) across all views.
    """
    metrics_by_type = {metric: [m[metric] for m in all_metrics.values()]
                       for metric in ['ADE', 'ALDE', 'MedADE', 'MnRE']}
    
    plt.figure(figsize=(15, 10))
    for i, (metric_name, values) in enumerate(metrics_by_type.items(), 1):
        plt.subplot(2, 2, i)
        plt.hist(values, bins=10, alpha=0.7)
        plt.title(f'{metric_name} Distribution Across All Views')
        plt.xlabel(metric_name)
        plt.ylabel('Frequency')
        if metric_name != 'MedADE':
            plt.axvline(np.mean(values), color='r', linestyle='dashed', linewidth=1,
                        label=f'Mean: {np.mean(values):.4f}')
        else:
            plt.axvline(np.median(values), color='r', linestyle='dashed', linewidth=1,
                        label=f'Median: {np.median(values):.4f}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_histograms_all_views.png'))
    plt.close()


def plot_scene_metrics_histograms(args, all_metrics, output_dir):
    """
    If multiple scenes are evaluated, plot histograms for scene-level (averaged) metrics.
    """
    if len(args.scene_name.split(',')) <= 1:
        return

    scene_metrics = {}
    for scene_name in args.scene_name.split(','):
        scene_metrics[scene_name] = {}
        for metric_type in ['ADE', 'ALDE', 'MedADE', 'MnRE']:
            scene_views = [v for k, v in all_metrics.items() if scene_name in k]
            if scene_views:
                scene_metrics[scene_name][metric_type] = np.mean([v[metric_type] for v in scene_views])
    
    plt.figure(figsize=(15, 10))
    for i, metric_name in enumerate(['ADE', 'ALDE', 'MedADE', 'MnRE'], 1):
        values = [metrics[metric_name] for metrics in scene_metrics.values() if metric_name in metrics]
        if values:
            plt.subplot(2, 2, i)
            plt.bar(list(scene_metrics.keys()), values, alpha=0.7)
            plt.title(f'Average {metric_name} Across Scenes')
            plt.xlabel('Scene')
            plt.ylabel(metric_name)
            plt.xticks(rotation=45, ha='right')
            plt.axhline(np.mean(values), color='r', linestyle='dashed', linewidth=1,
                        label=f'Mean: {np.mean(values):.4f}')
            plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_histograms_by_scene.png'))
    plt.close()


##########################################
# 4. Main Evaluation Workflow
##########################################

def run_density_evaluation(args):
    # Set up paths and output directory
    scenes_dir = PathConfig.get_scenes_dir(args.data_dir)
    scene_dir = PathConfig.get_scene_dir(args.data_dir, args.scene_name)
    gt_density_dir = PathConfig.get_gt_density_dir(scene_dir)
    t_file = PathConfig.get_transforms_file(scene_dir)
    output_dir = PathConfig.get_evaluation_output_dir(scene_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLIP model and tokenizer
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)
    
    print(f"Predicting densities for scene: {args.scene_name}")
    density_results = get_density_predictions(args, scene_dir, clip_model, clip_tokenizer)
    
    # Check ground truth availability
    if os.path.exists(gt_density_dir):
        gt_files = [f for f in os.listdir(gt_density_dir) if f.endswith('.png') or f.endswith('.npy')]
        perform_evaluation = bool(gt_files)
    else:
        print(f"Warning: Ground truth density directory not found: {gt_density_dir}")
        perform_evaluation = False
        gt_files = []
    
    w2cs, K = parse_transforms_json(t_file, return_w2c=True)
    
    # Determine colormap range from ground truth if available
    cmap_min, cmap_max = args.cmap_min, args.cmap_max
    if perform_evaluation:
        for f in gt_files:
            if f.startswith('density_'):
                gt_file = os.path.join(gt_density_dir, f)
                try:
                    _, gt_vmin, gt_vmax = load_ground_truth_density(gt_file)
                    print(f"Using density range from ground truth: {gt_vmin:.2f} to {gt_vmax:.2f} kg/m³")
                    cmap_min, cmap_max = gt_vmin, gt_vmax
                    break
                except Exception as e:
                    print(f"Could not extract range from ground truth: {e}")
    
    all_metrics = {}
    rendered_data = []         # For grid visualization (non-zero density points)
    gt_data_list = []          # (view_idx, ground truth image)
    predicted_density_list = []  # (view_idx, predicted density image)
    
    for view_idx, w2c in enumerate(w2cs):
        print(f"Processing view {view_idx}")
        # Convert camera transform format for Open3D
        w2c_o3d = w2c.copy()
        w2c_o3d[[1, 2]] *= -1
        
        gt_file = find_gt_file(gt_density_dir, view_idx) if perform_evaluation else None
        hw = (1024, 1024)
        if gt_file and os.path.exists(gt_file):
            gt_data, _, _ = load_ground_truth_density(gt_file)
            hw = gt_data.shape[:2]
            print(f"Using ground truth dimensions for rendering: {hw}")
        
        rendered_rgb, rendered_density = render_density_view(
            density_results['points'],
            density_results['density_values'],
            w2c_o3d, K,
            hw=hw,
            cmap_min=cmap_min,
            cmap_max=cmap_max
        )
        
        # Save rendered images and data
        save_density_map(rendered_density, view_idx, cmap_min, cmap_max, output_dir)
        save_density_data(rendered_density, rendered_rgb, cmap_min, cmap_max, view_idx, output_dir)
        
        # For enhanced grid visualization, use non-zero pixels only
        nonzero_mask = rendered_density > 0
        if np.any(nonzero_mask):
            nonzero_density = np.copy(rendered_density)
            nonzero_density[~nonzero_mask] = np.nan
            rendered_data.append((view_idx, rendered_rgb, nonzero_density))
        
        # If ground truth exists, use it for evaluation
        if gt_file and os.path.exists(gt_file):
            predicted_density_list.append((view_idx, rendered_density))
            gt_data, gt_vmin, gt_vmax = load_ground_truth_density(gt_file)
            gt_data_list.append((view_idx, gt_data))
            is_gt_npy = gt_file.endswith('.npy')
            metrics = evaluate_density(rendered_density, rendered_rgb, gt_data, gt_vmin, gt_vmax, is_gt_npy=is_gt_npy)
            all_metrics[f'view_{view_idx}'] = metrics
            print(f"Metrics for view {view_idx}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
    
    # Create visualizations
    if rendered_data:
        create_grid_visualization(rendered_data, output_dir, cmap_min, cmap_max)
    
    common_views = create_comparisons(gt_data_list, predicted_density_list)
    if perform_evaluation and common_views:
        create_contextual_difference_grid(common_views, output_dir)
    
    if all_metrics:
        metrics_file = PathConfig.get_metrics_file(output_dir)
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        avg_metrics = {metric: np.mean([m[metric] for m in all_metrics.values()])
                       for metric in next(iter(all_metrics.values()))}
        print("\nAverage metrics across all views:")
        for metric_name, value in avg_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        avg_metrics_file = PathConfig.get_avg_metrics_file(output_dir)
        with open(avg_metrics_file, 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        
        plot_metrics_histograms(all_metrics, output_dir)
        plot_scene_metrics_histograms(args, all_metrics, output_dir)
    
    print(f"Density evaluation complete. Results saved to {output_dir}")


if __name__ == '__main__':
    args = get_args()
    run_density_evaluation(args)

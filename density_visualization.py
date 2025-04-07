"""
Density Visualization for NeRF2Physics
Creates the requested plots and tables for the density evaluation results
across multiple scenes and views.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob
from matplotlib.colors import Normalize
from argparse import ArgumentParser
from tqdm import tqdm

# Import our configuration
from density_config import PathConfig, VisualizationConfig


def load_metrics_for_scene(scene_dir):
    """Load metrics for a scene from its density_metrics_avg.json file."""
    output_dir = PathConfig.get_evaluation_output_dir(scene_dir)
    metrics_path = PathConfig.get_avg_metrics_file(output_dir)
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def load_view_metrics_for_scene(scene_dir):
    """Load per-view metrics for a scene from its density_metrics.json file."""
    output_dir = PathConfig.get_evaluation_output_dir(scene_dir)
    metrics_path = PathConfig.get_metrics_file(output_dir)
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def load_density_map(scene_dir, view_idx=0):
    """Load a density map for a specific view of a scene."""
    output_dir = PathConfig.get_evaluation_output_dir(scene_dir)
    density_path = PathConfig.get_predicted_density_map_file(output_dir, view_idx, 'npy')
    if not os.path.exists(density_path):
        return None
    
    density_map = np.load(density_path)
    return density_map


def load_gt_density_map(scene_dir, view_idx=0):
    """Load a ground truth density map for a specific view of a scene."""
    gt_density_dir = PathConfig.get_gt_density_dir(scene_dir)
    if not os.path.exists(gt_density_dir):
        return None
    
    # Try to find the ground truth file using PathConfig
    gt_file = PathConfig.get_gt_density_file(gt_density_dir, view_idx)
    if gt_file and os.path.exists(gt_file) and gt_file.endswith('.npy'):
        gt_data = np.load(gt_file)
        return gt_data
    
    return None


def create_aggregate_metrics_table(scene_metrics):
    """
    Create Table 1: Aggregate Evaluation Metrics Summary
    """
    # Extract metrics into arrays
    ade_values = [m.get('ADE', 0) for m in scene_metrics.values() if m is not None]
    alde_values = [m.get('ALDE', 0) for m in scene_metrics.values() if m is not None]
    ape_values = [m.get('APE', 0) for m in scene_metrics.values() if m is not None]
    mnre_values = [m.get('MnRE', 0) for m in scene_metrics.values() if m is not None]
    
    # Create DataFrame for the table
    data = {
        'Metric': ['ADE', 'ALDE', 'APE', 'MnRE'],
        'Mean (All Scenes)': [
            np.mean(ade_values),
            np.mean(alde_values),
            np.mean(ape_values),
            np.mean(mnre_values)
        ],
        'Std. Dev. (All Scenes)': [
            np.std(ade_values),
            np.std(alde_values),
            np.std(ape_values),
            np.std(mnre_values)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Format the table for display
    df['Mean (All Scenes)'] = df['Mean (All Scenes)'].map(lambda x: f"{x:.2f}")
    df['Std. Dev. (All Scenes)'] = df['Std. Dev. (All Scenes)'].map(lambda x: f"± {x:.2f}")
    
    # Special handling for APE (percentage)
    if 'APE' in df['Metric'].values:
        idx = df.index[df['Metric'] == 'APE'].tolist()[0]
        df.at[idx, 'Mean (All Scenes)'] = f"{float(df.at[idx, 'Mean (All Scenes)']):.2f}%"
    
    # Save as CSV
    df.to_csv('metrics_summary_table.csv', index=False)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    with open('metrics_summary_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"Table 1 created and saved as metrics_summary_table.csv and metrics_summary_table.tex")
    
    return df


def create_metrics_distribution_plot(scene_metrics):
    """
    Create Figure 1: Distribution of Evaluation Metrics Across Scenes
    """
    # Convert metrics dictionary to DataFrame suitable for seaborn
    metrics_data = []
    for scene_id, metrics in scene_metrics.items():
        if metrics is None:
            continue
        
        for metric_name, value in metrics.items():
            metrics_data.append({
                'Scene': scene_id,
                'Metric': metric_name,
                'Value': value
            })
    
    df = pd.DataFrame(metrics_data)
    
    # Create violin plot with box plot inside
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Metric', y='Value', data=df, inner='box', palette='viridis')
    
    # Customize plot
    plt.title('Distribution of Evaluation Metrics Across All Scenes', fontsize=16)
    plt.xlabel('Metric Type', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(VisualizationConfig.METRICS_DIST_FILENAME, dpi=300)
    plt.close()
    
    print(f"Figure 1 created and saved as {VisualizationConfig.METRICS_DIST_FILENAME}")


def create_scenewise_ade_plot(scene_metrics):
    """
    Create Figure 2: Scene-wise ADE (Sorted)
    """
    # Extract ADE values with scene IDs
    ade_values = []
    for scene_id, metrics in scene_metrics.items():
        if metrics is not None and 'ADE' in metrics:
            scene_num = os.path.basename(scene_id).replace('scene_', '')
            ade_values.append((scene_num, metrics['ADE']))
    
    # Sort by ADE value
    sorted_ade = sorted(ade_values, key=lambda x: x[1])
    scene_ids = [x[0] for x in sorted_ade]
    ade_vals = [x[1] for x in sorted_ade]
    
    # Create bar plot
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(len(ade_vals)), ade_vals, color=plt.cm.viridis(np.linspace(0, 1, len(ade_vals))))
    
    # Customize plot
    plt.title('Scene-wise Average Density Error (ADE) - Sorted', fontsize=16)
    plt.xlabel('Scene ID (sorted by ADE)', fontsize=14)
    plt.ylabel('ADE', fontsize=14)
    plt.xticks(range(0, len(scene_ids), max(1, len(scene_ids)//10)), 
               [scene_ids[i] for i in range(0, len(scene_ids), max(1, len(scene_ids)//10))], 
               rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(VisualizationConfig.SCENEWISE_ADE_FILENAME, dpi=300)
    plt.close()
    
    print(f"Figure 2 created and saved as {VisualizationConfig.SCENEWISE_ADE_FILENAME}")


def create_grid_diff_density(scene_dirs, view_idx=0, grid_size=None):
    # Use default grid size from config if not specified
    if grid_size is None:
        grid_size = VisualizationConfig.GRID_SIZE
    """
    Create Figure 3: 10x10 Grid - Pixel-Level Density Difference
    """
    rows, cols = grid_size
    max_scenes = rows * cols
    valid_scenes = []
    
    # First pass to identify valid scenes and determine global colormap scale
    diff_maps = []
    print("Finding valid scenes for difference grid...")
    for scene_dir in tqdm(scene_dirs[:max_scenes*2]):  # Check more scenes than needed to ensure we get enough valid ones
        if len(valid_scenes) >= max_scenes:
            break
            
        pred_density = load_density_map(scene_dir, view_idx)
        gt_density = load_gt_density_map(scene_dir, view_idx)
        
        if pred_density is not None and gt_density is not None:
            # Make sure shapes match
            if pred_density.shape == gt_density.shape:
                # Calculate absolute difference
                abs_diff = np.abs(pred_density - gt_density)
                
                # Set -1 values in GT (missing data) to 0 in diff
                if np.any(gt_density == -1):
                    abs_diff[gt_density == -1] = 0
                
                diff_maps.append(abs_diff)
                valid_scenes.append(scene_dir)
    
    if len(valid_scenes) < max_scenes:
        print(f"Warning: Only {len(valid_scenes)} valid scenes found for grid visualization.")
    
    # Determine global min/max for consistent colormap
    all_diffs = np.concatenate([diff.flatten() for diff in diff_maps])
    all_diffs = all_diffs[all_diffs > 0]  # Exclude zeros for better scaling
    if len(all_diffs) > 0:
        vmin = np.percentile(all_diffs, 5)  # Use 5th percentile to avoid outliers
        vmax = np.percentile(all_diffs, 95)  # Use 95th percentile to avoid outliers
    else:
        vmin, vmax = 0, 1
    
    # Create the grid visualization
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()
    
    print("Creating difference grid visualization...")
    for i, (scene_dir, diff_map) in enumerate(zip(valid_scenes[:max_scenes], diff_maps[:max_scenes])):
        scene_id = os.path.basename(scene_dir)
        ax = axes[i]
        
        # Display the difference map
        im = ax.imshow(diff_map, cmap=VisualizationConfig.DIFF_COLORMAP, vmin=vmin, vmax=vmax)
        ax.set_title(scene_id, fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(valid_scenes), max_scenes):
        axes[i].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Absolute Difference in Density (kg/m³)')
    
    # Set title for the entire figure
    plt.suptitle(f'Absolute Density Difference (|Prediction - Ground Truth|) - View {view_idx}', 
                 fontsize=16, y=0.98)
    
    # Save figure
    plt.savefig(f'grid_diff_density_view{view_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure 3 created and saved as grid_diff_density_view{view_idx}.png")


def create_grid_pred_density(scene_dirs, view_idx=0, grid_size=(10, 10)):
    """
    Create Figure 4: 10x10 Grid - Raw Predicted Densities
    """
    rows, cols = grid_size
    max_scenes = rows * cols
    valid_scenes = []
    
    # First pass to identify valid scenes and determine global colormap scale
    density_maps = []
    print("Finding valid scenes for predicted density grid...")
    for scene_dir in tqdm(scene_dirs[:max_scenes*2]):  # Check more scenes than needed to ensure we get enough valid ones
        if len(valid_scenes) >= max_scenes:
            break
            
        pred_density = load_density_map(scene_dir, view_idx)
        
        if pred_density is not None:
            density_maps.append(pred_density)
            valid_scenes.append(scene_dir)
    
    if len(valid_scenes) < max_scenes:
        print(f"Warning: Only {len(valid_scenes)} valid scenes found for grid visualization.")
    
    # Determine global min/max for consistent colormap
    all_densities = np.concatenate([density.flatten() for density in density_maps])
    all_densities = all_densities[all_densities > 0]  # Exclude zeros for better scaling
    if len(all_densities) > 0:
        vmin = 0  # Always start at 0 for density
        vmax = np.percentile(all_densities, 95)  # Use 95th percentile to avoid outliers
    else:
        vmin, vmax = 0, 3000  # Default range
    
    # Create the grid visualization
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()
    
    print("Creating predicted density grid visualization...")
    for i, (scene_dir, density_map) in enumerate(zip(valid_scenes[:max_scenes], density_maps[:max_scenes])):
        scene_id = os.path.basename(scene_dir)
        ax = axes[i]
        
        # Display the density map
        im = ax.imshow(density_map, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(scene_id, fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(valid_scenes), max_scenes):
        axes[i].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Predicted Density (kg/m³)')
    
    # Set title for the entire figure
    plt.suptitle(f'Predicted Density Maps - View {view_idx}', fontsize=16, y=0.98)
    
    # Save figure
    plt.savefig(f'grid_pred_density_view{view_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure 4 created and saved as grid_pred_density_view{view_idx}.png")


def create_grid_valid_mask(scene_dirs, view_idx=0, grid_size=(10, 10)):
    """
    Create Figure 5: 10x10 Grid - Valid Prediction Masks
    """
    rows, cols = grid_size
    max_scenes = rows * cols
    valid_scenes = []
    
    # First pass to identify valid scenes
    mask_maps = []
    print("Finding valid scenes for prediction mask grid...")
    for scene_dir in tqdm(scene_dirs[:max_scenes*2]):  # Check more scenes than needed to ensure we get enough valid ones
        if len(valid_scenes) >= max_scenes:
            break
            
        pred_density = load_density_map(scene_dir, view_idx)
        
        if pred_density is not None:
            # Create binary mask where prediction exists
            mask = (pred_density > 0).astype(np.float32)
            mask_maps.append(mask)
            valid_scenes.append(scene_dir)
    
    if len(valid_scenes) < max_scenes:
        print(f"Warning: Only {len(valid_scenes)} valid scenes found for grid visualization.")
    
    # Create the grid visualization
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()
    
    print("Creating valid prediction mask grid visualization...")
    for i, (scene_dir, mask) in enumerate(zip(valid_scenes[:max_scenes], mask_maps[:max_scenes])):
        scene_id = os.path.basename(scene_dir)
        ax = axes[i]
        
        # Display the binary mask (white = valid prediction, black = no prediction)
        im = ax.imshow(mask, cmap='binary', vmin=0, vmax=1)
        ax.set_title(scene_id, fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(valid_scenes), max_scenes):
        axes[i].axis('off')
    
    # Set title for the entire figure
    plt.suptitle(f'Valid Prediction Masks - View {view_idx}', fontsize=16, y=0.98)
    
    # Save figure
    plt.savefig(f'grid_valid_mask_view{view_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure 5 created and saved as grid_valid_mask_view{view_idx}.png")


def create_multiview_scene_analysis(scene_dir, views=None):
    """
    Create Figure 6: Case Study - Multi-View Scene Analysis
    Compare predicted and ground truth density maps over multiple views for one selected scene.
    """
    if views is None:
        views = list(range(0, 30, 3))[:9]  # Views 0, 3, 6, ..., 27 (9 total)
    
    scene_id = os.path.basename(scene_dir)
    valid_views = []
    pred_maps = []
    gt_maps = []
    
    # Find views that have both prediction and ground truth
    for view_idx in views:
        pred_density = load_density_map(scene_dir, view_idx)
        gt_density = load_gt_density_map(scene_dir, view_idx)
        
        if pred_density is not None and gt_density is not None:
            # Make sure shapes match
            if pred_density.shape == gt_density.shape:
                valid_views.append(view_idx)
                pred_maps.append(pred_density)
                gt_maps.append(gt_density)
    
    if not valid_views:
        print(f"Warning: No valid views found for multi-view analysis of {scene_id}")
        return
    
    # Determine global colormap scale for consistent visualization
    all_densities = np.concatenate([pred.flatten() for pred in pred_maps] + [gt.flatten() for gt in gt_maps])
    valid_densities = all_densities[(all_densities > 0) & (all_densities != -1)]  # Exclude zeros and -1 values
    if len(valid_densities) > 0:
        vmin = 0  # Always start at 0 for density
        vmax = np.percentile(valid_densities, 95)  # Use 95th percentile to avoid outliers
    else:
        vmin, vmax = 0, 3000  # Default range
    
    # Create a grid of subplots for the comparison
    n_views = len(valid_views)
    if n_views == 0:
        print(f"Error: No valid views available for {scene_id}")
        return
    
    # Create a figure with side-by-side comparisons for each view
    fig, axes = plt.subplots(n_views, 2, figsize=(12, 3*n_views))
    if n_views == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array for consistent indexing
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    for i, (view_idx, pred, gt) in enumerate(zip(valid_views, pred_maps, gt_maps)):
        # Left: predicted density
        im_pred = axes[i, 0].imshow(pred, cmap='viridis', norm=norm)
        axes[i, 0].set_title(f'View {view_idx} - Predicted')
        axes[i, 0].axis('off')
        
        # Right: ground truth
        invalid_mask = gt == -1
        gt_display = gt.copy()
        # Replace -1 values with NaN for display (will appear transparent)
        if np.any(invalid_mask):
            gt_display[invalid_mask] = np.nan
        
        im_gt = axes[i, 1].imshow(gt_display, cmap='viridis', norm=norm)
        axes[i, 1].set_title(f'View {view_idx} - Ground Truth')
        axes[i, 1].axis('off')
    
    # Add a colorbar for the entire figure
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_pred, cax=cbar_ax)
    cbar.set_label('Density (kg/m³)')
    
    # Set a title for the entire figure
    plt.suptitle(f'Multi-View Comparison: {scene_id}', fontsize=16, y=0.98)
    
    # Save figure
    plt.savefig(f'{scene_id}_multiview_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure 6 created and saved as {scene_id}_multiview_comparison.png")


def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="Generate density evaluation plots and tables")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory containing scene folders')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save output plots and tables')
    parser.add_argument('--case_study_scene', type=str, default='scene_42',
                        help='Scene to use for the multi-view case study')
    parser.add_argument('--view_idx', type=int, default=0,
                        help='View index to use for grid visualizations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)  # Change to output directory
    
    # Get all scene directories
    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scene_dirs = glob(os.path.join(scenes_dir, 'scene_*'))
    scene_dirs.sort()
    
    print(f"Found {len(scene_dirs)} scene directories")
    
    # Load metrics for all scenes
    scene_metrics = {}
    print("Loading scene metrics...")
    for scene_dir in tqdm(scene_dirs):
        scene_id = os.path.basename(scene_dir)
        metrics = load_metrics_for_scene(scene_dir)
        scene_metrics[scene_dir] = metrics
    
    # Create Table 1: Aggregate Evaluation Metrics Summary
    create_aggregate_metrics_table(scene_metrics)
    
    # Create Figure 1: Distribution of Evaluation Metrics Across Scenes
    create_metrics_distribution_plot(scene_metrics)
    
    # Create Figure 2: Scene-wise ADE (Sorted)
    create_scenewise_ade_plot(scene_metrics)
    
    # Create Figure 3: 10x10 Grid – Pixel-Level Density Difference
    create_grid_diff_density(scene_dirs, view_idx=args.view_idx)
    
    # Create Figure 4: 10x10 Grid – Raw Predicted Densities
    create_grid_pred_density(scene_dirs, view_idx=args.view_idx)
    
    # Create Figure 5: 10x10 Grid – Valid Prediction Masks
    create_grid_valid_mask(scene_dirs, view_idx=args.view_idx)
    
    # Create Figure 6: Case Study – Multi-View Scene Analysis
    # Find case study scene directory
    case_study_dir = None
    for scene_dir in scene_dirs:
        if os.path.basename(scene_dir) == args.case_study_scene:
            case_study_dir = scene_dir
            break
    
    if case_study_dir:
        create_multiview_scene_analysis(case_study_dir)
    else:
        print(f"Warning: Case study scene {args.case_study_scene} not found.")
        # Use the first scene that has both predictions and ground truth
        for scene_dir in scene_dirs:
            if (load_density_map(scene_dir, 0) is not None and 
                load_gt_density_map(scene_dir, 0) is not None):
                create_multiview_scene_analysis(scene_dir)
                break
    
    print("All plots and tables generated successfully!")


if __name__ == '__main__':
    main()

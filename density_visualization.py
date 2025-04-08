import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob
from matplotlib.colors import Normalize, ListedColormap
from argparse import ArgumentParser
from tqdm import tqdm

# Configuration
from density_config import PathConfig, VisualizationConfig

##########################################
# Utility Functions
##########################################

def load_metrics(scene_dir, averaged=True):
    output_dir = PathConfig.get_evaluation_output_dir(scene_dir)
    metrics_path = (PathConfig.get_avg_metrics_file(output_dir)
                    if averaged else PathConfig.get_metrics_file(output_dir))
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def load_density_map(scene_dir, view_idx, ground_truth=False):
    if ground_truth:
        gt_dir = PathConfig.get_gt_density_dir(scene_dir)
        gt_file = PathConfig.get_gt_density_file(gt_dir, view_idx)
        if gt_file and os.path.exists(gt_file) and gt_file.endswith('.npy'):
            return np.load(gt_file)
    else:
        output_dir = PathConfig.get_evaluation_output_dir(scene_dir)
        pred_file = PathConfig.get_predicted_density_map_file(output_dir, view_idx, 'npy')
        if os.path.exists(pred_file):
            return np.load(pred_file)
    return None

def compute_global_range(maps, low=5, high=95):
    values = np.concatenate([m.flatten() for m in maps])
    values = values[(values > 0) & (values != -1)]
    if values.size > 0:
        return np.percentile(values, low), np.percentile(values, high)
    return 0, 1

##########################################
# Visualization Functions
##########################################

def create_metrics_violinplot(scene_metrics):
    """Create two violin plots: [ADE + MedADE] and [ALDE + MnRE]"""
    rows = []
    for sid, metrics in scene_metrics.items():
        if metrics:
            for k, v in metrics.items():
                rows.append({'Scene': sid, 'Metric': k, 'Value': v})
    df = pd.DataFrame(rows)

    # Split violin plots
    group1 = ['ADE', 'MedADE']
    group2 = ['ALDE', 'MnRE']

    for group, filename in zip([group1, group2],
                                ['metrics_dist_violin1.png', 'metrics_dist_violin2.png']):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Metric', y='Value', data=df[df['Metric'].isin(group)],
                       inner='box', palette='Set2')
        plt.title('Distribution of Evaluation Metrics')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[INFO] Violin plot saved as {filename}")

def create_grid_image(scene_dirs, view_idx, mode, cmap, filename, label, grid_size):
    rows, cols = grid_size
    max_items = rows * cols
    images, labels = [], []

    for d in scene_dirs:
        pred = load_density_map(d, view_idx)
        gt = load_density_map(d, view_idx, ground_truth=True)
        if pred is not None:
            if mode == 'pred':
                images.append(pred)
                labels.append(os.path.basename(d))
            elif mode == 'diff' and gt is not None and pred.shape == gt.shape:
                diff = np.abs(pred - gt)
                diff[pred == 0] = np.nan
                images.append((diff, gt))
                labels.append(os.path.basename(d))
            elif mode == 'mask':
                mask = (pred > 0).astype(np.float32)
                images.append(mask)
                labels.append(os.path.basename(d))
        if len(images) >= max_items:
            break

    vmin, vmax = compute_global_range(images)

    # Set up figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for ax, img, title in zip(axes.flatten(), images, labels):
        if mode == 'mask':
            im = ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='none')
        elif mode == 'pred':
            # Custom colormap: white for 0
            base_cmap = plt.get_cmap('jet')
            cmap_data = base_cmap(np.linspace(0, 1, 256))
            cmap_data[0] = [1, 1, 1, 1]  # white for 0
            custom_cmap = ListedColormap(cmap_data)
            im = ax.imshow(img, cmap=custom_cmap, vmin=vmin, vmax=vmax, interpolation='none')
        elif mode == 'diff':
            diff, gt = img
            im = ax.imshow(gt, cmap='gray', alpha=1.0)
            # add a dark overlay
            im = ax.imshow(np.zeros_like(gt), cmap='gray', alpha=0.8, interpolation='none')
            im = ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        ax.set_title(title, fontsize=6)
        ax.axis('off')

    # Turn off remaining unused axes
    for ax in axes.flatten()[len(images):]:
        ax.axis('off')

    # Adjust spacing
    fig.subplots_adjust(wspace=0.05, hspace=0.01, right=0.9)

    # Colorbar (skip for mask)
    if mode != 'mask':
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=label)

    plt.suptitle(filename.replace('.png', '').replace('_', ' ').title())
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Grid saved as {filename}")

##########################################
# Main
##########################################

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--view_idx', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)

    scenes_dir = PathConfig.get_scenes_dir(args.data_dir)
    scene_dirs = sorted(glob(os.path.join(scenes_dir, '*')))
    print(f"[INFO] Found {len(scene_dirs)} scenes")

    scene_metrics = {d: load_metrics(d) for d in tqdm(scene_dirs)}

    # Visualizations
    create_metrics_violinplot(scene_metrics)
    create_grid_image(scene_dirs, args.view_idx, mode='diff', cmap='jet',
                      filename='grid_diff_density.png', label='|Prediction - GT|', grid_size=VisualizationConfig.GRID_SIZE)
    create_grid_image(scene_dirs, args.view_idx, mode='pred', cmap='jet',
                      filename='grid_pred_density.png', label='Predicted Density (kg/m³)', grid_size=VisualizationConfig.GRID_SIZE)
    create_grid_image(scene_dirs, args.view_idx, mode='mask', cmap='gray',
                      filename='grid_valid_mask.png', label='', grid_size=VisualizationConfig.GRID_SIZE)

    print("[INFO] All visualizations generated successfully.")

if __name__ == '__main__':
    main()

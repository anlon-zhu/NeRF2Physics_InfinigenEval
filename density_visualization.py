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

# Configuration
from density_config import PathConfig, VisualizationConfig

##########################################
# Utility Functions
##########################################

def load_metrics(scene_dir, averaged=True):
    """Load metrics from density_metrics_avg.json or density_metrics.json."""
    output_dir = PathConfig.get_evaluation_output_dir(scene_dir)
    metrics_path = (PathConfig.get_avg_metrics_file(output_dir)
                    if averaged else PathConfig.get_metrics_file(output_dir))
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def load_density_map(scene_dir, view_idx, ground_truth=False):
    """Load predicted or GT density map for a given view."""
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
    """Compute percentile-based min/max for colormaps."""
    values = np.concatenate([m.flatten() for m in maps])
    values = values[(values > 0) & (values != -1)]
    if values.size > 0:
        return np.percentile(values, low), np.percentile(values, high)
    return 0, 1

##########################################
# Visualization Functions
##########################################

def create_metrics_summary(scene_metrics):
    """Create Table 1: Aggregate Metrics Summary."""
    metrics = ['ADE', 'ALDE', 'MedADE', 'MnRE']
    data = {
        'Metric': metrics,
        'Aggregate (All Scenes)': [],
        'Spread (All Scenes)': []
    }

    for m in metrics:
        vals = [s.get(m, 0) for s in scene_metrics.values() if s is not None]
        agg = np.median(vals) if m == 'MedADE' else np.mean(vals)
        q75, q25 = np.nanquantile(vals, [0.75, 0.25])
        iqr = q75 - q25
        spread = iqr if m == 'MedADE' else np.std(vals)
        data['Aggregate (All Scenes)'].append(f"{agg:.2f}")
        data['Spread (All Scenes)'].append(f"{spread:.2f}")

    df = pd.DataFrame(data)
    df.to_csv('metrics_summary_table.csv', index=False)
    with open('metrics_summary_table.tex', 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print("[INFO] Table 1 saved as CSV and LaTeX")
    return df

def create_metrics_violinplot(scene_metrics):
    """Figure 1: Violin distribution of all metrics."""
    rows = []
    for sid, metrics in scene_metrics.items():
        if metrics:
            for k, v in metrics.items():
                rows.append({'Scene': sid, 'Metric': k, 'Value': v})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Metric', y='Value', data=df, inner='box', palette='viridis')
    plt.title('Distribution of Evaluation Metrics Across All Scenes')
    plt.savefig(VisualizationConfig.METRICS_DIST_FILENAME, dpi=300)
    plt.close()
    print(f"[INFO] Figure 1 saved as {VisualizationConfig.METRICS_DIST_FILENAME}")

def create_scenewise_bar(scene_metrics, metric='ADE', filename='scenewise_ADE.png'):
    """Figure 2: Bar plot of scene-wise ADE."""
    data = [(os.path.basename(k).replace('scene_', ''), v[metric])
            for k, v in scene_metrics.items() if v and metric in v]
    data.sort(key=lambda x: x[1])
    scene_ids, values = zip(*data)
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(values)), values, color=plt.cm.viridis(np.linspace(0, 1, len(values))))
    plt.xticks(range(0, len(values), max(1, len(values)//10)), 
               [scene_ids[i] for i in range(0, len(values), max(1, len(values)//10))], 
               rotation=45)
    plt.title(f'Scene-wise {metric} (Sorted)')
    plt.ylabel(metric)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[INFO] Scene-wise {metric} saved as {filename}")

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
                diff[gt == -1] = 0
                images.append(diff)
                labels.append(os.path.basename(d))
            elif mode == 'mask':
                mask = (pred > 0).astype(np.float32)
                images.append(mask)
                labels.append(os.path.basename(d))
        if len(images) >= max_items:
            break

    vmin, vmax = compute_global_range(images)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for ax, img, title in zip(axes.flatten(), images, labels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    for ax in axes.flatten()[len(images):]:
        ax.axis('off')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
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

    create_metrics_summary(scene_metrics)
    create_metrics_violinplot(scene_metrics)
    create_scenewise_bar(scene_metrics, metric='MnRE')
    create_scenewise_bar(scene_metrics, metric='ADE')
    create_scenewise_bar(scene_metrics, metric='MedADE')

    create_grid_image(scene_dirs, args.view_idx, mode='diff', cmap=VisualizationConfig.DIFF_COLORMAP,
                      filename='grid_diff_density.png', label='|Prediction - GT|', grid_size=VisualizationConfig.GRID_SIZE)
    create_grid_image(scene_dirs, args.view_idx, mode='pred', cmap='viridis',
                      filename='grid_pred_density.png', label='Predicted Density (kg/m³)', grid_size=VisualizationConfig.GRID_SIZE)
    create_grid_image(scene_dirs, args.view_idx, mode='mask', cmap='binary',
                      filename='grid_valid_mask.png', label='Valid Prediction Mask', grid_size=VisualizationConfig.GRID_SIZE)

    print("[INFO] All aggregate visualizations generated successfully.")

if __name__ == '__main__':
    main()

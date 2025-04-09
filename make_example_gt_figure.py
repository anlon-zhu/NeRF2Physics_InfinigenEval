import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def collect_all_scene_paths(parent_dir, date_dirs):
    all_scene_paths = []
    for date_dir in date_dirs:
        full_date_path = os.path.join(parent_dir, date_dir)
        if not os.path.isdir(full_date_path):
            continue
        for scene_id in os.listdir(full_date_path):
            scene_path = os.path.join(full_date_path, scene_id, "frames/MaterialsDensity/camera_0")
            if os.path.isdir(scene_path):
                all_scene_paths.append(scene_path)
    return all_scene_paths


def create_scene_grid_from_paths(
    scene_paths,
    output_path="scene_grid_gt_combined.png",
    rows=2,
    cols=5,
    thumbs_per_scene=4
):
    selected_scenes = random.sample(scene_paths, rows * cols)
    scenes_images = []

    for path in selected_scenes:
        image_files = sorted([f for f in os.listdir(path) if f.endswith(".png")])
        if not image_files:
            continue

        main_img_file = image_files[0]
        main_img = Image.open(os.path.join(path, main_img_file))

        mini_candidates = image_files[1:] if len(image_files) > 1 else image_files
        if len(mini_candidates) < thumbs_per_scene:
            mini_candidates = image_files
        mini_imgs_files = random.sample(mini_candidates, thumbs_per_scene)
        mini_imgs = [Image.open(os.path.join(path, m)) for m in mini_imgs_files]

        scenes_images.append((main_img, mini_imgs))

    fig, axes = plt.subplots(rows * 2, cols, figsize=(20, 8))
    plt.subplots_adjust(
        left=0.01, right=0.99,
        top=0.02, bottom=0.01,
        wspace=0.01, hspace=0.01
    )

    for idx, (main_img, mini_imgs) in enumerate(scenes_images):
        row, col = divmod(idx, cols)

        ax_main = axes[row * 2][col]
        ax_main.imshow(main_img)
        ax_main.axis("off")

        thumb_height = main_img.height // 5
        mini_strip = Image.new("RGB", (main_img.width, thumb_height))
        width_per_thumb = main_img.width // thumbs_per_scene

        for i, m in enumerate(mini_imgs):
            m_resized = m.resize((width_per_thumb, thumb_height))
            mini_strip.paste(m_resized, (i * width_per_thumb, 0))

        ax_mini = axes[row * 2 + 1][col]
        ax_mini.imshow(mini_strip)
        ax_mini.axis("off")

    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved combined ground truth grid to {output_path}")


# Example usage
if __name__ == "__main__":
    parent_dir = "/n/fs/scratch/az4244/mvs_30_renders"
    date_dirs = [
        "2025-03-30_02-32_density_gt_1",
        "2025-03-31_01-32_density_gt_1",
        "2025-04-01_19-22_density_gt_1",
        "2025-04-02_01-09_density_gt_1",
    ]

    all_scene_paths = collect_all_scene_paths(parent_dir, date_dirs)

    create_scene_grid_from_paths(
        scene_paths=all_scene_paths,
        output_path="scene_grid_gt_combined.png",
        rows=2,
        cols=5
    )

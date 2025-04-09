import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Path config
base_dir = "/n/fs/scratch/az4244/nerf2physics/infinigen_nerf_data/scenes"
output_path = "scene_grid.png"

# Settings
main_image_name = "Image_0_0_0048_0.png"  # fallback if specific one isn't found
rows, cols = 2, 5
thumbs_per_scene = 4

# Select 10 random scene folders
scene_folders = sorted(os.listdir(base_dir))
selected_scenes = random.sample(scene_folders, rows * cols)

# Load image groups
scenes_images = []
for scene_id in selected_scenes:
    scene_path = os.path.join(base_dir, scene_id, "infinigen_images")
    image_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".png")])
    
    # Try to get the designated main image or fallback to the first
    main_img_file = main_image_name if main_image_name in image_files else image_files[0]
    main_img = Image.open(os.path.join(scene_path, main_img_file))
    
    # Sample 4 distinct mini-images (excluding the main one)
    mini_choices = [f for f in image_files if f != main_img_file]
    mini_imgs = [Image.open(os.path.join(scene_path, f)) for f in random.sample(mini_choices, thumbs_per_scene)]
    
    scenes_images.append((main_img, mini_imgs))

# Plot
fig, axes = plt.subplots(rows * 2, cols, figsize=(20, 8))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for idx, (main_img, mini_imgs) in enumerate(scenes_images):
    row, col = divmod(idx, cols)

    # Display main image
    ax_main = axes[row * 2][col]
    ax_main.imshow(main_img)
    ax_main.axis("off")

    # Display mini images below
    for i in range(thumbs_per_scene):
        ax_mini = axes[row * 2 + 1][col]
        mini_concat = Image.new("RGB", (main_img.width, main_img.height // 5))
        width_per_img = main_img.width // thumbs_per_scene
        
        # Resize and paste the minis side-by-side
        mini_row = Image.new("RGB", (main_img.width, mini_concat.height))
        for j, m in enumerate(mini_imgs):
            m_resized = m.resize((width_per_img, mini_concat.height))
            mini_row.paste(m_resized, (j * width_per_img, 0))
        ax_mini.imshow(mini_row)
        ax_mini.axis("off")
        break  # only add one row per cell

plt.savefig(output_path, bbox_inches='tight', dpi=200)
print(f"Saved to {output_path}")

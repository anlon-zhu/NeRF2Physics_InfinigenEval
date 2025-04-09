import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def create_scene_grid_infinigen(
    base_dir,
    output_path="scene_grid_infinigen.png",
    rows=2,
    cols=5,
    thumbs_per_scene=4,
    main_image_pattern="Image_0_0_0048_0.png"
):
    """
    Creates a rows x cols grid, each cell has:
      - 1 main image from infinigen_images
      - 4 mini-images from the same folder
    """
    # Collect scene folders
    scene_folders = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    # Pick exactly rows*cols scenes randomly
    selected_scenes = random.sample(scene_folders, rows * cols)

    # Load image data
    scenes_images = []
    for scene_id in selected_scenes:
        scene_path = os.path.join(base_dir, scene_id, "infinigen_images")
        if not os.path.exists(scene_path):
            # Skip if no infinigen_images folder
            continue
        
        # Get all .png images
        image_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".png")])
        if not image_files:
            continue
        
        # Main image is either the specified pattern or fallback to first
        main_img_file = main_image_pattern if main_image_pattern in image_files else image_files[0]
        main_img_path = os.path.join(scene_path, main_img_file)
        main_img = Image.open(main_img_path)

        # Pick 4 mini-images, excluding the main image if present
        mini_candidates = [f for f in image_files if f != main_img_file]
        # If there are fewer than 4 other images, just sample as many as possible
        if len(mini_candidates) < thumbs_per_scene:
            mini_candidates = image_files  # fallback to use main image too
        mini_imgs_files = random.sample(mini_candidates, thumbs_per_scene)
        mini_imgs = [Image.open(os.path.join(scene_path, m)) for m in mini_imgs_files]

        scenes_images.append((main_img, mini_imgs))

    # Create the figure & axes
    fig, axes = plt.subplots(rows * 2, cols, figsize=(20, 8))

    # Reduce spacing between images
    # You can tweak these to get the exact look you want
    plt.subplots_adjust(
        left=0.01, right=0.99,
        top=0.99, bottom=0.01,
        wspace=0.01, hspace=0.01
    )

    # Populate subplots
    for idx, (main_img, mini_imgs) in enumerate(scenes_images):
        row, col = divmod(idx, cols)

        # Main image goes on top
        ax_main = axes[row * 2][col]
        ax_main.imshow(main_img)
        ax_main.axis("off")

        # Create a horizontal strip for the mini-images
        # We'll do it by pasting them side-by-side into one pillow image
        thumb_height = main_img.height // 5  # scale as you like
        mini_strip = Image.new("RGB", (main_img.width, thumb_height))
        width_per_thumb = main_img.width // thumbs_per_scene

        for i, m in enumerate(mini_imgs):
            # Resize each mini to fit side by side
            m_resized = m.resize((width_per_thumb, thumb_height))
            mini_strip.paste(m_resized, (i * width_per_thumb, 0))

        ax_mini = axes[row * 2 + 1][col]
        ax_mini.imshow(mini_strip)
        ax_mini.axis("off")

    # Save
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Regular Infinigen grid saved to {output_path}")


# Example usage (edit base_dir to your actual path):
if __name__ == "__main__":
    base_dir_infinigen = "/n/fs/scratch/az4244/nerf2physics/infinigen_nerf_data/scenes"
    create_scene_grid_infinigen(
        base_dir=base_dir_infinigen,
        output_path="scene_grid_infinigen.png",
        rows=2,
        cols=5
    )

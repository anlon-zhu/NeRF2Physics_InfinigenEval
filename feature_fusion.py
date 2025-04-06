import os
import numpy as np
import json
import torch
import open_clip
from PIL import Image
import logging
import time
from datetime import datetime

from utils import *
from arguments import get_args

# Import debug-only modules conditionally
def setup_debug_environment(debug_mode):
    global tqdm, plt
    if debug_mode:
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        
        # Configure logging
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'feature_fusion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        # Simple logging to console only with less verbose output
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Define a no-op tqdm replacement
        class NoOpTqdm:
            def __init__(self, iterable=None, **kwargs):
                self.iterable = iterable
            def __iter__(self):
                return iter(self.iterable)
            def __enter__(self):
                return self
            def __exit__(self, *args, **kwargs):
                pass
        tqdm = NoOpTqdm


CLIP_BACKBONE = 'ViT-B-16'
CLIP_CHECKPOINT = 'datacomp_xl_s13b_b90k'
CLIP_INPUT_SIZE = 224
CLIP_OUTPUT_SIZE = 512


def get_patch_features(pts, imgs, depths, w2cs, K, model, preprocess_fn, occ_thr,
                       patch_size=56, batch_size=8, device='cuda', debug_dir=None, debug_mode=False):
    if debug_mode:
        logging.info(f"Starting feature extraction for {len(pts)} points across {len(imgs)} images")
        start_time = time.time()
    n_imgs = len(imgs)
    n_pts = len(pts)

    patch_features = torch.zeros(n_imgs, n_pts, CLIP_OUTPUT_SIZE, device=device, requires_grad=False)
    is_visible = torch.zeros(n_imgs, n_pts, device=device, dtype=torch.bool, requires_grad=False)
    half_patch_size = patch_size // 2
    
    # Debug stats (only tracked if debug mode is enabled)
    total_visible_points = 0
    points_per_image = [0] * n_imgs if debug_mode else None
    out_of_bounds_points = 0
    occluded_points = 0

    K = np.array(K)
    with torch.no_grad(), torch.cuda.amp.autocast():
        model.to(device)

        for i in tqdm(range(n_imgs), desc="Processing images" if debug_mode else None):
            if debug_mode:
                logging.info(f"Processing image {i+1}/{n_imgs}")
            h, w, c = imgs[i].shape
            if len(K.shape) == 3:
                curr_K = K[i]
            else:
                curr_K = K
            pts_2d, dists = project_3d_to_2d(pts, w2cs[i], curr_K, return_dists=True)
            pts_2d = np.round(pts_2d).astype(np.int32)

            observed_dists = depths[i]
        
            # loop through pts in batches
            for batch_start in tqdm(range(0, n_pts, batch_size), 
                                   desc=f"Image {i+1} batches" if debug_mode else None, 
                                   leave=False):
                curr_batch_size = min(batch_size, n_pts - batch_start)
                batch_patches = torch.zeros(curr_batch_size, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, device=device)
                
                for j in range(curr_batch_size):
                    x, y = pts_2d[batch_start + j]

                    if x >= half_patch_size and x < w - half_patch_size and \
                       y >= half_patch_size and y < h - half_patch_size:
                        is_occluded = dists[batch_start + j] > observed_dists[y, x] + occ_thr
                        if not is_occluded:
                            patch = imgs[i][y - half_patch_size:y + half_patch_size, x - half_patch_size:x + half_patch_size]
                            patch = Image.fromarray(patch)
                            
                            patch = preprocess_fn(patch).unsqueeze(0).to(device)
                            batch_patches[j] = patch
                            is_visible[i, batch_start + j] = True
                            
                            if debug_mode:
                                points_per_image[i] += 1
                                total_visible_points += 1
                                
                                # Save some sample patches for debugging (only a few to avoid excessive files)
                                if debug_dir and (batch_start + j) % 100 == 0:
                                    os.makedirs(os.path.join(debug_dir, f"image_{i}"), exist_ok=True)
                                    patch_img = Image.fromarray(imgs[i][y - half_patch_size:y + half_patch_size, x - half_patch_size:x + half_patch_size])
                                    patch_img.save(os.path.join(debug_dir, f"image_{i}", f"patch_{batch_start + j}.png"))
                        elif debug_mode:
                            occluded_points += 1
                    elif debug_mode:
                        out_of_bounds_points += 1

                if is_visible[i, batch_start:batch_start + batch_size].any():
                    patch_features[i, batch_start:batch_start + curr_batch_size] = model.encode_image(batch_patches)

    if debug_mode:
        elapsed_time = time.time() - start_time
        logging.info(f"Feature extraction completed in {elapsed_time:.2f} seconds")
        logging.info(f"Total visible points: {total_visible_points} out of {n_pts * n_imgs} possible point-image pairs")
        logging.info(f"Points out of image bounds: {out_of_bounds_points}")
        logging.info(f"Occluded points: {occluded_points}")
        
        # Create visibility heatmap
        if debug_dir:
            plt.figure(figsize=(12, 8))
            plt.bar(range(n_imgs), points_per_image)
            plt.xlabel('Image Index')
            plt.ylabel('Number of Visible Points')
            plt.title('Points Visibility per Image')
            plt.savefig(os.path.join(debug_dir, 'visibility_per_image.png'))
            plt.close()
            
            # Save visibility map
            visibility_map = is_visible.cpu().numpy()
            plt.figure(figsize=(12, 8))
            plt.imshow(visibility_map, aspect='auto', cmap='viridis')
            plt.colorbar(label='Visible')
            plt.xlabel('Point Index')
            plt.ylabel('Image Index')
            plt.title('Point Visibility Map')
            plt.savefig(os.path.join(debug_dir, 'visibility_map.png'))
            plt.close()
            
            # Save feature statistics
            feature_norms = torch.norm(patch_features, dim=2).cpu().numpy()
            plt.figure(figsize=(12, 8))
            plt.imshow(feature_norms, aspect='auto')
            plt.colorbar(label='Feature Norm')
            plt.xlabel('Point Index')
            plt.ylabel('Image Index')
            plt.title('Feature Norm Map')
            plt.savefig(os.path.join(debug_dir, 'feature_norm_map.png'))
            plt.close()
    
    return patch_features, is_visible


def process_scene(args, scene_dir, model, preprocess_fn):
    debug_mode = args.debug
    
    if debug_mode:
        logging.info(f"\n{'='*50}\nProcessing scene: {os.path.basename(scene_dir)}\n{'='*50}")
        start_time = time.time()
    
    scene_name = os.path.basename(scene_dir)
    pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
    t_file = os.path.join(scene_dir, 'transforms.json')
    img_dir = os.path.join(scene_dir, 'images')
    depth_dir = os.path.join(scene_dir, 'ns', 'renders', 'depth')
    
    # Create debug directory only if debug mode is enabled
    debug_dir = None
    if debug_mode:
        debug_dir = os.path.join(scene_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        logging.info(f"Debug outputs will be saved to: {debug_dir}")

    if debug_mode:
        logging.info("Loading point cloud...")
    pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.feature_voxel_size)
    if debug_mode:
        logging.info(f"Loaded {len(pts)} points after downsampling with voxel size {args.feature_voxel_size}")
    
    if debug_mode:
        logging.info("Loading camera transforms...")
    w2cs, K = parse_transforms_json(t_file, return_w2c=True, different_Ks=args.different_Ks)
    if debug_mode:
        logging.info(f"Loaded {len(w2cs)} camera transforms")
    
    if debug_mode:
        logging.info("Loading nerfstudio transforms...")
    ns_transform, scale = parse_dataparser_transforms_json(dt_file)
    
    if debug_mode:
        logging.info("Loading images...")
    imgs = load_images(img_dir)
    if debug_mode:
        logging.info(f"Loaded {len(imgs)} images")
    
    if debug_mode:
        logging.info("Loading depth maps...")
    depths = load_depths(depth_dir, Ks=None)
    if debug_mode:
        logging.info(f"Loaded {len(depths)} depth maps")
    
    # Visualize point cloud distribution (only in debug mode)
    if debug_mode and debug_dir:
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        sample_pts = pts[::max(1, len(pts)//1000)]  # Sample points to avoid overcrowding
        ax.scatter(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2], s=1, alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud Distribution (sampled 1/{max(1, len(pts)//1000)})')
        plt.savefig(os.path.join(debug_dir, 'point_cloud_distribution.png'))
        plt.close()

    if debug_mode:
        logging.info('scene: %s, points: %d, scale: %.4f' % (scene_name, len(pts), scale))
    else:
        print('Processing scene: %s, points: %d, scale: %.4f' % (scene_name, len(pts), scale))

    with torch.no_grad():
        occ_thr = args.occ_thr * scale
        if debug_mode:
            logging.info(f"Using occlusion threshold: {occ_thr} (base: {args.occ_thr}, scale: {scale})")
            logging.info(f"Extracting features with patch size: {args.patch_size}, batch size: {args.batch_size}")
        
        patch_features, is_visible = get_patch_features(pts, imgs, depths, w2cs, K, 
                                                        model, preprocess_fn, 
                                                        occ_thr, patch_size=args.patch_size, batch_size=args.batch_size, 
                                                        device=args.device, debug_dir=debug_dir, debug_mode=debug_mode)
        
    out_dir = os.path.join(scene_dir, 'features')
    os.makedirs(out_dir, exist_ok=True)
    
    # Save features and visibility
    feature_path = os.path.join(out_dir, f'patch_features_{args.feature_save_name}.pt')
    visibility_path = os.path.join(out_dir, f'is_visible_{args.feature_save_name}.pt')
    voxel_path = os.path.join(out_dir, f'voxel_size_{args.feature_save_name}.json')
    
    if debug_mode:
        logging.info(f"Saving features to {feature_path}")
    torch.save(patch_features, feature_path)
    
    if debug_mode:
        logging.info(f"Saving visibility to {visibility_path}")
    torch.save(is_visible, visibility_path)
    
    if debug_mode:
        logging.info(f"Saving voxel size to {voxel_path}")
    with open(voxel_path, 'w') as f:
        json.dump({'voxel_size': args.feature_voxel_size}, f, indent=4)
    
    # Compute and log feature statistics (only in debug mode)
    if debug_mode:
        visible_features = patch_features[is_visible]
        if len(visible_features) > 0:
            feature_mean = torch.mean(visible_features, dim=0)
            feature_std = torch.std(visible_features, dim=0)
            feature_min = torch.min(visible_features, dim=0)[0]
            feature_max = torch.max(visible_features, dim=0)[0]
            
            logging.info(f"Feature statistics:")
            logging.info(f"  Mean norm: {torch.norm(feature_mean).item():.4f}")
            logging.info(f"  Std norm: {torch.norm(feature_std).item():.4f}")
            logging.info(f"  Min norm: {torch.norm(feature_min).item():.4f}")
            logging.info(f"  Max norm: {torch.norm(feature_max).item():.4f}")
            
            # Save feature PCA visualization
            if len(visible_features) > 10 and debug_dir:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                features_np = visible_features.cpu().numpy()
                features_pca = pca.fit_transform(features_np)
                
                plt.figure(figsize=(10, 10))
                plt.scatter(features_pca[:, 0], features_pca[:, 1], c=features_pca[:, 2], s=1, alpha=0.5, cmap='viridis')
                plt.colorbar(label='PCA Component 3')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.title('PCA of Feature Vectors')
                plt.savefig(os.path.join(debug_dir, 'feature_pca.png'))
                plt.close()
                
                logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        else:
            logging.warning("No visible features found!")
        
        elapsed_time = time.time() - start_time
        logging.info(f"Scene processing completed in {elapsed_time:.2f} seconds")

    return pts, patch_features, is_visible
    
    
if __name__ == '__main__':   
    args = get_args()
    
    # Set up debug environment based on debug flag
    setup_debug_environment(args.debug)
    
    if args.debug:
        start_time = time.time()
        # Log all arguments
        logging.info(f"Starting feature fusion with arguments:")
        for arg, value in vars(args).items():
            logging.info(f"  {arg}: {value}")

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)
    
    if args.scene_name is not None:
        scenes = [args.scene_name]
    
    if args.debug:
        logging.info(f"Processing {len(scenes)} scenes: {', '.join(scenes)}")
        logging.info(f"Loading CLIP model: {CLIP_BACKBONE} with checkpoint {CLIP_CHECKPOINT}")
    else:
        print(f"Processing {len(scenes)} scenes")
        print(f"Loading CLIP model: {CLIP_BACKBONE}")
        
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    model.to(args.device)
    
    if args.debug:
        logging.info(f"Model loaded and moved to {args.device}")

    for j, scene in enumerate(scenes):
        if args.debug:
            logging.info(f"Processing scene {j+1}/{len(scenes)}: {scene}")
        pts, patch_features, is_visible = process_scene(args, os.path.join(scenes_dir, scene), model, preprocess)
        
        if args.debug:
            logging.info(f"Scene {scene} processed.")
            
            # Log visibility statistics
            total_visible = torch.sum(is_visible).item()
            total_points = is_visible.numel()
            logging.info(f"Visibility statistics for {scene}:")
            logging.info(f"  Total visible point-image pairs: {total_visible} out of {total_points} ({total_visible/total_points*100:.2f}%)")
            
            # Memory usage stats
            if torch.cuda.is_available():
                logging.info(f"GPU memory usage:")
                logging.info(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                logging.info(f"  Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        else:
            print(f"Scene {scene} processed.")
    
    if args.debug:
        total_elapsed_time = time.time() - start_time
        logging.info(f"All scenes processed in {total_elapsed_time:.2f} seconds")
    else:
        print("All scenes processed.")


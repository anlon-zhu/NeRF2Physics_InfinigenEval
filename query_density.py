import os
import json
import torch
import numpy as np
import open_clip
from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from predict_property import predict_physical_property_query
from arguments import get_args
from utils import get_scenes_list, load_ns_point_cloud, parse_dataparser_transforms_json

def query_point_densities(args, save_results=True):
    """
    Get and optionally save density predictions for each point in the scene.
    Returns dictionary with results for each scene.
    """
    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    # Initialize CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)

    results = {}
    for scene in scenes:
        scene_dir = os.path.join(scenes_dir, scene)
        
        # Get the grid of points
        pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
        dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
        
        # Option 1: Use grid points
        query_pts = 'grid'
        
        # Option 2: If you want to specify custom points:
        # query_pts = torch.tensor([[x1, y1, z1], [x2, y2, z2], ...]).to(args.device)
        
        # Get full prediction information by setting return_all=True
        prediction_info = predict_physical_property_query(
            args, query_pts, scene_dir, clip_model, clip_tokenizer, return_all=True
        )
        
        # To make the results serializable for JSON
        serializable_info = {
            'query_pred_probs': prediction_info['query_pred_probs'].tolist(),
            'query_pred_vals': prediction_info['query_pred_vals'].tolist(),
            'source_pts': prediction_info['source_pts'].tolist(),
            'mat_names': prediction_info['mat_names'],
        }
        
        # If query_pts was 'grid', we load the actual points
        if query_pts == 'grid':
            points = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.sample_voxel_size)
            serializable_info['query_points'] = points.tolist()
            
        results[scene] = serializable_info
        
        if save_results:
            # Save the point densities for later use
            os.makedirs(os.path.join(scene_dir, 'point_densities'), exist_ok=True)
            output_file = os.path.join(scene_dir, 'point_densities', f'densities_{args.property_name}.json')
            with open(output_file, 'w') as f:
                json.dump(serializable_info, f)
            print(f"Saved point densities to {output_file}")

    return results

if __name__ == '__main__':
    args = get_args()
    query_point_densities(args)

import os
import subprocess
import shutil
import json
from utils import get_last_file_in_folder, get_scenes_list
from arguments import get_args


def move_files_to_folder(source_dir, target_dir):
    for file in os.listdir(source_dir):
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))


def is_scene_reconstructed(metadata_file, scene):
    """Check if a scene has been reconstructed according to metadata"""
    if not os.path.exists(metadata_file):
        return False
        
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return scene in metadata.get('reconstructed_seeds', [])
    except (json.JSONDecodeError, FileNotFoundError):
        return False


def update_metadata_with_reconstructed(metadata_file, scene):
    """Update metadata file with newly reconstructed scene"""
    # Create metadata file if it doesn't exist
    if not os.path.exists(metadata_file):
        metadata = {
            "included_seeds": [],
            "skipped_seeds": [],
            "reconstructed_seeds": [],
            "total_seeds": 0,
            "included_count": 0,
            "skipped_count": 0,
            "reconstructed_count": 0,
            "creation_time": ""
        }
    else:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            metadata = {
                "included_seeds": [],
                "skipped_seeds": [],
                "reconstructed_seeds": [],
                "total_seeds": 0,
                "included_count": 0,
                "skipped_count": 0,
                "reconstructed_count": 0,
                "creation_time": ""
            }
    
    # Add scene to reconstructed_seeds if not already there
    if scene not in metadata['reconstructed_seeds']:
        metadata['reconstructed_seeds'].append(scene)
        metadata['reconstructed_count'] = len(metadata['reconstructed_seeds'])
    
    # Write updated metadata back to file
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated metadata for reconstructed scene: {scene}")


def reconstruct_scene(args, scene):
    """Reconstruct a single scene"""
    print(f"Starting reconstruction for scene: {scene}")
    
    scenes_dir = os.path.join(args.data_dir, 'scenes')
    
    # Verify the scene directory exists
    scene_dir = os.path.join(scenes_dir, scene)
    if not os.path.isdir(scene_dir):
        print(f"Error: Scene directory {scene_dir} does not exist, skipping")
        return False
        
    base_dir = os.path.join(scene_dir, 'ns')
    os.makedirs(base_dir, exist_ok=True)

    # Calling ns-train with explicit logging of command
    cmd = [
        'ns-train', 'nerfacto',
        '--data', scene_dir,
        '--output_dir', base_dir,
        '--project_name', args.project_name,
        '--experiment_name', scene,
        '--max_num_iterations', str(args.training_iters),
        '--pipeline.model.proposal-initial-sampler', 'uniform',
        '--steps-per-eval-image', '10000',
        '--viewer.quit-on-train-completion', 'True'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Error: ns-train failed for scene {scene}")
        return False

    ns_dir = get_last_file_in_folder(os.path.join(base_dir, '%s/nerfacto' % scene))

    # Copying dataparser_transforms (contains scale)
    result = subprocess.run([
        'scp', '-r', 
        os.path.join(ns_dir, 'dataparser_transforms.json'), 
        os.path.join(base_dir, 'dataparser_transforms.json')
    ])

    half_bbox_size = args.bbox_size / 2

    # Calling ns-export pcd
    result = subprocess.run([
        'ns-export', 'pointcloud',
        '--load-config', os.path.join(ns_dir, 'config.yml'),
        '--output-dir', base_dir,
        '--num-points', str(args.num_points),
        '--remove-outliers', 'True',
        '--normal-method', 'open3d',
        '--num-rays-per-batch', '16384'
    ])
    
    if result.returncode != 0:
        print(f"Error: ns-export failed for scene {scene}")
        return False

    # Calling ns-render 
    result = subprocess.run([
        'ns-render', 'dataset',
        '--load-config', os.path.join(ns_dir, 'config.yml'),
        '--output-path', os.path.join(base_dir, 'renders'),
        '--rendered-output-names', 'raw-depth',
        '--split', 'train+test',
    ])
    
    if result.returncode != 0:
        print(f"Error: ns-render failed for scene {scene}")
        return False

    # Collect all depths in one folder
    os.makedirs(os.path.join(base_dir, 'renders', 'depth'), exist_ok=True)
    move_files_to_folder(os.path.join(base_dir, 'renders', 'test', 'raw-depth'), os.path.join(base_dir, 'renders', 'depth'))
    move_files_to_folder(os.path.join(base_dir, 'renders', 'train', 'raw-depth'), os.path.join(base_dir, 'renders', 'depth'))
    
    return True


if __name__ == '__main__':
    
    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    metadata_file = os.path.join(args.data_dir, 'metadata.json')
    
    # Create metadata file if it doesn't exist
    if not os.path.exists(metadata_file):
        print(f"Creating new metadata file at {metadata_file}")
        metadata = {
            "included_seeds": [],
            "skipped_seeds": [],
            "reconstructed_seeds": [],
            "total_seeds": 0,
            "included_count": 0,
            "skipped_count": 0,
            "reconstructed_count": 0,
            "creation_time": ""
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # If a specific scene name is provided, only process that scene
    if hasattr(args, 'scene_name') and args.scene_name:
        scenes = [args.scene_name]
        print(f"Processing single scene: {args.scene_name}")
    else:
        scenes = get_scenes_list(args)
        print(f"Processing {len(scenes)} scenes from directory")

    for scene in scenes:
        # Check if this scene has already been reconstructed
        if is_scene_reconstructed(metadata_file, scene):
            print(f"Scene {scene} has already been reconstructed, skipping")
            continue
            
        # Reconstruct the scene
        success = reconstruct_scene(args, scene)
        
        # Update metadata if successful
        if success:
            update_metadata_with_reconstructed(metadata_file, scene)

    print("NeRF reconstruction complete.")

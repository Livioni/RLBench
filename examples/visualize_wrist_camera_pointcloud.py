import numpy as np
import os
import json
import argparse
from pathlib import Path

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed. Please install it with: pip install open3d")

from pyrep.objects import VisionSensor


def depth_to_pointcloud(depth, extrinsics, intrinsics):
    """
    Convert depth image to 3D point cloud using camera parameters.
    This follows the RLBench/PyRep convention and CoppeliaSim v4.1.0 coordinate system.
    
    Args:
        depth: (H, W) depth image in meters
        extrinsics: (4, 4) camera extrinsics matrix (camera-to-world transformation)
        intrinsics: (3, 3) camera intrinsics matrix
    
    Returns:
        pointcloud: (H*W, 3) array of 3D points in world coordinates
    """
    # Use PyRep's built-in method which handles CoppeliaSim coordinate conventions
    pointcloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        depth, extrinsics, intrinsics
    )
    return pointcloud


def load_frame_data(episode_dir, frame_idx):
    """
    Load RGB, depth, pose, and intrinsics for a specific frame.
    
    Args:
        episode_dir: Path to episode directory
        frame_idx: Frame index to load
    
    Returns:
        dict with 'rgb', 'depth', 'pose', 'intrinsics' keys
    """
    frame_name = f'frame_{frame_idx:04d}'
    
    # Load depth (in meters)
    depth_path = os.path.join(episode_dir, 'depth', f'{frame_name}.npy')
    depth = np.load(depth_path)
    
    # Load pose (camera extrinsics)
    pose_path = os.path.join(episode_dir, 'pose', f'{frame_name}.npy')
    pose = np.load(pose_path)
    
    # Load intrinsics
    intrinsic_path = os.path.join(episode_dir, 'intrinsic', f'{frame_name}.npy')
    intrinsics = np.load(intrinsic_path)
    
    # Load RGB (optional, for coloring point cloud)
    rgb_path = os.path.join(episode_dir, 'images', f'{frame_name}.png')
    if os.path.exists(rgb_path):
        from PIL import Image
        rgb = np.array(Image.open(rgb_path))
    else:
        rgb = None
    
    return {
        'rgb': rgb,
        'depth': depth,
        'pose': pose,
        'intrinsics': intrinsics
    }


def create_colored_pointcloud(points, colors):
    """
    Create an Open3D point cloud with colors.
    
    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255)
    
    Returns:
        open3d.geometry.PointCloud
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required for visualization")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # Normalize colors to [0, 1]
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    return pcd


def visualize_pointcloud(pcd, window_name="Point Cloud Visualization"):
    """
    Visualize point cloud using Open3D.
    
    Args:
        pcd: open3d.geometry.PointCloud
        window_name: Window title
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required for visualization")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    
    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)
    
    # Run visualizer
    vis.run()
    vis.destroy_window()


def visualize_episode_sequence(episode_dir, start_frame=0, end_frame=None, 
                               step=1, save_ply=False, output_dir=None):
    """
    Visualize point clouds from a sequence of frames in an episode.
    
    Args:
        episode_dir: Path to episode directory
        start_frame: Starting frame index
        end_frame: Ending frame index (None = last frame)
        step: Frame step size
        save_ply: Whether to save point clouds as PLY files
        output_dir: Directory to save PLY files (if save_ply=True)
    """
    # Load metadata to get number of frames
    metadata_path = os.path.join(episode_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_frames = metadata['num_frames']
    
    if end_frame is None:
        end_frame = num_frames - 1
    
    print(f"Episode: {os.path.basename(episode_dir)}")
    print(f"Total frames: {num_frames}")
    print(f"Visualizing frames {start_frame} to {end_frame} (step={step})")
    print("-" * 60)
    
    if save_ply and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving PLY files to: {output_dir}")
    
    for frame_idx in range(start_frame, end_frame + 1, step):
        print(f"\nProcessing frame {frame_idx}...")
        
        # Load frame data
        frame_data = load_frame_data(episode_dir, frame_idx)
        depth = frame_data['depth']
        pose = frame_data['pose']
        intrinsics = frame_data['intrinsics']
        rgb = frame_data['rgb']
        
        print(f"  Depth shape: {depth.shape}")
        print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}] meters")
        print(f"  Pose (extrinsics) shape: {pose.shape}")
        print(f"  Intrinsics shape: {intrinsics.shape}")
        
        # Convert depth to point cloud
        print("  Converting depth to point cloud...")
        points = depth_to_pointcloud(depth, pose, intrinsics)
        
        # Reshape points (H*W, 3)
        H, W = depth.shape
        points = points.reshape(-1, 3)
        
        # Filter out invalid points (e.g., points at infinity or too far)
        valid_mask = np.isfinite(points).all(axis=1)
        valid_mask &= (np.linalg.norm(points, axis=1) < 10.0)  # Filter points > 10m away
        
        points = points[valid_mask]
        
        # Prepare colors if RGB is available
        colors = None
        if rgb is not None:
            colors = rgb.reshape(-1, 3)[valid_mask]
        
        print(f"  Valid points: {len(points)} / {H*W}")
        print(f"  Point cloud bounds:")
        print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # Create Open3D point cloud
        pcd = create_colored_pointcloud(points, colors)
        
        # Save as PLY if requested
        if save_ply and output_dir:
            ply_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.ply')
            o3d.io.write_point_cloud(ply_path, pcd)
            print(f"  Saved: {ply_path}")
        
        # Visualize
        window_name = f"Frame {frame_idx} - Press Q to close and continue"
        print(f"  Visualizing... (close window to continue)")
        visualize_pointcloud(pcd, window_name)


def visualize_merged_episode(episode_dir, start_frame=0, end_frame=None, 
                             step=1, save_ply=False, output_path=None):
    """
    Merge and visualize point clouds from multiple frames in an episode.
    
    Args:
        episode_dir: Path to episode directory
        start_frame: Starting frame index
        end_frame: Ending frame index (None = last frame)
        step: Frame step size
        save_ply: Whether to save merged point cloud as PLY file
        output_path: Path to save merged PLY file
    """
    # Load metadata
    metadata_path = os.path.join(episode_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_frames = metadata['num_frames']
    
    if end_frame is None:
        end_frame = num_frames - 1
    
    print(f"Merging frames {start_frame} to {end_frame} (step={step})...")
    
    all_points = []
    all_colors = []
    
    for frame_idx in range(start_frame, end_frame + 1, step):
        print(f"  Processing frame {frame_idx}...")
        
        # Load frame data
        frame_data = load_frame_data(episode_dir, frame_idx)
        depth = frame_data['depth']
        pose = frame_data['pose']
        intrinsics = frame_data['intrinsics']
        rgb = frame_data['rgb']
        
        # Convert to point cloud
        points = depth_to_pointcloud(depth, pose, intrinsics)
        points = points.reshape(-1, 3)
        
        # Filter valid points
        valid_mask = np.isfinite(points).all(axis=1)
        valid_mask &= (np.linalg.norm(points, axis=1) < 10.0)
        
        points = points[valid_mask]
        all_points.append(points)
        
        if rgb is not None:
            colors = rgb.reshape(-1, 3)[valid_mask]
            all_colors.append(colors)
    
    # Merge all points
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors) if all_colors else None
    
    print(f"\nMerged point cloud:")
    print(f"  Total points: {len(merged_points)}")
    print(f"  Bounds:")
    print(f"    X: [{merged_points[:, 0].min():.3f}, {merged_points[:, 0].max():.3f}]")
    print(f"    Y: [{merged_points[:, 1].min():.3f}, {merged_points[:, 1].max():.3f}]")
    print(f"    Z: [{merged_points[:, 2].min():.3f}, {merged_points[:, 2].max():.3f}]")
    
    # Create point cloud
    pcd = create_colored_pointcloud(merged_points, merged_colors)
    
    # Optionally downsample for better visualization
    if len(merged_points) > 1000000:
        print(f"  Downsampling point cloud (voxel size=0.001)...")
        pcd = pcd.voxel_down_sample(voxel_size=0.001)
        print(f"  Downsampled points: {len(pcd.points)}")
    
    # Save if requested
    if save_ply and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"  Saved: {output_path}")
    
    # Visualize
    # print("\nVisualizing merged point cloud...")
    # visualize_pointcloud(pcd, "Merged Point Cloud - Press Q to close")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize point clouds from extracted wrist camera data"
    )
    parser.add_argument(
        '--episode_dir', 
        type=str, 
        required=True,
        help='Path to episode directory (e.g., datasets/basketball_in_hoop/episode_0000)'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['sequence', 'merged'], 
        default='merged',
        help='Visualization mode: "sequence" (frame by frame) or "merged" (all frames combined)'
    )
    parser.add_argument(
        '--start_frame', 
        type=int, 
        default=0,
        help='Starting frame index'
    )
    parser.add_argument(
        '--end_frame', 
        type=int, 
        default=10,
        help='Ending frame index (default: last frame)'
    )
    parser.add_argument(
        '--step', 
        type=int, 
        default=2,
        help='Frame step size'
    )
    parser.add_argument(
        '--save_ply', 
        action='store_true',
        help='Save point clouds as PLY files'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output directory for PLY files (for sequence mode)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=None,
        help='Output path for merged PLY file (for merged mode)'
    )
    
    args = parser.parse_args()
    
    # Check if Open3D is available
    if not HAS_OPEN3D:
        print("Error: Open3D is not installed.")
        print("Please install it with: pip install open3d")
        return
    
    # Check if episode directory exists
    if not os.path.exists(args.episode_dir):
        print(f"Error: Episode directory not found: {args.episode_dir}")
        return
    
    # Run visualization
    if args.mode == 'sequence':
        output_dir = args.output_dir or os.path.join(args.episode_dir, 'pointclouds')
        visualize_episode_sequence(
            args.episode_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step,
            save_ply=args.save_ply,
            output_dir=output_dir if args.save_ply else None
        )
    else:  # merged
        output_path = args.output_path or os.path.join(
            args.episode_dir, 'merged_pointcloud.ply'
        )
        visualize_merged_episode(
            args.episode_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            step=args.step,
            save_ply=args.save_ply,
            output_path=output_path if args.save_ply else None
        )


if __name__ == '__main__':
    main()

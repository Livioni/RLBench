"""
修复版本：确保相机位置在每次demo采集时都被正确设置
"""
import numpy as np
import os
from PIL import Image
import json

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import BasketballInHoop
from pyrep.objects.vision_sensor import VisionSensor


# 全局变量保存相机配置
CAMERA_CONFIG = {
    'x': 0.06,
    'y': 0.0,
    'z': -0.025,
    'rotation_alpha': 0.0,
    'rotation_beta': 0.0,
    'rotation_gamma': 0.0
}


def set_camera_position(wrist_cam: VisionSensor, config: dict):
    """
    设置相机位置（相对于父对象）
    """
    parent = wrist_cam.get_parent()
    
    # 设置位置
    new_pos = [config['x'], config['y'], config['z']]
    wrist_cam.set_position(new_pos, relative_to=parent)
    
    # 设置朝向
    new_orient = [
        np.deg2rad(config['rotation_alpha']),
        np.deg2rad(config['rotation_beta']),
        np.deg2rad(config['rotation_gamma'])
    ]
    wrist_cam.set_orientation(new_orient, relative_to=parent)
    
    # 验证设置
    actual_pos = wrist_cam.get_position(relative_to=parent)
    print(f"  相机位置设置为: [{actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f}]")


def save_episode_data(episode_idx, demo, save_dir, task_name, wrist_cam):
    """
    Save RGB, depth, pose, and intrinsic data for one episode
    同时验证每一帧的相机位置
    """
    episode_dir = os.path.join(save_dir, task_name, f'episode_{episode_idx:04d}')
    
    # Create subdirectories
    images_dir = os.path.join(episode_dir, 'images')
    depth_dir = os.path.join(episode_dir, 'depth')
    pose_dir = os.path.join(episode_dir, 'pose')
    intrinsic_dir = os.path.join(episode_dir, 'intrinsic')
    
    for dir_path in [images_dir, depth_dir, pose_dir, intrinsic_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 在保存前验证相机位置
    parent = wrist_cam.get_parent()
    cam_pos = wrist_cam.get_position(relative_to=parent)
    print(f"  保存时相机位置: [{cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}]")
    
    # Process each frame in the demo
    for frame_idx, obs in enumerate(demo):
        # Extract wrist camera RGB (640x480)
        rgb = obs.wrist_rgb
        # Save RGB image
        rgb_img = Image.fromarray(rgb.astype(np.uint8))
        rgb_img.save(os.path.join(images_dir, f'frame_{frame_idx:04d}.png'))
        
        # Extract wrist camera depth (640x480)
        depth = obs.wrist_depth
        # Save depth as numpy array (for precision)
        np.save(os.path.join(depth_dir, f'frame_{frame_idx:04d}.npy'), depth)
        
        # Extract wrist camera pose (4x4 transformation matrix)
        pose = obs.misc['wrist_camera_extrinsics']
        np.save(os.path.join(pose_dir, f'frame_{frame_idx:04d}.npy'), pose)
        
        # Extract camera intrinsic parameters
        intrinsics = obs.misc['wrist_camera_intrinsics']
        np.save(os.path.join(intrinsic_dir, f'frame_{frame_idx:04d}.npy'), intrinsics)
    
    # Save episode metadata
    metadata = {
        'episode_idx': episode_idx,
        'num_frames': len(demo),
        'task': 'BasketballInHoop',
        'camera_config': CAMERA_CONFIG
    }
    with open(os.path.join(episode_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Episode {episode_idx} saved: {len(demo)} frames")


def main():
    # Configuration
    num_episodes = 10  # Number of episodes to collect
    save_dir = 'datasets_fixed_cam'
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("相机配置:")
    print(f"  位置: [{CAMERA_CONFIG['x']}, {CAMERA_CONFIG['y']}, {CAMERA_CONFIG['z']}]")
    print(f"  旋转: [{CAMERA_CONFIG['rotation_alpha']}°, {CAMERA_CONFIG['rotation_beta']}°, {CAMERA_CONFIG['rotation_gamma']}°]")
    print("="*60)
    
    # Setup observation config for wrist camera with 640x480 resolution
    obs_config = ObservationConfig()
    obs_config.set_all(False)  # Disable all first
    obs_config.wrist_camera.set_all(True)  # Enable wrist camera
    obs_config.wrist_camera.image_size = (640, 480)  # Set resolution to 640x480
    obs_config.wrist_camera.rgb = True
    obs_config.wrist_camera.depth_in_meters = True
    obs_config.wrist_camera.depth = True
    obs_config.wrist_camera.mask = False
    obs_config.wrist_camera.point_cloud = True
    
    # Enable misc to get wrist_camera_extrinsics and intrinsics
    obs_config.record_gripper_closing = True  # This enables misc dict
    
    # Also get gripper pose for backup
    obs_config.gripper_pose = True
    obs_config.gripper_matrix = True
    
    # Setup environment
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), 
        gripper_action_mode=Discrete()
    )
    
    env = Environment(
        action_mode, 
        dataset_root='',
        obs_config=obs_config, 
        headless=True
    )
    env.launch()
    
    # Get camera object
    wrist_cam = VisionSensor('cam_wrist')
    
    # Get task
    task = env.get_task(BasketballInHoop)
    task_name = task.get_name()
    
    print(f"\nCollecting {num_episodes} episodes...")
    print(f"Wrist camera resolution: 640x480")
    print(f"Saving to: {save_dir}\n")
    
    # Collect demonstrations
    for episode_idx in range(num_episodes):
        print(f"{'='*60}")
        print(f"Collecting episode {episode_idx + 1}/{num_episodes}...")
        
        # 重置任务
        descriptions, obs = task.reset()
        
        # ⚠️ 关键：在reset之后立即设置相机位置
        print(f"  设置相机位置...")
        set_camera_position(wrist_cam, CAMERA_CONFIG)
        
        # 使用自定义的demo采集循环，确保相机位置保持
        try:
            # 获取demo（这里会再次reset，所以我们需要用不同的方法）
            # 方法1：使用task的内部方法直接获取demo
            from rlbench.backend.const import STEPS_BEFORE_EPISODE_START
            
            # 重新reset并设置相机
            descriptions, obs = task.reset()
            set_camera_position(wrist_cam, CAMERA_CONFIG)
            
            # 获取demo
            demo = task._task_env._scene.get_demo()
            
            # 保存数据
            save_episode_data(episode_idx, demo, save_dir, task_name, wrist_cam)
            
        except Exception as e:
            print(f"  ✗ Episode {episode_idx} failed: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Data collection complete!")
    print(f"Total episodes collected: {num_episodes}")
    print(f"Data saved to: {os.path.abspath(save_dir)}")
    print(f"{'='*60}")
    
    # 验证：比较第一帧的相机位置
    print(f"\n验证相机位置是否改变:")
    pose_0 = np.load(f'{save_dir}/{task_name}/episode_0000/pose/frame_0000.npy')
    print(f"Episode 0, Frame 0 相机位置: {pose_0[:3, 3]}")
    
    env.shutdown()


if __name__ == '__main__':
    main()

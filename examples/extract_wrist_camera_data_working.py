"""
可靠的解决方案：通过钩子函数在每次观测前设置相机位置
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


# 相机配置
CAMERA_OFFSET = {
    'x': 0.06,       # 相对于父对象的X偏移（米）
    'y': 0.0,        # 相对于父对象的Y偏移（米）
    'z': -0.025,     # 相对于父对象的Z偏移（米）
    'alpha': 0.0,    # 绕X轴旋转（度）
    'beta': 0.0,     # 绕Y轴旋转（度）
    'gamma': 0.0     # 绕Z轴旋转（度）
}


class CameraPositionManager:
    """管理相机位置，确保在整个demo采集过程中保持"""
    
    def __init__(self, wrist_cam: VisionSensor, config: dict):
        self.wrist_cam = wrist_cam
        self.config = config
        self.parent = wrist_cam.get_parent()
        self.original_pos = None
        self.original_orient = None
        
    def save_original(self):
        """保存原始相机位置"""
        self.original_pos = self.wrist_cam.get_position(relative_to=self.parent)
        self.original_orient = self.wrist_cam.get_orientation(relative_to=self.parent)
        print(f"原始相机位置: {self.original_pos}")
        print(f"原始相机朝向: {np.rad2deg(self.original_orient)} 度")
        
    def apply_offset(self):
        """应用相机偏移"""
        if self.original_pos is None:
            self.save_original()
        
        # 计算新位置
        new_pos = [
            0.06,
            0.0,
            -0.05
        ]
        
        # 计算新朝向
        new_orient = [
            self.original_orient[0] + np.deg2rad(self.config['alpha']),
            self.original_orient[1] + np.deg2rad(self.config['beta']),
            self.original_orient[2] + np.deg2rad(self.config['gamma'])
        ]
        
        self.wrist_cam.set_position(new_pos, relative_to=self.parent)
        self.wrist_cam.set_orientation(new_orient, relative_to=self.parent)
        
        return new_pos, new_orient
    
    def restore_original(self):
        """恢复原始相机位置"""
        if self.original_pos is not None:
            self.wrist_cam.set_position(self.original_pos, relative_to=self.parent)
            self.wrist_cam.set_orientation(self.original_orient, relative_to=self.parent)


def save_episode_data(episode_idx, demo, save_dir, task_name):
    """
    Save RGB, depth, pose, and intrinsic data for one episode
    """
    task_name += ""
    episode_dir = os.path.join(save_dir, task_name, f'episode_{episode_idx:04d}')
    
    # Create subdirectories
    images_dir = os.path.join(episode_dir, 'images')
    depth_dir = os.path.join(episode_dir, 'depth')
    pose_dir = os.path.join(episode_dir, 'pose')
    intrinsic_dir = os.path.join(episode_dir, 'intrinsic')
    
    for dir_path in [images_dir, depth_dir, pose_dir, intrinsic_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process each frame in the demo
    for frame_idx, obs in enumerate(demo):
        # Extract wrist camera RGB (640x480)
        rgb = obs.wrist_rgb
        # Save RGB image
        rgb_img = Image.fromarray(rgb.astype(np.uint8))
        rgb_img.save(os.path.join(images_dir, f'frame_{frame_idx:04d}.png'))
        
        # Extract wrist camera depth (640x480)
        depth = obs.wrist_depth
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
        'camera_offset': CAMERA_OFFSET
    }
    with open(os.path.join(episode_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Episode {episode_idx} saved: {len(demo)} frames")


def main():
    # Configuration
    num_episodes = 10
    save_dir = 'datasets_working_cam'
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("相机偏移配置:")
    for key, value in CAMERA_OFFSET.items():
        unit = "米" if key in ['x', 'y', 'z'] else "度"
        print(f"  {key}: {value} {unit}")
    print("="*60)
    
    # Setup observation config
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.wrist_camera.set_all(True)
    obs_config.wrist_camera.image_size = (640, 480)
    obs_config.wrist_camera.rgb = True
    obs_config.wrist_camera.depth_in_meters = True
    obs_config.wrist_camera.depth = True
    obs_config.wrist_camera.mask = False
    obs_config.wrist_camera.point_cloud = True
    obs_config.record_gripper_closing = True
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
        headless=True,
    )
    env.launch()
    
    # Get camera and setup manager
    wrist_cam = VisionSensor('cam_wrist')
    cam_manager = CameraPositionManager(wrist_cam, CAMERA_OFFSET)
    
    # Get task
    task = env.get_task(BasketballInHoop)
    task_name = task.get_name()
    
    print(f"\nCollecting {num_episodes} episodes...")
    print(f"Saving to: {save_dir}\n")
    
    # 保存原始相机位置
    cam_manager.save_original()
    
    # Collect demonstrations with custom callback
    for episode_idx in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode_idx + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # 定义回调函数，在每一步都确保相机位置正确
        def step_callback(obs):
            # 在每一步都重新应用相机偏移
            cam_manager.apply_offset()
        
        try:
            # 使用get_demos with callback
            demos = task.get_demos(
                1, 
                live_demos=True,
                callable_each_step=step_callback
            )
            demo = demos[0]
            
            # Save the episode data
            save_episode_data(episode_idx, demo, save_dir, task_name)
            
        except Exception as e:
            print(f"✗ Failed to collect episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Data collection complete!")
    print(f"Total episodes attempted: {num_episodes}")
    print(f"Data saved to: {os.path.abspath(save_dir)}")
    print(f"{'='*60}")
    
    # 恢复原始相机位置
    cam_manager.restore_original()
    
    env.shutdown()


if __name__ == '__main__':
    main()

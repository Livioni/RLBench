"""
批量采集所有RLBench任务的腕关节相机数据
根据 extract_wrist_camera_data_working.py 修改，支持所有任务的批量采集
"""
import numpy as np
import os
from PIL import Image
import json
import importlib
import inspect

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

# 检查PyRep是否可用
try:
    from pyrep.objects.vision_sensor import VisionSensor
    PYREP_AVAILABLE = True
except ImportError:
    print("警告: PyRep 不可用。相机位置管理功能将被禁用。")
    PYREP_AVAILABLE = False
    VisionSensor = None


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

    def __init__(self, wrist_cam, config: dict):
        self.wrist_cam = wrist_cam
        self.config = config
        self.pyrep_available = PYREP_AVAILABLE
        if self.pyrep_available:
            self.parent = wrist_cam.get_parent()
        self.original_pos = None
        self.original_orient = None

    def save_original(self):
        """保存原始相机位置"""
        if not self.pyrep_available:
            print("PyRep不可用，跳过相机位置保存")
            return

        self.original_pos = self.wrist_cam.get_position(relative_to=self.parent)
        self.original_orient = self.wrist_cam.get_orientation(relative_to=self.parent)
        print(f"原始相机位置: {self.original_pos}")
        print(f"原始相机朝向: {np.rad2deg(self.original_orient)} 度")

    def apply_offset(self):
        """应用相机偏移"""
        if not self.pyrep_available:
            return None, None

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
        if not self.pyrep_available or self.original_pos is None:
            return

        self.wrist_cam.set_position(self.original_pos, relative_to=self.parent)
        self.wrist_cam.set_orientation(self.original_orient, relative_to=self.parent)


def save_episode_data(episode_idx, demo, save_dir, task_name):
    """
    Save RGB, depth, pose, and intrinsic data for one episode
    """
    task_name += ""
    episode_dir = os.path.join(save_dir, task_name, f'episode_{episode_idx:04d}')
    print("saved to ", episode_dir)
    
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
        # Save depth as numpy array (for precision)
        np.save(os.path.join(depth_dir, f'frame_{frame_idx:04d}.npy'), depth)
        
        # Extract wrist camera pose (4x4 transformation matrix)
        pose = obs.misc['wrist_camera_extrinsics']
        np.save(os.path.join(pose_dir, f'frame_{frame_idx:04d}.npy'), pose)
        
        # Extract camera intrinsic parameters
        intrinsics = obs.misc['wrist_camera_intrinsics']
        # Fix negative focal lengths (convert from OpenGL/CoppeliaSim coordinate system to standard camera coordinates)
        intrinsics_fixed = intrinsics.copy()
        # intrinsics_fixed[0, 0] = abs(intrinsics[0, 0])  # fx should be positive
        # intrinsics_fixed[1, 1] = abs(intrinsics[1, 1])  # fy should be positive
        np.save(os.path.join(intrinsic_dir, f'frame_{frame_idx:04d}.npy'), intrinsics_fixed)
    
    # Save episode metadata
    metadata = {
        'episode_idx': episode_idx,
        'num_frames': len(demo),
        'task': task_name,
        'camera_offset': CAMERA_OFFSET
    }
    with open(os.path.join(episode_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Episode {episode_idx} saved: {len(demo)} frames")


def get_all_task_classes():
    """
    获取所有RLBench任务类的列表
    通过直接导入所有已知任务类来避免动态导入问题
    """
    # 已知的所有任务类列表（从rlbench/tasks/__init__.py中提取）
    task_classes = [
        BasketballInHoop, BeatTheBuzz, BlockPyramid, ChangeChannel, ChangeClock,
        CloseBox, CloseDoor, CloseDrawer, CloseFridge, CloseGrill, CloseJar,
        CloseLaptopLid, CloseMicrowave, EmptyContainer, EmptyDishwasher,
        GetIceFromFridge, HangFrameOnHanger, HitBallWithQueue, Hockey,
        InsertOntoSquarePeg, InsertUsbInComputer, LampOff, LampOn,
        LiftNumberedBlock, LightBulbIn, LightBulbOut, MeatOffGrill, MeatOnGrill,
        MoveHanger, OpenBox, OpenDoor, OpenDrawer, OpenFridge, OpenGrill,
        OpenJar, OpenMicrowave, OpenOven, OpenWashingMachine, OpenWindow,
        OpenWineBottle, PhoneOnBase, PickAndLift, PickAndLiftSmall, PickUpCup,
        PlaceCups, PlaceHangerOnRack, PlaceShapeInShapeSorter, PlayJenga,
        PlugChargerInPowerSupply, PourFromCupToCup, PressSwitch, PushButton,
        PushButtons, PutAllGroceriesInCupboard, PutBooksOnBookshelf,
        PutBottleInFridge, PutGroceriesInCupboard, PutItemInDrawer,
        PutKnifeInKnifeBlock, PutKnifeOnChoppingBoard, PutMoneyInSafe,
        PutPlateInColoredDishRack, PutRubbishInBin, PutShoesInBox,
        PutToiletRollOnStand, PutTrayInOven, PutUmbrellaInUmbrellaStand,
        ReachAndDrag, ReachTarget, RemoveCups, ScoopWithSpatula, ScrewNail,
        SetTheTable, SetupCheckers, SetupChess, SlideBlockToTarget,
        SlideCabinetOpen, SlideCabinetOpenAndPlaceCups, SolvePuzzle, StackBlocks,
        StackChairs, StackCups, StackWine, StraightenRope, SweepToDustpan,
        TakeCupOutFromCabinet, TakeFrameOffHanger, TakeItemOutOfDrawer,
        TakeLidOffSaucepan, TakeMoneyOutSafe, TakeOffWeighingScales,
        TakePlateOffColoredDishRack, TakeShoesOutOfBox, TakeToiletRollOffStand,
        TakeTrayOutOfOven, TakeUmbrellaOutOfUmbrellaStand, TakeUsbOutOfComputer,
        ToiletSeatDown, ToiletSeatUp, TurnOvenOn, TurnTap, TvOn, UnplugCharger,
        WaterPlants, WeighingScales, WipeDesk
    ]

    # 按照任务名称排序
    task_classes.sort(key=lambda x: x.__name__)
    return task_classes


def main():
    # Configuration
    episodes_per_task = 10  # Number of episodes to collect per task
    save_dir = 'datasets'
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

    # Get camera and setup manager (if PyRep available)
    wrist_cam = VisionSensor('cam_wrist')
    cam_manager = CameraPositionManager(wrist_cam, CAMERA_OFFSET)

    # Get all task classes
    task_classes = get_all_task_classes()
    print(f"\n找到 {len(task_classes)} 个任务类")

    total_tasks = len(task_classes)
    total_episodes = total_tasks * episodes_per_task

    print(f"\n开始批量数据采集...")
    print(f"每个任务采集 {episodes_per_task} 个episodes")
    print(f"总任务数: {total_tasks}")
    print(f"预计总episodes数: {total_episodes}")
    print(f"保存目录: {save_dir}\n")

    # 保存原始相机位置
    cam_manager.save_original()

    task_success_count = 0
    total_episodes_collected = 0

    # 遍历所有任务
    for task_idx, task_class in enumerate(task_classes):
        task_name = task_class.__name__
        print(f"\n{'='*80}")
        print(f"任务 {task_idx + 1}/{total_tasks}: {task_name}")
        print(f"{'='*80}")

        task_success = True
        task_episodes_collected = 0

        try:
            # 获取任务实例
            task = env.get_task(task_class)
            actual_task_name = task.get_name()

            # 为每个任务采集指定数量的episodes
            for episode_idx in range(episodes_per_task):
                print(f"\n  Episode {episode_idx + 1}/{episodes_per_task} - 任务: {actual_task_name}")

                try:
                    # 定义回调函数，在每一步都确保相机位置正确
                    def step_callback(obs):
                        if PYREP_AVAILABLE:
                            # 在每一步都重新应用相机偏移
                            cam_manager.apply_offset()

                    # 使用get_demos with callback
                    demos = task.get_demos(
                        1,
                        live_demos=True,
                        callable_each_step=step_callback
                    )
                    demo = demos[0]

                    # Save the episode data
                    save_episode_data(episode_idx, demo, save_dir, actual_task_name)

                    task_episodes_collected += 1
                    total_episodes_collected += 1
                    print(f"  ✓ Episode {episode_idx} 保存成功: {len(demo)} 帧")

                except Exception as e:
                    print(f"  ✗ Episode {episode_idx} 采集失败: {e}")
                    continue

        except Exception as e:
            print(f"✗ 任务 {task_name} 初始化失败: {e}")
            task_success = False
            import traceback
            traceback.print_exc()
            continue

        if task_success:
            task_success_count += 1
            print(f"\n✓ 任务 {task_name} 完成: {task_episodes_collected}/{episodes_per_task} episodes")

    print(f"\n{'='*80}")
    print(f"批量数据采集完成!")
    print(f"成功任务数: {task_success_count}/{total_tasks}")
    print(f"总采集episodes数: {total_episodes_collected}/{total_episodes}")
    print(f"数据保存目录: {os.path.abspath(save_dir)}")
    print(f"{'='*80}")

    # 恢复原始相机位置
    cam_manager.restore_original()

    env.shutdown()


if __name__ == '__main__':
    main()

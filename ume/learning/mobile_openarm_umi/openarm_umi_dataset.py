import torch
from torch.utils.data import Dataset
from ume.learning.mobile_openarm_ume.lowdim_dataset import LowDimDataset
from ume.learning.video_mp4_dataset import VideoMp4Dataset

import cv2
import numpy as np
from ume.tools.fisheye_camera import jieruiweitong_fisheye_center_crop
from ume.learning.mobile_openarm_umi.openarm_ik_solver import OpenArmIKSolver


def get_timestamps(timestamp, horizon=16, duration=2.0, backward=False):
    """
    Generate a list of timestamps relative to timestamp.

    backward=False (action): [t, t+dt, ..., t+(horizon-1)*dt]
    backward=True  (obs):    [t-(horizon-1)*dt, ..., t-dt, t]

    Args:
        timestamp: Reference timestamp in seconds (current timestep)
        horizon: Number of steps to generate
        duration: Total time span covered in seconds (dt = duration / horizon)
        backward: If True, go backward in time; if False, go forward from current step

    Returns:
        List of timestamps of length horizon, always in ascending order
    """
    dt = duration / horizon
    if backward:
        return [timestamp - (horizon - 1 - i) * dt for i in range(horizon)]
    else:
        return [timestamp + i * dt for i in range(horizon)]


class OpenArmUMIDataset(Dataset):
    def __init__(
        self, 
        data_root, 
        camera_names, 
        action_horizon=20, 
        action_duration=2.0,
        obs_horizon=2,
        obs_duration=1.0):
        super().__init__()
        self.data_root = data_root
        self.camera_names = camera_names
        self.action_horizon = action_horizon
        self.action_duration = action_duration
        self.obs_horizon = obs_horizon
        self.obs_duration = obs_duration
        self.reference_camera_idx = 0 # we will use the first camera as the reference for timestamps and indexing
        self.video_dataset = VideoMp4Dataset(folder_path=data_root, camera_names=camera_names)
        self.lowdim_dataset = LowDimDataset(folder_path=data_root)
        self.reference_timestamps = []
        self.umi_openarm_pin_solver = OpenArmIKSolver()
        for traj in self.video_dataset.traj_paths:
            ref_cam = self.camera_names[self.reference_camera_idx]
            self.reference_timestamps.append(self.video_dataset.get_video_timestamps(traj, ref_cam))

        # compute the length of the dataset as the total number of frames across all trajectories
        self.length = sum(len(timestamps) for timestamps in self.reference_timestamps)
        print(f"dataset initialized with {len(self.video_dataset.traj_paths)} trajectories and {self.length} samples in total.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # given the idx, we need to figure out which trajectory and which timestamp it corresponds to
        for i, timestamps in enumerate(self.reference_timestamps):
            if idx < len(timestamps):
                traj_path = self.video_dataset.traj_paths[i]
                timestamp = timestamps[idx]
                break
            else:
                idx -= len(timestamps)
        
        data = {}
        data['obs'] = {}
        data['action'] = {}

        # print(f"loading trajectory {traj_path} at timestamp {timestamp} (idx {idx})")
        camera_timestamps = get_timestamps(timestamp, horizon=self.obs_horizon, duration=self.obs_duration, backward=True)
        camera_names_to_frames, _ = self.video_dataset.seek_and_decode_multiple_timestamps(traj_path, camera_timestamps)
        for cam_name in camera_names_to_frames:  # fisheye center crop
            frames = camera_names_to_frames[cam_name]  # (T, H, W, 3)
            frames = jieruiweitong_fisheye_center_crop(frames)
            frames = np.stack([cv2.resize(f, (224, 224)) for f in frames], axis=0)  # (T, 224, 224, 3)
            camera_names_to_frames[cam_name] = np.moveaxis(frames, -1, 1).astype(np.float32) / 255.0  # (T, 3, 224, 224)
        data['obs'].update(camera_names_to_frames)
        # joint_timestamps = get_timestamps(timestamp, horizon=self.obs_horizon, duration=self.obs_duration, backward=True)
        # left_actual = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_left_state:actual_angle_rad", timestamp_name="openarm_follower_left_state:timestamp", timestamps=joint_timestamps)
        # right_actual = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_right_state:actual_angle_rad", timestamp_name="openarm_follower_right_state:timestamp", timestamps=joint_timestamps)
        # data['obs']['joint_state'] = np.concatenate([left_actual, right_actual], axis=1).astype(np.float32)

        action_timestamps = get_timestamps(timestamp, horizon=self.action_horizon, duration=self.action_duration, backward=False)
        # print the action timestamps and the range of the follower_left_state_timestamp for debugging
        # print(f"action_timestamps: {action_timestamps}")
        # print(f"range in follower_left_state_timestamp: {self.lowdim_dataset.data_arrays[traj_path]['follower_left_state_timestamp'].min()}, max: {self.lowdim_dataset.data_arrays[traj_path]['follower_left_state_timestamp'].max()}")

        # umi baseline: compute relative pose
        left_action_qpos = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_left_state:actual_angle_rad", timestamp_name="openarm_follower_left_state:timestamp", timestamps=action_timestamps)
        right_action_qpos = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_right_state:actual_angle_rad", timestamp_name="openarm_follower_right_state:timestamp", timestamps=action_timestamps)
        
        debug_action_qpos = []
        left_action_poses = []
        right_action_poses = []
        q_zeros2 = np.zeros(2)
        for i in range(len(action_timestamps)):
            qpos = np.concatenate([left_action_qpos[i][:7], q_zeros2, right_action_qpos[i][:7], q_zeros2], axis=0)
            debug_action_qpos.append(qpos)
            self.umi_openarm_pin_solver.forward_kinematics(qpos)
            left_action_poses.append(self.umi_openarm_pin_solver.get_ee_L_pose())
            right_action_poses.append(self.umi_openarm_pin_solver.get_ee_R_pose())

        left_action_poses = np.stack(left_action_poses)
        right_action_poses = np.stack(right_action_poses)
        
        left_action_gripper_qpos = left_action_qpos[:, 7:8]
        right_action_gripper_qpos = right_action_qpos[:, 7:8]

        base_action = self.lowdim_dataset.seek_and_decode(traj_path, data_name="hexfellow_base_imu_teleop:desired_vel_xyt", timestamp_name="hexfellow_base_imu_teleop:timestamp", timestamps=action_timestamps)
        data['action_base'] = base_action.astype(np.float32)
        data['debug_action_qpos'] = np.stack(debug_action_qpos).astype(np.float32)
        data['obs_left_in_right_pose'] = np.linalg.inv(right_action_poses[0]) @ left_action_poses[0:1].astype(np.float32)
        data['obs_right_in_left_pose'] = np.linalg.inv(left_action_poses[0]) @ right_action_poses[0:1].astype(np.float32)
        data['action_left_hand'] = left_action_poses.astype(np.float32)
        data['action_right_hand'] = right_action_poses.astype(np.float32)
        data['action_left_hand_relative'] = np.linalg.inv(left_action_poses[0]) @ left_action_poses.astype(np.float32)
        data['action_right_hand_relative'] = np.linalg.inv(right_action_poses[0]) @ right_action_poses.astype(np.float32)
        data['action_gripper'] = np.concatenate([left_action_gripper_qpos, right_action_gripper_qpos], axis=1).astype(np.float32)
        return data

if __name__ == "__main__":
    dataset = OpenArmUMIDataset(data_root="data/2026_05_18_fridge_grasp_drink", camera_names=["camera_wrist_left", "camera_head", "camera_wrist_right"], obs_horizon=1)
    data = dataset[100]
    print("Observation shapes:")
    for k, v in data['obs'].items():
        print(f"    {k}: {v.shape}")
    print(f"Action shape base      : {data['action_base'].shape}")
    print(f"Action shape left_hand : {data['action_left_hand'].shape}")
    print(f"Action shape right_hand: {data['action_right_hand'].shape}")
    print(f"Action shape gripper   : {data['action_gripper'].shape}")


    from ume.sim.mujoco_env import MujocoBaseEnv, mujoco, mink
    from dm_control import mjcf
    from ume.robot.openarm1.models import OPENARM_MODEL_PATH
    class Sim(MujocoBaseEnv):
        
        def construct_model(self):
            
            self.robot_model = mjcf.from_path(OPENARM_MODEL_PATH)
            
            root = self.robot_model
            self.add_coordinate_frame_mocap(root, "world_origin", size=0.05)
            self.add_coordinate_frame_mocap(root, "R_ee", size=0.05)
            self.add_coordinate_frame_mocap(root, "L_ee", size=0.05)
            for i in range(20):
                self.add_coordinate_frame_mocap(root, f"R_ee_action_{i}", size=0.025)
                self.add_coordinate_frame_mocap(root, f"L_ee_action_{i}", size=0.025)

            model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())
            configuration = mink.Configuration(model)
            return model, configuration
    
    sim = Sim()
    sim_viewer = sim.launch_viewer()
    
    for data in dataset:
        cv2.imshow("camera_head", cv2.cvtColor(data["obs"]["camera_head"][0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        left_hand_poses = data['action_left_hand']
        right_hand_poses = data['action_right_hand']
        left_hand_rel_poses = data['action_left_hand_relative']
        right_hand_rel_poses = data['action_right_hand_relative']
        debug_action_qpos = data['debug_action_qpos']
        sim.set_qpos(debug_action_qpos[0])
        sim.set_mocap_pose("L_ee", left_hand_poses[0])
        sim.set_mocap_pose("R_ee", right_hand_poses[0])
        for i in range(20):
            sim.set_mocap_pose(f"L_ee_action_{i}", left_hand_poses[0] @ left_hand_rel_poses[i])
            sim.set_mocap_pose(f"R_ee_action_{i}", right_hand_poses[0] @ right_hand_rel_poses[i])
        sim.update()
        sim_viewer.sync()

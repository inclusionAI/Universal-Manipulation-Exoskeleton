import torch
from torch.utils.data import Dataset
from ume.learning.mobile_openarm_ume.lowdim_dataset import LowDimDataset
from ume.learning.video_mp4_dataset import VideoMp4Dataset
import cv2
import numpy as np
from ume.tools.fisheye_camera import jieruiweitong_fisheye_center_crop


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


class OpenArmDataset(Dataset):
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
        joint_timestamps = get_timestamps(timestamp, horizon=self.obs_horizon, duration=self.obs_duration, backward=True)
        left_actual = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_left_state:actual_angle_rad", timestamp_name="openarm_follower_left_state:timestamp", timestamps=joint_timestamps)
        # left_desired = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_controller_state:tau_interaction_L", timestamp_name="openarm_follower_controller_state:timestamp", timestamps=joint_timestamps)
        right_actual = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_right_state:actual_angle_rad", timestamp_name="openarm_follower_right_state:timestamp", timestamps=joint_timestamps)
        # right_desired = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_controller_state:tau_interaction_R", timestamp_name="openarm_follower_controller_state:timestamp", timestamps=joint_timestamps)
        data['obs']['joint_state'] = np.concatenate([left_actual, right_actual], axis=1).astype(np.float32)

        action_timestamps = get_timestamps(timestamp, horizon=self.action_horizon, duration=self.action_duration, backward=False)
        # print the action timestamps and the range of the follower_left_state_timestamp for debugging
        # print(f"action_timestamps: {action_timestamps}")
        # print(f"range in follower_left_state_timestamp: {self.lowdim_dataset.data_arrays[traj_path]['follower_left_state_timestamp'].min()}, max: {self.lowdim_dataset.data_arrays[traj_path]['follower_left_state_timestamp'].max()}")

        left_action = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_left_mit_command:q", timestamp_name="openarm_follower_left_mit_command:timestamp", timestamps=action_timestamps)
        right_action = self.lowdim_dataset.seek_and_decode(traj_path, data_name="openarm_follower_right_mit_command:q", timestamp_name="openarm_follower_right_mit_command:timestamp", timestamps=action_timestamps)
        base_action = self.lowdim_dataset.seek_and_decode(traj_path, data_name="hexfellow_base_imu_teleop:desired_vel_xyt", timestamp_name="hexfellow_base_imu_teleop:timestamp", timestamps=action_timestamps)
        data['action'] = np.concatenate([left_action, right_action, base_action], axis=1).astype(np.float32)
        return data

if __name__ == "__main__":
    dataset = OpenArmDataset(data_root="data/2026_04_26_fridge_open_v2_combined", camera_names=["camera_wrist_left", "camera_head", "camera_wrist_right"], obs_horizon=1)
    data = dataset[100]
    print("Observation shapes:")
    for k, v in data['obs'].items():
        print(f"    {k}: {v.shape}")
    print(f"Action shape: {data['action'].shape}")
import os
import time
from ume.tools.video_util import load_image_frames_pts, decode_video_frame_at_pts, decode_video_frames_at_timestamps

class VideoMp4Dataset:
    
    def __init__(self, folder_path, camera_names):
        self.folder_path = folder_path
        self.camera_names = camera_names
        self.traj_paths = [os.path.join(folder_path, traj_name) for traj_name in os.listdir(folder_path)]
        
        self.n_videos = 0
        self.n_frames = 0
        self.video_timestamps = {}
        for traj_path in self.traj_paths:
            # read timestamps for each camera
            camera_to_timestamps = {}
            for camera_name in camera_names:
                video_path = os.path.join(traj_path, f"{camera_name}.mp4")
                video_timestamps = load_image_frames_pts(video_path)
                camera_to_timestamps[camera_name] = video_timestamps
                self.n_videos += 1
                self.n_frames += len(video_timestamps)
            # store the timestamps for this trajectory
            self.video_timestamps[traj_path] = camera_to_timestamps
        
        
        print(f"VideoMp4Dataset initialization completed")
        print(f"    dataset folder: {folder_path}")
        print(f"    camera names: {camera_names}")
        print(f"    found {len(self.traj_paths)} trajectories")
        print(f"    loaded timestamps for {self.n_videos} videos and {self.n_frames} frames in total.")

    def get_video_timestamps(self, traj_path, camera_name):
        return self.video_timestamps[traj_path][camera_name]
    
    def get_reference_timestamps_and_episode_paths(self, camera_name):
        """ helper for torch dataset __len__ and __getitem__ implementation """
        res = [] # list of dict: {timestamp, traj_path}
        for traj_path in self.traj_paths:
            traj_timestamps = self.video_timestamps[traj_path][camera_name]
            for timestamp in traj_timestamps:
                res.append({"timestamp": timestamp, "traj_path": traj_path})
        return res

    def seek_and_decode(self, traj_path, target_pts):
        """
        Given a trajectory path and a target timestamp (in milliseconds), seek to the closest frame in each camera video and decode it.
        Returns a dict mapping camera names to decoded frames, and a dict mapping camera names to the loading times.
        """
        camera_name_to_frames = {}
        camera_name_load_times = {}
        for camera_name in self.camera_names:
            video_path = os.path.join(traj_path, f"{camera_name}.mp4")
            t0 = time.perf_counter()
            frame = decode_video_frame_at_pts(video_path, target_pts)
            t1 = time.perf_counter()
            camera_name_to_frames[camera_name] = frame
            camera_name_load_times[camera_name] = t1 - t0
        return camera_name_to_frames, camera_name_load_times

    def seek_and_decode_multiple_timestamps(self, traj_path, target_timestamps):
        """
        Given a trajectory path and a list of target timestamps (in seconds), seek to the closest
        frames in each camera video and decode them.
        Returns a dict mapping camera names to stacked frame arrays of shape (T, H, W, 3),
        and a dict mapping camera names to the loading times.
        """
        camera_name_to_frames = {}
        camera_name_load_times = {}
        for camera_name in self.camera_names:
            video_path = os.path.join(traj_path, f"{camera_name}.mp4")
            t0 = time.perf_counter()
            frames = decode_video_frames_at_timestamps(video_path, target_timestamps)
            t1 = time.perf_counter()
            camera_name_to_frames[camera_name] = frames
            camera_name_load_times[camera_name] = t1 - t0
        return camera_name_to_frames, camera_name_load_times

if __name__ == "__main__":
    dataset = VideoMp4Dataset(
        folder_path="data/example_openarm_dataset", 
        camera_names=["camera_wrist_left", "camera_head", "camera_wrist_right"])    
    traj_path = dataset.traj_paths[0]
    timestamps = dataset.get_video_timestamps(traj_path, dataset.camera_names[0])
    print(f"First 10 timestamps: {timestamps[:10]}")
    print(f"Min: {timestamps.min()}, Max: {timestamps.max()}")
    print(f"Range: {timestamps.max() - timestamps.min()}")
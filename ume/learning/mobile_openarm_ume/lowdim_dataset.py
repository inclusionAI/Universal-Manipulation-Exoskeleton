import os
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.interpolate import interp1d

class LowDimDataset:
    """
    Dataset class for low-dimensional data (e.g., joint values, proprioception).
    Similar to VideoMp4Dataset but for numpy arrays instead of videos.
    """

    def __init__(self, folder_path: str):
        """
        Args:
            folder_path: Root directory containing trajectory subfolders
        """
        self.folder_path = folder_path
        self.traj_paths = [os.path.join(folder_path, traj_name)
                          for traj_name in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, traj_name))]

        self.n_trajectories = 0
        self.data_arrays = {}
        self.data_names = None  # Will be set from the first trajectory

        for traj_path in self.traj_paths:
            lowdim_dir = os.path.join(traj_path, "low_dim_npys")
            if not os.path.exists(lowdim_dir):
                print(f"Warning: low_dim_npys directory not found for trajectory {traj_path}, skipping.")
                continue

            traj_arrays = {}
            traj_data_names = set()

            # Load all .npy files
            for npy_file in Path(lowdim_dir).glob("*.npy"):
                data_name = npy_file.stem
                data_array = np.load(npy_file)
                traj_arrays[data_name] = data_array
                traj_data_names.add(data_name)

            if not traj_arrays:
                continue

            # Check consistency
            if self.data_names is None:
                self.data_names = traj_data_names
            elif traj_data_names != self.data_names:
                raise ValueError(f"Trajectory {traj_path} has inconsistent data names. "
                                f"Expected: {sorted(self.data_names)}, Found: {sorted(traj_data_names)}")

            self.data_arrays[traj_path] = traj_arrays
            self.n_trajectories += 1

        if self.data_names is not None:
            self.data_names = sorted(list(self.data_names))  # Sort for consistency

        print(f"LowDimDataset initialization completed")
        print(f"    dataset folder: {folder_path}")
        print(f"    loaded data for {self.n_trajectories} trajectories")

    def get_data(self, traj_path: str, data_name: str) -> np.ndarray:
        """Get data array for a specific trajectory and data name."""
        return self.data_arrays[traj_path][data_name]

    def get_all_data(self, traj_path: str) -> Dict[str, np.ndarray]:
        """Get all data arrays for a trajectory."""
        return self.data_arrays[traj_path]

    def seek_and_decode(self, traj_path: str, data_name: str, timestamp_name: str, timestamps: List[float]) -> np.ndarray:
        """
        Interpolate data at specific timestamps using scipy's interp1d.
        Uses the mean timestamp across joints for interpolation.
        
        Args:
            traj_path: Path to trajectory
            data_name: Name of the data field (e.g., 'follower_left_mit_command_q')
            timestamp_name: Name of the timestamp field (e.g., 'follower_left_mit_command_timestamp')
            timestamps: List of timestamps to interpolate at
        
        Returns:
            Interpolated data array at the specified timestamps, shape (len(timestamps), n_joints)
        """
        # Get the data and timestamps
        data_array = self.data_arrays[traj_path][data_name]
        timestamp_array = self.data_arrays[traj_path][timestamp_name]
        
        # Use mean timestamp across joints for interpolation
        # OpenArm motors have separate timestamps for each joint, while the base does not
        mean_timestamps = np.mean(timestamp_array, axis=1) if timestamp_array.ndim > 1 else timestamp_array
        
        # Create interpolator and interpolate at requested timestamps
        f = interp1d(mean_timestamps, data_array, axis=0, kind='linear', bounds_error=False, fill_value=(data_array[0], data_array[-1]))
        return f(timestamps)

    def __repr__(self) -> str:
        if not self.data_arrays:
            return "\nLowDimDataset(empty)"
        
        # Use the first trajectory to get shapes
        first_traj_path = next(iter(self.data_arrays))
        lines = ["\nLowDimDataset:"]
        for data_name in self.data_names:  # Already sorted
            shape = self.data_arrays[first_traj_path][data_name].shape
            lines.append(f"  {data_name}: {shape}")
        return "\n".join(lines) + "\n"

if __name__ == "__main__":
    dataset = LowDimDataset(folder_path="data/example_openarm_dataset")
    print(dataset)
    traj_path = next(iter(dataset.data_arrays))
    timestamps = dataset.data_arrays[traj_path]['follower_left_mit_command_timestamp']
    print(timestamps[:10])  # Print first 10 values
    print(f"Min: {timestamps.min()}, Max: {timestamps.max()}")
    print(f"Range: {timestamps.max() - timestamps.min()}")
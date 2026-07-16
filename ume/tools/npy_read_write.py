import numpy as np
import os

def save_dict_to_npy(data_dict, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for key, array in data_dict.items():
        # Save as standard uncompressed binary
        file_path = os.path.join(folder_path, f"{key}.npy")
        np.save(file_path, array)

def load_dict_from_npy(folder_path, mmap_mode=None):
    """
        use mmap_mode='r' for memory-mapped loading
        use mmap_mode=None for normal loading (which loads the entire array into memory)
    """
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            key = filename[:-4] # Remove .npy extension
            # mmap_mode='r' makes the "load" nearly instant
            data_dict[key] = np.load(os.path.join(folder_path, filename), mmap_mode=mmap_mode)
    return data_dict


import time
import os
import cv2
import numpy as np
import typer
from threadpoolctl import threadpool_limits
from fractions import Fraction

from ume.redis_client import RedisClient
from ume.tools.data_collection_path import get_data_path
from ume.tools.time import get_time_hmstz
from ume.tools.npy_read_write import save_dict_to_npy

import yaml


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    output_dir: str = "data/collecting",
    low_dim_stream_keys: str = "",
    low_dim_stream_dtypes: str = "",
    low_dim_stream_mapping_config_path: str = "",
):
    
    # get stream keys and dtypes from config file or command line arguments
    if low_dim_stream_mapping_config_path != "":
        with open(low_dim_stream_mapping_config_path, "r") as f:
            low_dim_stream_mapping_config = yaml.safe_load(f)
            all_stream_keys = list(low_dim_stream_mapping_config["low_dim"].keys())
            all_stream_dtypes = [value["dtype"] for value in low_dim_stream_mapping_config["low_dim"].values()]
    else:
        all_stream_keys = low_dim_stream_keys.split(",")
        all_stream_dtypes = low_dim_stream_dtypes.split(",")
    
    client = RedisClient(host=redis_host, port=redis_port)
    
    # recording state variables
    recording_state = "idle"
    recording_start_time = None
    recording_path = None
    recording_data: dict[str, list] = dict()
    iter_idx = 0
    
    # recording command loop
    iter_idx = 0
    iter_last_health_update_time = time.time()
    while True:
        
        collect_data_command = client.stream_get_batch({f"collect_data:command": str, f"collect_data:timestamp": float})
        command, timestamp = collect_data_command["collect_data:command"], collect_data_command["collect_data:timestamp"]
        
        if recording_state == "idle":
            # idle -> recording
            if command == "start_recording" and time.time() - timestamp < 0.2:
                recording_state = "recording"
                recording_start_time = timestamp
                print(f"low-dim recorder started recording at {get_time_hmstz(recording_start_time)}")
            
        elif recording_state == "recording":
            # prepare for recording
            recording_path = f"{get_data_path("low_dim_npys", output_dir, recording_start_time)}"
            for key in all_stream_keys:
                recording_data[key] = []
            print(f"low-dim recorder start recording data to {recording_path}")
            
            # recording loop
            last_timestamp: dict[str, float] = {stream_key: recording_start_time for stream_key in all_stream_keys}
            while recording_state == "recording":
                
                # construct redis query
                redis_data_query = {stream_key: eval(dtype_str) for stream_key, dtype_str in zip(all_stream_keys, all_stream_dtypes)}
                redis_after_query = last_timestamp
                command_data_query = {f"collect_data:command": str}
                command_after_query = {f"collect_data:command": recording_start_time}
                
                print(redis_data_query)
                print(redis_after_query)
                print(command_data_query)
                print(command_after_query)
                
                # get data from redis in batch
                data_insertion_timestamps, data_batch = client.stream_get_batch_after(
                    stream_keys=(redis_data_query | command_data_query),
                    timestamps=(redis_after_query | command_after_query)
                )
                
                # if no new data received, wait a bit before asking again
                if len(data_insertion_timestamps) == 0:
                    print(f"low-dim recorder no new data received at {get_time_hmstz(time.time())}, waiting...")
                    time.sleep(0.01)
                    continue
                
                # check for stop command
                if f"collect_data:command" in data_insertion_timestamps and data_batch[f"collect_data:command"][-1] == "stop_recording":
                    # save recorded data to npy file
                    save_dict_to_npy(recording_data, recording_path)
                    
                    recording_state = "idle"
                    recording_start_time = None
                    recording_path = None
                    recording_data: dict[str, list] = dict()
                    
                    print(f"low-dim recorder stopped recording at {get_time_hmstz(time.time())}, saved data to {recording_path}")
                    print("total episodes:", len(os.listdir(output_dir)))
                    break
                
                # append new data to recording_data
                print(data_batch)
                for stream_key in all_stream_keys:
                    if stream_key in data_insertion_timestamps:  # if there is new data received in this iteration
                        stream_data = data_batch[stream_key]
                        recording_data[stream_key].extend(stream_data)
                
                # update last_timestamp for the next query
                for stream_key in all_stream_keys:
                    if stream_key in data_insertion_timestamps:  # if there is new data received in this iteration
                        last_timestamp[stream_key] = data_insertion_timestamps[stream_key][-1]
        
        else:
            raise RuntimeError(f"Unknown state: {recording_state}")

        data_batch = {"low_dim_recorder:state": recording_state}
        data_batch_maxlen = {"low_dim_recorder:state": 100}
        client.stream_add_batch(data_batch, data_batch_maxlen)

        if iter_idx == 0:
            print("low_dim recorder ready")

        if time.time() - iter_last_health_update_time > 1.0:
            print(f"low-dim recorder is healthy, current state: {recording_state} at {get_time_hmstz(time.time())}")
            iter_last_health_update_time = time.time()
        
        iter_idx += 1

if __name__ == "__main__":
    typer.run(main)
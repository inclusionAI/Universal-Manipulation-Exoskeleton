import cv2
import numpy as np
import time
import typer
from ume.redis_client import RedisClient
from ume.tools.keycounter import KeystrokeCounter, Key, KeyCode
from ume.tools.time import get_time_hmstz
from typing import Dict, List, Literal, Optional, Union

def main(
    control_redis_host: str = "localhost",
    control_redis_port: int = 6379,
    perception_redis_host: str = "localhost",
    perception_redis_port: int = 6380,
    recording_camera_names: str = "camera_top,camera_bottom",
    start_recording_key: str = "`",
    stop_recording_key: str = "s"
):
    
    control_redis = RedisClient(host=control_redis_host, port=control_redis_port)
    perception_redis = RedisClient(host=perception_redis_host, port=perception_redis_port)
    recording_camera_names = recording_camera_names.split(",")
    
    data_batch_maxlen = {
        f"collect_data:command": 100,
        f"collect_data:timestamp": 100
    }
    
    # green for idle, red for recording
    recording_indicator = np.zeros((200, 200, 3), dtype=np.uint8)
    recording_indicator[:] = (0, 255, 0)
    
    iter_idx = 0
    with KeystrokeCounter() as counter:
        while True:
            # get camera states and images
            video_recorder_states_query = {f"video_recorder:{name}:state": str for name in recording_camera_names}
            video_stream_image_query = {f"{name}:image": np.ndarray for name in recording_camera_names}
            all_queries = video_recorder_states_query | video_stream_image_query
            
            # get data from redis in batch
            input_data = perception_redis.stream_get_batch(all_queries)
            
            # recorder states, camera stream images
            video_recorder_states = {key: value for key, value in input_data.items() if key in video_recorder_states_query}
            video_images = {key: value for key, value in input_data.items() if key in video_stream_image_query}
            
            # update recording state based on camera states
            data_batch = None
            for press_events in counter.get_press_events():
                if press_events == KeyCode(char=start_recording_key):
                    # if all recorders are idle, command all recorders to start recording
                    if all(state == "idle" for state in video_recorder_states.values()):
                        data_batch = {
                            f"collect_data:command": "start_recording",
                            f"collect_data:timestamp": time.time()
                        }
                        
                        perception_redis.stream_add_batch(data_batch, data_batch_maxlen)
                        control_redis.stream_add_batch(data_batch, data_batch_maxlen)
                        print(f"collect data start recording at {get_time_hmstz(data_batch['collect_data:timestamp'])}")
                    # refuse to start again if any recorder is still recording
                    else:
                        print("already recording, or waiting all recorder to finish recording the current session")
                elif press_events == KeyCode(char=stop_recording_key):
                    # command all recorders to stop recording
                    data_batch = {
                        f"collect_data:command": "stop_recording",
                        f"collect_data:timestamp": time.time()
                    }
                    perception_redis.stream_add_batch(data_batch, data_batch_maxlen)
                    control_redis.stream_add_batch(data_batch, data_batch_maxlen)
                    print(f"collect data stop recording at {get_time_hmstz(data_batch['collect_data:timestamp'])}")
                else:
                    print("unmapped key:", press_events)
            
            # visualize camera states and images
            for camera_name in recording_camera_names:
                if f"{camera_name}:image" in video_images:
                    cv2.imshow(camera_name, video_images[f"{camera_name}:image"])
                    cv2.waitKey(1)
            
            # update recording indicator window
            if data_batch is not None:
                if data_batch["collect_data:command"] == "start_recording":
                    recording_indicator[:] = (0, 0, 255)
                elif data_batch["collect_data:command"] == "stop_recording":
                    recording_indicator[:] = (0, 255, 0)
            cv2.imshow("recording_indicator", recording_indicator)
            cv2.waitKey(1)
            
            if iter_idx == 0:
                print("collect data ready")
            iter_idx += 1
            time.sleep(0.1)


if __name__ == "__main__":
    typer.run(main)

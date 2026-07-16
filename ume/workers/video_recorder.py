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

import av

def main(
    redis_host: str = "localhost",
    redis_port: int = 6380,
    output_dir: str = "data/collecting",
    camera_stream_key: str = "camera_top",
    video_gop_size: int = 10  # Litian: h264 encoding I-frame interval, default to 10 for faster seek during training
):
    
    print(f"video recorder of {camera_stream_key} starting")
    client = RedisClient(host=redis_host, port=redis_port)
    
    # recording state variables
    recording_state = "idle"
    recording_start_time = None
    recording_video_path = None
    
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
                print(f"video recorder of {camera_stream_key} started recording")
        
        elif recording_state == "recording":
            
            # prepare for recording
            recording_frame_idx = 0
            recording_video_path = f"{get_data_path(camera_stream_key, output_dir, recording_start_time)}.mp4"
            os.makedirs(os.path.dirname(recording_video_path), exist_ok=True)
            print(f"video recorder of {camera_stream_key} start recording video to {recording_video_path}")
            
            # create video file, write frames, and check for stop command in a loop
            with av.open(recording_video_path, "w") as container:
                
                # mp4 container setting
                stream = container.add_stream("h264", rate=30)  # Litian: non-constant fps if pts is set
                stream.gop_size = video_gop_size  # controls the h264 encoding I-frame interval, default to 10 for faster seek during training
                
                # recording loop
                last_timestamp = recording_start_time  # last timestamp of the previous chunk of frames
                while recording_state == "recording":
                    
                    # ask redis for frames
                    data_insertion_timestamps, data_batch = client.stream_get_batch_after(
                        stream_keys={
                            f"collect_data:command": str,
                            f"{camera_stream_key}:timestamp": float,
                            f"{camera_stream_key}:image": np.ndarray
                        },
                        timestamps={
                            f"collect_data:command": last_timestamp,
                            f"{camera_stream_key}:timestamp": last_timestamp,
                            f"{camera_stream_key}:image": last_timestamp
                        }
                    )
                    
                    # if no new frames received, wait a bit before asking again
                    if len(data_insertion_timestamps) == 0 or f"{camera_stream_key}:image" not in data_insertion_timestamps:
                        print(f"video recorder of {camera_stream_key} no new frames received at {get_time_hmstz(time.time())}, waiting...")
                        time.sleep(0.01)
                        continue
                    
                    # check for stop command
                    if ("collect_data:command" in data_insertion_timestamps
                        and data_batch["collect_data:command"][-1] == "stop_recording"
                    ):
                        recording_state = "idle"
                        recording_start_time = None
                        recording_video_path = None
                        
                        # flush encoder
                        for packet in stream.encode():
                            # packet.rescale_ts(stream.time_base)
                            container.mux(packet)
                        
                        print(f"video recorder of {camera_stream_key} stopped recording at {get_time_hmstz(time.time())}", "last frame idx:", recording_frame_idx, "last timestamp:", last_timestamp)
                        break
                    
                    # write frames to video file
                    assert len(data_insertion_timestamps[f"{camera_stream_key}:image"]) > 0, "No frames received during recording!"
                    for timestamp, frame in zip(data_batch[f"{camera_stream_key}:timestamp"], data_batch[f"{camera_stream_key}:image"]):
                        
                        # initialize stream settings based on the first frame received
                        if recording_frame_idx == 0:
                            h, w, _ = frame.shape
                            stream.height = h
                            stream.width = w
                            micro_s = Fraction(1, 1000000)
                            stream.time_base = micro_s
                            stream.codec_context.pix_fmt = "yuv420p"
                            stream.codec_context.options = {"crf": "18", "profile": "high"}
                            stream.codec_context.time_base = micro_s  # microsecond precision
                        
                        # convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                        av_frame.pts = int(round(timestamp / stream.codec_context.time_base))  # set presentation timestamp (pts) based on the frame timestamp and stream time base
                        
                        # encode frame and write to container
                        print(f"video recorder of {camera_stream_key} writing frame idx {recording_frame_idx} with pts {av_frame.pts} at timestamp {get_time_hmstz(timestamp)}")
                        for packet in stream.encode(av_frame):
                            container.mux(packet)
                        recording_frame_idx += 1
                    
                    # update last timestamp for the next round of frame retrieval
                    last_timestamp = data_insertion_timestamps[f"{camera_stream_key}:image"][-1]
        else:
            raise RuntimeError(f"Unknown state: {recording_state}")
        
        data_batch = {f"video_recorder:{camera_stream_key}:state": recording_state}
        data_batch_maxlen = {f"video_recorder:{camera_stream_key}:state": 100}
        client.stream_add_batch(data_batch, data_batch_maxlen)
        
        if iter_idx == 0:
            print("video recorder ready")
            
        if time.time() - iter_last_health_update_time > 1.0:
            print(f"video recorder of {camera_stream_key} is healthy, current state: {recording_state} at {get_time_hmstz(time.time())}")
            iter_last_health_update_time = time.time()
        
        iter_idx += 1

if __name__ == "__main__":
    typer.run(main)

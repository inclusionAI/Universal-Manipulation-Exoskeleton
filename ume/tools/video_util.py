import av
import numpy as np
from typing import Dict, List, Literal, Optional, Union, Tuple


def load_image_frames_pts(video_path: str) -> List[np.ndarray]:
    with av.open(video_path, "r") as container:
        in_stream = container.streams.video[0]
        presentation_times = []
        for packet in container.demux(in_stream):
            if packet.pts is not None:
                presentation_times.append(float(packet.pts * in_stream.time_base))  # in seconds
        presentation_times = np.array(sorted(presentation_times)) # packets are not in presentation order
    return presentation_times

def decode_video_frames_at_timestamps(video_path: str, target_timestamps: List[float]) -> np.ndarray:
    frames = []
    with av.open(video_path, "r") as container:
        in_stream = container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = 1
        for target_pts in target_timestamps:
            seek_pts = int(round(target_pts / in_stream.time_base))
            container.seek(seek_pts, backward=True, any_frame=False, stream=in_stream)
            last_abs_delta = 1e6
            decoded_pts = []
            decoded_frames = []
            for frame in container.decode(in_stream):
                frame_pts = float(frame.pts * frame.time_base)
                abs_delta = abs(frame_pts - target_pts)
                decoded_pts.append(frame_pts)
                decoded_frames.append(frame.to_ndarray(format="rgb24"))
                if abs_delta > last_abs_delta:
                    break
                last_abs_delta = abs_delta
            best_idx = np.argmin(np.abs(np.array(decoded_pts) - target_pts))
            frames.append(decoded_frames[best_idx])
            del decoded_frames, decoded_pts
    return np.stack(frames, axis=0)


def decode_video_frame_at_pts(video_path: str, target_pts: float) -> np.ndarray:
    with av.open(video_path, "r") as container:
        in_stream = container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = 1
        # seek to the last keyframe before the required frame
        seek_pts = int(round(target_pts / in_stream.time_base))
        container.seek(seek_pts, backward=True, any_frame=False, stream=in_stream)
        # start decoding frames
        last_abs_delta_pts = 1e6
        decoded_pts = []
        decoded_frames = []
        for frame in container.decode(in_stream):
            frame_pts_in_seconds = float(frame.pts * frame.time_base)
            # print(frame_pts_in_seconds)
            abs_delta_pts = abs(frame_pts_in_seconds - target_pts)
            decoded_pts.append(frame_pts_in_seconds)
            decoded_frames.append(frame.to_ndarray(format="rgb24"))
            if abs_delta_pts > last_abs_delta_pts:
                # passed the target pts
                break
            last_abs_delta_pts = abs_delta_pts
        decoded_pts = np.array(decoded_pts)
        res_frame_idx = np.argmin(np.abs(decoded_pts - target_pts))
        # print(len(decoded_frames), "decoded frames")
        return decoded_frames[res_frame_idx]
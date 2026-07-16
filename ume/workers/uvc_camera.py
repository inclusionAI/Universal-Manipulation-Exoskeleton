import time

import cv2
import typer
from threadpoolctl import threadpool_limits

from ume.redis_client import RedisClient

def main(
    redis_host: str = "localhost",
    redis_port: int = 6380,
    stream_key: str = "camera_top",
    stream_max_duration_s: int = 10,
    
    dev_path: str = "/dev/video0",
    fps: float = 30,
    resolution: str = "640,480",
    brightness: float = None,
    use_yuv: bool = False,
    num_threads: int = 2,
    vis: bool = False,
    buffer_size: int = 2
):

    threadpool_limits(num_threads)
    cv2.setNumThreads(num_threads)
    
    client = RedisClient(host=redis_host, port=redis_port)

    w, h = [int(x) for x in resolution.split(",")]

    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

    if use_yuv:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "V"))
    else:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    cap.set(cv2.CAP_PROP_FPS, fps)

    iter_idx = 0
    frame = None
    last_recv = None
    while True:
        ret, frame = cap.read(frame)
        # print(frame.shape, frame.dtype)
        t_recv = time.time()
        if not ret:
            raise RuntimeError(f"Failed to capture frame from {dev_path}. Make sure OBS is not open.")
        
        mt_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        t_cap = mt_cap - time.monotonic() + time.time()

        # redis push
        data_batch = {
            f"{stream_key}:image": frame,
            f"{stream_key}:timestamp": t_cap
        }
        data_batch_maxlen = {
            f"{stream_key}:image": int(fps * stream_max_duration_s),
            f"{stream_key}:timestamp": int(fps * stream_max_duration_s)
        }
        client.stream_add_batch(data_batch, batch_maxlen=data_batch_maxlen)

        # print stats
        if iter_idx % 10 == 0 and last_recv is not None:
            print(t_cap, "{:.2f} FPS".format(1 / (t_recv - last_recv)), f"{stream_key}:image", "Len:", client.stream_get_len(f"{stream_key}:image"))
        last_recv = t_recv

        # visualize image
        if vis:
            cv2.imshow(f"{stream_key}:image {dev_path}", frame)
            cv2.pollKey()

        if iter_idx == 0:
            print("camera ready")
        iter_idx += 1


if __name__ == "__main__":
    typer.run(main)

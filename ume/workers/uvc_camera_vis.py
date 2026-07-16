import time

import cv2
import typer
import numpy as np
from threadpoolctl import threadpool_limits
from ume.tools.fisheye_camera import jieruiweitong_fisheye_center_crop
from ume.redis_client import RedisClient

def main(
    redis_host: str = "localhost",
    redis_port: int = 6380,
    stream_key: str = "camera_top",
    scale: float = 1.0
):
    client = RedisClient(host=redis_host, port=redis_port)
    while True:
        input_batch = client.stream_get_batch(
            {
                f"{stream_key}:image": np.ndarray
            }
        )
        frame = input_batch[f"{stream_key}:image"]
        frame = jieruiweitong_fisheye_center_crop(frame)
        if scale != 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        cv2.imshow(stream_key, frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    typer.run(main)

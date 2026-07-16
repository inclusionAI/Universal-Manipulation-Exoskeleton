import typer
from ume.redis_client import RedisClient
from ume.robot.ybimu.ybimu_simple import YbImu
from ume.tools.precise_sleep import FrequencyRegulator


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    name: str = "ybimu",
    history_length_s: int = 5,
    frequency: int = 5000
):
    ybimu = YbImu()
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=frequency)
    while True:
        imu_pose = ybimu.get_pose()
        client.stream_add_batch(
            {
                f"{name}_state:imu_pose": imu_pose
            },
            batch_maxlen={
                f"{name}_state:imu_pose": history_length_s * frequency
            }
        )
        reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=f"{name} redis worker {imu_pose}")
        if reg.iter_idx == 1:
            print(f"ybimu redis worker ready")


if __name__ == "__main__":
    typer.run(main)
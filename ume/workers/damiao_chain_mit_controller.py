import time

import can
import numpy as np

from ume.damiao_chain import DamiaoChain, MotorType
from ume.tools.precise_sleep import FrequencyRegulator

import typer
from ume.redis_client import RedisClient

def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    input_name: str = "damiao_chain_command",
    output_name: str = "damiao_chain_state",
    control_frequency_hz: float = 1000,
    can_channel: str = "can0",
    can_fd: bool = True,
    motor_ids: str = "1,2,3,4,5,6,7,8",
    motor_types: str = "DM8009,DM8009,DM4340,DM4340,DM4310,DM4310,DM4310,DM4310",
):
    client = RedisClient(host=redis_host, port=redis_port)
    with DamiaoChain(
        motor_ids=[int(motor_id) for motor_id in motor_ids.split(",")],
        motor_types=[MotorType[motor_type] for motor_type in motor_types.split(",")],
        can_channel=can_channel,
        can_fd=can_fd,
        timeout=0.1
    ) as damiao_chain:

        reg = FrequencyRegulator(frequency=control_frequency_hz)
        q_zeros = np.zeros(len(damiao_chain.motor_ids), dtype=np.float32)
        init_input = {
            f"{input_name}:kp" : q_zeros,
            f"{input_name}:kd" : q_zeros,
            f"{input_name}:q"  : q_zeros,
            f"{input_name}:dq" : q_zeros,
            f"{input_name}:tau": q_zeros,
        }
        init_input_maxlen = {key: 1 for key in init_input.keys()}
        client.stream_add_batch(init_input, init_input_maxlen)

        iter_idx = 0
        while True:
            command_batch = client.stream_get_batch(
                {
                    f"{input_name}:kp" : np.ndarray,
                    f"{input_name}:kd" : np.ndarray,
                    f"{input_name}:q"  : np.ndarray,
                    f"{input_name}:dq" : np.ndarray,
                    f"{input_name}:tau": np.ndarray
                }
            )
            desired_kp = command_batch[f"{input_name}:kp"]
            desired_kd = command_batch[f"{input_name}:kd"]
            desired_q = command_batch[f"{input_name}:q"]
            desired_dq = command_batch[f"{input_name}:dq"]
            desired_tau = command_batch[f"{input_name}:tau"]
            
            try:
                data_dict = damiao_chain.mit_control(
                    kp=desired_kp,
                    kd=desired_kd,
                    q=desired_q,
                    dq=desired_dq,
                    tau=desired_tau
                )
            except TimeoutError as e:
                print("TimeoutError during mit_control:", e)
                damiao_chain.__enter__()  # Litian HACK: clear error and re-enable motors
                continue
            
            data_batch = {f"{output_name}:{key}": value for key, value in data_dict.items()}
            data_batch_maxlen = {f"{output_name}:{key}": int(control_frequency_hz * 10) for key in data_dict.keys()}  # Litian HACK: store 10 seconds of data
            client.stream_add_batch(data_batch, data_batch_maxlen)
            
            reg.sleep(verbose=True, verbose_interval=1, verbose_prefix="damiao chain mit control:")
            iter_idx += 1
            
            if iter_idx == 1:
                print("damiao chain mit controller ready")


if __name__ == "__main__":
    typer.run(main)

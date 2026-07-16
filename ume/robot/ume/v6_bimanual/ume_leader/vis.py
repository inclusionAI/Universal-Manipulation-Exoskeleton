import numpy as np

import typer
from ume.redis_client import RedisClient
from ume.tools.precise_sleep import FrequencyRegulator

import scipy.spatial.transform as st
from ume.robot.ume.v6_bimanual.ume_mujoco_env import UMEPlayEnv


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    ume_controller_name: str = "ume_leader_controller"
):
    
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=30)
    
    # mujoco sim
    sim = UMEPlayEnv()
    sim_viewer = sim.launch_viewer()
    
    iter_idx = 0
    while True:
        data_batch = client.stream_get_batch(
            {
                f"{ume_controller_name}_state:mujoco_qpos": np.ndarray,
                
                f"{ume_controller_name}_state:desired_tx_base_R_shoulder": np.ndarray,
                f"{ume_controller_name}_state:desired_tx_base_R_wrist": np.ndarray,
                
                f"{ume_controller_name}_state:desired_tx_base_L_shoulder": np.ndarray,
                f"{ume_controller_name}_state:desired_tx_base_L_wrist": np.ndarray,
            }
        )
        # ume solver state
        mujoco_qpos = data_batch[f"{ume_controller_name}_state:mujoco_qpos"]
        tx_base_R_shoulder = data_batch[f"{ume_controller_name}_state:desired_tx_base_R_shoulder"]
        tx_base_R_wrist = data_batch[f"{ume_controller_name}_state:desired_tx_base_R_wrist"]
        tx_base_L_shoulder = data_batch[f"{ume_controller_name}_state:desired_tx_base_L_shoulder"]
        tx_base_L_wrist = data_batch[f"{ume_controller_name}_state:desired_tx_base_L_wrist"]
        
        # visualize frames
        sim.set_qpos(mujoco_qpos)
        sim.set_mocap_pose("world_origin", np.eye(4))
        sim.set_mocap_pose("R_shoulder", tx_base_R_shoulder)
        sim.set_mocap_pose("R_wrist", tx_base_R_wrist)
        sim.set_mocap_pose("L_shoulder", tx_base_L_shoulder)
        sim.set_mocap_pose("L_wrist", tx_base_L_wrist)
        sim.update()
        sim_viewer.sync()

        # regulate frequency
        reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=f"ume visualization")
        iter_idx += 1
        
        if iter_idx == 1:
            print("ume vis ready")


if __name__ == "__main__":
    typer.run(main)

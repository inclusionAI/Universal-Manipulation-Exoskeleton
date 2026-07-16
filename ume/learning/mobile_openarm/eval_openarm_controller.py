import time

import can
import numpy as np

import typer
from ume.redis_client import RedisClient
from ume.tools.precise_sleep import FrequencyRegulator

from ume.robot.openarm1.solver import OpenArm1Solver
from ume.tools.keycounter import KeystrokeCounter, Key, KeyCode
from ume.tools.pose_interp import get_interp1d
from ume.tools.npy_read_write import save_dict_to_npy, load_dict_from_npy

FOLLOWER_KP = np.array([240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0])
FOLLOWER_KD = np.array([3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])

def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    follower_left: str = "follower_left",
    follower_right: str = "follower_right",
    frequency_hz: float = 2000,
    initial_arm_state_reference_episode_lowdim_path: str = "data/initial_pose_default/low_dim_npys",
    model_inference_worker_name: str = "model_inference",
    chunk_renew_every_s: float = 2.0,
    name: str = "openarm_controller",
    stitch_duration: float = 0.45
):
    
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=frequency_hz)
    solver = OpenArm1Solver()
    q_zeros = np.zeros(8)
    
    reference_episode = load_dict_from_npy(initial_arm_state_reference_episode_lowdim_path)
    reference_qpos_L = reference_episode[f"openarm_follower_left_state:actual_angle_rad"][0]
    reference_qpos_R = reference_episode[f"openarm_follower_right_state:actual_angle_rad"][0]
    print(f"Reference initial qpos_L: {reference_qpos_L.shape}, qpos_R: {reference_qpos_R.shape}")
    
    eval_state = "idle" # idle, enabling, evaluating
    qpos_interp_L = None
    qpos_interp_R = None
    
    with KeystrokeCounter() as key_counter:
    
        iter_idx = 0
        while True:
            data_batch = client.stream_get_batch(
                {
                    f"{follower_left}_state:actual_angle_rad": np.ndarray,
                    f"{follower_left}_state:actual_velocity_radps": np.ndarray,
                    f"{follower_left}_state:actual_torque_nm": np.ndarray,

                    f"{follower_right}_state:actual_angle_rad": np.ndarray,
                    f"{follower_right}_state:actual_velocity_radps": np.ndarray,
                    f"{follower_right}_state:actual_torque_nm": np.ndarray,
                    
                    f"{model_inference_worker_name}:action_timestamp": np.ndarray,
                    f"{model_inference_worker_name}:action_desired_qpos_left": np.ndarray,
                    f"{model_inference_worker_name}:action_desired_qpos_right": np.ndarray
                }
            )
            
            # model inference state
            model_action_timestamp = data_batch[f"{model_inference_worker_name}:action_timestamp"]
            model_desired_qpos_left = data_batch[f"{model_inference_worker_name}:action_desired_qpos_left"]
            model_desired_qpos_right = data_batch[f"{model_inference_worker_name}:action_desired_qpos_right"]
            
            # robot state
            qpos_L = data_batch[f"{follower_left}_state:actual_angle_rad"]
            qpos_R = data_batch[f"{follower_right}_state:actual_angle_rad"]
            sim_qpos_L = np.concatenate([qpos_L[:7], [0, 0]])
            sim_qpos_R = np.concatenate([qpos_R[:7], [0, 0]])
            sim_qpos = np.concatenate([sim_qpos_L, sim_qpos_R])
            
            qvel_L = data_batch[f"{follower_left}_state:actual_velocity_radps"]
            qvel_R = data_batch[f"{follower_right}_state:actual_velocity_radps"]
            sim_qvel_L = np.concatenate([qvel_L[:7], [0, 0]])
            sim_qvel_R = np.concatenate([qvel_R[:7], [0, 0]])
            sim_qvel = np.concatenate([sim_qvel_L, sim_qvel_R])
            
            solver.forward_kinematics(sim_qpos)
            
            desired_qacc = np.zeros_like(sim_qpos)
            tau = solver.get_gravity_compensation_torques(sim_qpos, sim_qvel, desired_qacc)
            desired_tau_L = np.concatenate([tau[0:7], [0.0]])
            desired_tau_R = np.concatenate([tau[8:15], [0.0]])
            
            # compute interaction torque
            tau_actual_L = data_batch[f"{follower_left}_state:actual_torque_nm"]
            tau_actual_R = data_batch[f"{follower_right}_state:actual_torque_nm"]
            tau_interaction_L = tau_actual_L - desired_tau_L
            tau_interaction_R = tau_actual_R - desired_tau_R
            
            desired_timestamp = time.time()
            for press_events in key_counter.get_press_events():
                if press_events == KeyCode(char='b'):
                    eval_state = "enabling"
                    enabling_duration = 1.5
                    enabling_timestamps = np.array([desired_timestamp, desired_timestamp + enabling_duration])
                    qpos_L_start_end = np.stack([qpos_L, reference_qpos_L], axis=0)
                    qpos_R_start_end = np.stack([qpos_R, reference_qpos_R], axis=0)
                    qpos_interp_L = get_interp1d(enabling_timestamps, qpos_L_start_end)
                    qpos_interp_R = get_interp1d(enabling_timestamps, qpos_R_start_end)
                    print("Enabling... Interpolating from current position to reference initial position over 5 seconds.")
                elif press_events == KeyCode(char='`'):
                    if model_action_timestamp is not None:
                        eval_state = "evaluating"
                        qpos_interp_L = get_interp1d(model_action_timestamp, model_desired_qpos_left)
                        qpos_interp_R = get_interp1d(model_action_timestamp, model_desired_qpos_right)
                        print("Evaluating... Interpolating desired qpos from model inference.")
                    else:
                        print("No model inference data received yet. Cannot start evaluating.")
                elif press_events == KeyCode(char='s'):
                    eval_state = "idle"
                    qpos_interp_L = None
                    qpos_interp_R = None
                    print("Stopping. Returning to idle state.")
            
            if eval_state == "evaluating" and t_now > qpos_interp_L.x[0] + chunk_renew_every_s + stitch_duration:
                # if we are evaluating and close to the end of the model predicted trajectory
                # recompute the interpolator with latest model output
                # qpos_interp_L = get_interp1d(model_action_timestamp, model_desired_qpos_left)
                # qpos_interp_R = get_interp1d(model_action_timestamp, model_desired_qpos_right)
                # last interp end time and qpos
                last_interp_end_time = qpos_interp_L.x[0] + chunk_renew_every_s + stitch_duration
                last_interp_end_qpos_L = qpos_interp_L(last_interp_end_time)
                last_interp_end_qpos_R = qpos_interp_R(last_interp_end_time)
                
                # Litian HACK: stich 2 trajectories
                # stitch_duration = 0.0
                # qpos_interp_L = get_interp1d(model_action_timestamp, model_desired_qpos_left)
                # qpos_interp_R = get_interp1d(model_action_timestamp, model_desired_qpos_right)
                stitched_model_action_timestamp = np.concatenate([[last_interp_end_time], model_action_timestamp + stitch_duration])
                stitched_model_desired_qpos_left = np.concatenate([last_interp_end_qpos_L[None, :], model_desired_qpos_left], axis=0)
                stitched_model_desired_qpos_right = np.concatenate([last_interp_end_qpos_R[None, :], model_desired_qpos_right], axis=0)
                
                qpos_interp_L = get_interp1d(stitched_model_action_timestamp, stitched_model_desired_qpos_left)
                qpos_interp_R = get_interp1d(stitched_model_action_timestamp, stitched_model_desired_qpos_right)
                print("Recomputing qpos interpolators with latest model inference output.")
                
            
            # send command that is appropriate for the current eval_state
            if eval_state == "idle":
                command_batch = {
                    f"{follower_left}_mit_command:timestamp": desired_timestamp,
                    f"{follower_left}_mit_command:kp" : q_zeros,
                    f"{follower_left}_mit_command:kd" : q_zeros,
                    f"{follower_left}_mit_command:q"  : qpos_L,
                    f"{follower_left}_mit_command:dq" : q_zeros,
                    f"{follower_left}_mit_command:tau": desired_tau_L,

                    f"{follower_right}_mit_command:timestamp": desired_timestamp,
                    f"{follower_right}_mit_command:kp" : q_zeros,
                    f"{follower_right}_mit_command:kd" : q_zeros,
                    f"{follower_right}_mit_command:q"  : qpos_R,
                    f"{follower_right}_mit_command:dq" : q_zeros,
                    f"{follower_right}_mit_command:tau": desired_tau_R,
                    
                    f"{name}_state:timestamp": desired_timestamp,
                    f"{name}_state:tau_interaction_L": tau_interaction_L,
                    f"{name}_state:tau_interaction_R": tau_interaction_R,
                }
            elif eval_state == "enabling" or eval_state == "evaluating":
                t_now = time.time()
                desired_qpos_L = qpos_interp_L(t_now)
                desired_qpos_R = qpos_interp_R(t_now)
                print(f"{time.time():.4f} Desired qpos delta (deg): Left max abs {np.rad2deg(np.abs(desired_qpos_L - qpos_L)).max():.2f}, Right max abs {np.rad2deg(np.abs(desired_qpos_R - qpos_R)).max():.2f}")
                command_batch = {
                    f"{follower_left}_mit_command:timestamp": desired_timestamp,
                    f"{follower_left}_mit_command:kp" : FOLLOWER_KP,
                    f"{follower_left}_mit_command:kd" : FOLLOWER_KD,
                    f"{follower_left}_mit_command:q"  : desired_qpos_L,
                    f"{follower_left}_mit_command:dq" : q_zeros,
                    f"{follower_left}_mit_command:tau": desired_tau_L,
                    
                    f"{follower_right}_mit_command:timestamp": desired_timestamp,
                    f"{follower_right}_mit_command:kp" : FOLLOWER_KP,
                    f"{follower_right}_mit_command:kd" : FOLLOWER_KD,
                    f"{follower_right}_mit_command:q"  : desired_qpos_R,
                    f"{follower_right}_mit_command:dq" : q_zeros,
                    f"{follower_right}_mit_command:tau": desired_tau_R,
                    
                    f"{name}_state:timestamp": desired_timestamp,
                    f"{name}_state:tau_interaction_L": tau_interaction_L,
                    f"{name}_state:tau_interaction_R": tau_interaction_R,
                }
            else:
                raise ValueError(f"Unknown eval_state: {eval_state}")

            command_batch_maxlen = {key: int(frequency_hz * 10) for key in command_batch.keys()}  # Litian HACK: 10 seconds
            client.stream_add_batch(command_batch, command_batch_maxlen)

            reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=f"openarm controller")
            iter_idx += 1
            
            if iter_idx == 1:
                print("openarm controller ready")


if __name__ == "__main__":
    typer.run(main)
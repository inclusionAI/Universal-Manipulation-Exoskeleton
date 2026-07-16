import time
import numpy as np

import typer
from ume.redis_client import RedisClient
from ume.tools.precise_sleep import FrequencyRegulator

import scipy.spatial.transform as st
from ume.robot.ume.v6_bimanual.openarm_follower.solver import OpenArm313Solver

from ume.tools.keycounter import KeystrokeCounter, KeyCode


FOLLOWER_KP = np.array([240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0])
FOLLOWER_KD = np.array([3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    
    ume_leader_right: str = "ume_leader_right",
    ume_leader_left: str = "ume_leader_left",
    
    ume_controller_name: str = "ume_leader_controller",
    
    follower_left: str = "openarm_follower_left",
    follower_right: str = "openarm_follower_right",
    
    name: str = "openarm_follower_controller",
    history_length_seconds: float = 30.0
):
    
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=2000)
    history_length_redis = int(history_length_seconds * reg.frequency)
    openarm_solver = OpenArm313Solver()
    q_zeros_arm = np.zeros(8)
    
    controller_state = "disabled"
    
    with KeystrokeCounter() as key_counter:
        while True:
            input_data = client.stream_get_batch(
                {
                    f"{ume_leader_right}_state:actual_angle_rad": np.ndarray,
                    f"{ume_leader_right}_state:actual_velocity_radps": np.ndarray,
                    
                    f"{ume_leader_left}_state:actual_angle_rad": np.ndarray,
                    f"{ume_leader_left}_state:actual_velocity_radps": np.ndarray,

                    f"{ume_controller_name}_state:desired_tx_base_R_shoulder": np.ndarray,
                    f"{ume_controller_name}_state:desired_tx_base_R_wrist": np.ndarray,
                    f"{ume_controller_name}_state:desired_R_gripper_pos": np.ndarray,
                    
                    f"{ume_controller_name}_state:xdot_R_shoulder": np.ndarray,
                    f"{ume_controller_name}_state:xdot_R_elbow": np.ndarray,
                    f"{ume_controller_name}_state:xdot_R_wrist": np.ndarray,
                    f"{ume_controller_name}_state:xdot_R_gripper": np.ndarray,
                    
                    f"{ume_controller_name}_state:desired_tx_base_L_shoulder": np.ndarray,
                    f"{ume_controller_name}_state:desired_tx_base_L_wrist": np.ndarray,
                    f"{ume_controller_name}_state:desired_L_gripper_pos": np.ndarray,

                    f"{ume_controller_name}_state:xdot_L_shoulder": np.ndarray,
                    f"{ume_controller_name}_state:xdot_L_elbow": np.ndarray,
                    f"{ume_controller_name}_state:xdot_L_wrist": np.ndarray,
                    f"{ume_controller_name}_state:xdot_L_gripper": np.ndarray,

                    f"{follower_left}_state:actual_angle_rad": np.ndarray,
                    f"{follower_left}_state:actual_velocity_radps": np.ndarray,
                    f"{follower_left}_state:actual_torque_nm": np.ndarray,

                    f"{follower_right}_state:actual_angle_rad": np.ndarray,
                    f"{follower_right}_state:actual_velocity_radps": np.ndarray,
                    f"{follower_right}_state:actual_torque_nm": np.ndarray
                }
            )

            qpos_R = input_data[f"{follower_right}_state:actual_angle_rad"]
            qvel_R = input_data[f"{follower_right}_state:actual_velocity_radps"]
            tau_actual_R = input_data[f"{follower_right}_state:actual_torque_nm"]

            qpos_L = input_data[f"{follower_left}_state:actual_angle_rad"]
            qvel_L = input_data[f"{follower_left}_state:actual_velocity_radps"]
            tau_actual_L = input_data[f"{follower_left}_state:actual_torque_nm"]

            sim_qpos = np.concatenate([qpos_L[:7], [0.0, 0.0], qpos_R[:7], [0.0, 0.0]])
            sim_qvel = np.concatenate([qvel_L[:7], [0.0, 0.0], qvel_R[:7], [0.0, 0.0]])
            desired_qacc = np.zeros_like(sim_qpos)

            openarm_solver.forward_kinematics(sim_qpos)
            tau_gravity_comp = openarm_solver.rnea(sim_qpos, sim_qvel, desired_qacc)

            tau_gravity_comp_L = np.concatenate([tau_gravity_comp[0:7], [0.0]])
            tau_gravity_comp_R = np.concatenate([tau_gravity_comp[9:16], [0.0]])

            # get desired poses
            ume_qpos_R = input_data[f"{ume_leader_right}_state:actual_angle_rad"]
            ume_qvel_R = input_data[f"{ume_leader_right}_state:actual_velocity_radps"]
            ume_qpos_L = input_data[f"{ume_leader_left}_state:actual_angle_rad"]
            ume_qvel_L = input_data[f"{ume_leader_left}_state:actual_velocity_radps"]

            desired_tx_base_R_shoulder = input_data[f"{ume_controller_name}_state:desired_tx_base_R_shoulder"]
            desired_tx_base_R_wrist = input_data[f"{ume_controller_name}_state:desired_tx_base_R_wrist"]
            desired_R_gripper_pos = input_data[f"{ume_controller_name}_state:desired_R_gripper_pos"]

            desired_tx_base_L_shoulder = input_data[f"{ume_controller_name}_state:desired_tx_base_L_shoulder"]
            desired_tx_base_L_wrist = input_data[f"{ume_controller_name}_state:desired_tx_base_L_wrist"]
            desired_L_gripper_pos = input_data[f"{ume_controller_name}_state:desired_L_gripper_pos"]

            # desired qpos
            desired_qpos = sim_qpos.copy()
            desired_qpos_R_shoulder, err_R_shoulder = openarm_solver.ik_ball_joint(sim_qpos, "R_shoulder", desired_tx_base_R_shoulder)
            desired_qpos_R_elbow, err_R_elbow = ume_qpos_R[3:4], ume_qpos_R[3:4] - qpos_R[3:4]
            desired_qpos_R_wrist, err_R_wrist = openarm_solver.ik_ball_joint(sim_qpos, "R_wrist", desired_tx_base_R_wrist)
            desired_qpos_R_gripper, err_R_gripper = desired_R_gripper_pos, desired_R_gripper_pos - qpos_R[7:8]

            desired_qpos[openarm_solver.ball_joint_q_slice["R_shoulder"]] = desired_qpos_R_shoulder
            desired_qpos[openarm_solver.ball_joint_q_slice["R_elbow"]] = desired_qpos_R_elbow
            desired_qpos[openarm_solver.ball_joint_q_slice["R_wrist"]] = desired_qpos_R_wrist

            desired_qpos_L_shoulder, err_L_shoulder = openarm_solver.ik_ball_joint(sim_qpos, "L_shoulder", desired_tx_base_L_shoulder)
            desired_qpos_L_elbow, err_L_elbow = -ume_qpos_L[3:4], -(ume_qpos_L[3:4] - qpos_L[3:4])
            desired_qpos_L_wrist, err_L_wrist = openarm_solver.ik_ball_joint(sim_qpos, "L_wrist", desired_tx_base_L_wrist)
            desired_qpos_L_gripper, err_L_gripper = desired_L_gripper_pos, desired_L_gripper_pos - qpos_L[7:8]

            desired_qpos[openarm_solver.ball_joint_q_slice["L_shoulder"]] = desired_qpos_L_shoulder
            desired_qpos[openarm_solver.ball_joint_q_slice["L_elbow"]] = desired_qpos_L_elbow
            desired_qpos[openarm_solver.ball_joint_q_slice["L_wrist"]] = desired_qpos_L_wrist
            
            # get desired xdots
            desired_xdot_R_shoulder = input_data[f"{ume_controller_name}_state:xdot_R_shoulder"]
            desired_xdot_R_elbow = input_data[f"{ume_controller_name}_state:xdot_R_elbow"]
            desired_xdot_R_wrist = input_data[f"{ume_controller_name}_state:xdot_R_wrist"]
            desired_xdot_R_gripper = input_data[f"{ume_controller_name}_state:xdot_R_gripper"]

            desired_xdot_L_shoulder = input_data[f"{ume_controller_name}_state:xdot_L_shoulder"]
            desired_xdot_L_elbow = input_data[f"{ume_controller_name}_state:xdot_L_elbow"]
            desired_xdot_L_wrist = input_data[f"{ume_controller_name}_state:xdot_L_wrist"]
            desired_xdot_L_gripper = input_data[f"{ume_controller_name}_state:xdot_L_gripper"]

            # desired qvel
            desired_qvel_R = np.zeros(8)
            desired_qvel_R_shoulder = openarm_solver.qvel_ball_joint(sim_qpos, "R_shoulder", desired_xdot_R_shoulder)
            desired_qvel_R_elbow = desired_xdot_R_elbow
            desired_qvel_R_wrist = openarm_solver.qvel_ball_joint(sim_qpos, "R_wrist", desired_xdot_R_wrist)
            desired_qvel_R_gripper = -desired_xdot_R_gripper
            desired_qvel_R[0:3] = desired_qvel_R_shoulder
            desired_qvel_R[3:4] = desired_qvel_R_elbow
            desired_qvel_R[4:7] = desired_qvel_R_wrist
            desired_qvel_R[7:8] = desired_qvel_R_gripper
            
            desired_qvel_L = np.zeros(8)
            desired_qvel_L_shoulder = openarm_solver.qvel_ball_joint(sim_qpos, "L_shoulder", desired_xdot_L_shoulder)
            desired_qvel_L_elbow = -desired_xdot_L_elbow
            desired_qvel_L_wrist = openarm_solver.qvel_ball_joint(sim_qpos, "L_wrist", desired_xdot_L_wrist)
            desired_qvel_L_gripper = desired_xdot_L_gripper
            desired_qvel_L[0:3] = desired_qvel_L_shoulder
            desired_qvel_L[3:4] = desired_qvel_L_elbow
            desired_qvel_L[4:7] = desired_qvel_L_wrist
            desired_qvel_L[7:8] = desired_qvel_L_gripper

            # wrench feedback
            tau_interaction_R = tau_actual_R - tau_gravity_comp_R
            wrench_R_shoulder = openarm_solver.wrench_ball_joint(sim_qpos, "R_shoulder", tau_interaction_R[:3])
            wrench_R_elbow = tau_interaction_R[3:4]
            wrench_R_wrist = openarm_solver.wrench_ball_joint(sim_qpos, "R_wrist", tau_interaction_R[4:7])
            wrench_R_gripper = tau_interaction_R[7:8]
            
            tau_interaction_L = tau_actual_L - tau_gravity_comp_L
            wrench_L_shoulder = openarm_solver.wrench_ball_joint(sim_qpos, "L_shoulder", tau_interaction_L[:3])
            wrench_L_elbow = tau_interaction_L[3:4]
            wrench_L_wrist = openarm_solver.wrench_ball_joint(sim_qpos, "L_wrist", tau_interaction_L[4:7])
            wrench_L_gripper = tau_interaction_L[7:8]

            # update controller state
            for event in key_counter.get_press_events():
                if event == KeyCode(char="b"):
                    controller_state = "enabling"
                    print("set controller state: disabled -> enabling")
                elif event == KeyCode(char="s"):
                    controller_state = "disabled"
                    print("set controller state: disabled")
                elif event == KeyCode(char="`") and controller_state == "enabling":
                    controller_state = "enabled"
                    print("set controller state: enabled")
            
            follower_left_mit_command_kp   = np.zeros(8)
            follower_left_mit_command_kd   = np.zeros(8)
            follower_left_mit_command_q    = np.zeros(8)
            follower_left_mit_command_dq   = np.zeros(8)
            follower_left_mit_command_tau  = np.zeros(8)

            follower_right_mit_command_kp  = np.zeros(8)
            follower_right_mit_command_kd  = np.zeros(8)
            follower_right_mit_command_q   = np.zeros(8)
            follower_right_mit_command_dq  = np.zeros(8)
            follower_right_mit_command_tau = np.zeros(8)
            
            if controller_state == "disabled":
                
                follower_right_mit_command_tau = tau_gravity_comp_R
                follower_left_mit_command_tau  = tau_gravity_comp_L
                
                # no interaction wrench feedback in disabled mode
                wrench_R_shoulder = np.zeros(3)
                wrench_R_elbow    = np.zeros(1)
                wrench_R_wrist    = np.zeros(3)
                wrench_R_gripper  = np.zeros(1)
                
                wrench_L_shoulder = np.zeros(3)
                wrench_L_elbow    = np.zeros(1)
                wrench_L_wrist    = np.zeros(3)
                wrench_L_gripper  = np.zeros(1)
                
            elif controller_state == "enabling":
                max_qpos_delta = np.deg2rad(0.5)
                # gradual move right arm
                desired_qpos_R = np.concatenate([desired_qpos[9:16], desired_qpos_R_gripper])
                qpos_diff_R = desired_qpos_R - qpos_R
                desired_qpos_R_clipped = qpos_R + np.clip(qpos_diff_R, -max_qpos_delta, max_qpos_delta)
                # gradual move left arm
                desired_qpos_L = np.concatenate([desired_qpos[0:7], desired_qpos_L_gripper])
                qpos_diff_L = desired_qpos_L - qpos_L
                desired_qpos_L_clipped = qpos_L + np.clip(qpos_diff_L, -max_qpos_delta, max_qpos_delta)
                # all commands right
                follower_right_mit_command_kp = FOLLOWER_KP
                follower_right_mit_command_kd = FOLLOWER_KD
                follower_right_mit_command_q = desired_qpos_R_clipped
                follower_right_mit_command_tau = tau_gravity_comp_R
                # all commands left
                follower_left_mit_command_kp = FOLLOWER_KP
                follower_left_mit_command_kd = FOLLOWER_KD
                follower_left_mit_command_q = desired_qpos_L_clipped
                follower_left_mit_command_tau = tau_gravity_comp_L
                
                # no interaction wrench feedback in enabling mode
                wrench_R_shoulder = np.zeros(3)
                wrench_R_elbow    = np.zeros(1)
                wrench_R_wrist    = np.zeros(3)
                wrench_R_gripper  = np.zeros(1)
                
                wrench_L_shoulder = np.zeros(3)
                wrench_L_elbow    = np.zeros(1)
                wrench_L_wrist    = np.zeros(3)
                wrench_L_gripper  = np.zeros(1)
                
            elif controller_state == "enabled":
                gripper_desired_delta_range = 0.16  # this is around 2.5 Nm
                # all commands
                desired_qpos_R = np.concatenate([desired_qpos[9:16], desired_qpos_R_gripper])
                follower_right_mit_command_kp = FOLLOWER_KP
                follower_right_mit_command_kd = FOLLOWER_KD
                follower_right_mit_command_q = desired_qpos_R
                # gripper desired position clip
                gripper_R_min, gripper_R_max = qpos_R[-1] - gripper_desired_delta_range, qpos_R[-1] + gripper_desired_delta_range
                follower_right_mit_command_q[-1] = np.clip(follower_right_mit_command_q[-1], gripper_R_min, gripper_R_max)  # gripper position limit
                follower_right_mit_command_dq = desired_qvel_R
                follower_right_mit_command_tau = tau_gravity_comp_R
                
                desired_qpos_L = np.concatenate([desired_qpos[0:7], desired_qpos_L_gripper])
                follower_left_mit_command_kp = FOLLOWER_KP
                follower_left_mit_command_kd = FOLLOWER_KD
                follower_left_mit_command_q = desired_qpos_L
                gripper_L_min, gripper_L_max = qpos_L[-1] - gripper_desired_delta_range, qpos_L[-1] + gripper_desired_delta_range
                follower_left_mit_command_q[-1] = np.clip(follower_left_mit_command_q[-1], gripper_L_min, gripper_L_max)  # gripper position limit
                follower_left_mit_command_dq = desired_qvel_L
                follower_left_mit_command_tau = tau_gravity_comp_L
            else:
                raise ValueError(f"invalid controller state: {controller_state}")

            controller_command_timestamp = time.time()
            
            output_batch = {

                f"{follower_left}_mit_command:kp":   follower_left_mit_command_kp,
                f"{follower_left}_mit_command:kd":   follower_left_mit_command_kd,
                f"{follower_left}_mit_command:q":    follower_left_mit_command_q,
                f"{follower_left}_mit_command:dq":   follower_left_mit_command_dq,
                f"{follower_left}_mit_command:tau":  follower_left_mit_command_tau,
                f"{follower_left}_mit_command:timestamp": controller_command_timestamp,

                f"{follower_right}_mit_command:kp":  follower_right_mit_command_kp,
                f"{follower_right}_mit_command:kd":  follower_right_mit_command_kd,
                f"{follower_right}_mit_command:q":   follower_right_mit_command_q,
                f"{follower_right}_mit_command:dq":  follower_right_mit_command_dq,
                f"{follower_right}_mit_command:tau": follower_right_mit_command_tau,
                f"{follower_right}_mit_command:timestamp": controller_command_timestamp,

                f"{name}_state:timestamp": controller_command_timestamp,

                f"{name}_state:tau_interaction_L": tau_interaction_L,
                f"{name}_state:tau_interaction_R": tau_interaction_R,

                f"{name}_state:interaction_wrench_R_shoulder": wrench_R_shoulder,
                f"{name}_state:interaction_wrench_R_elbow": wrench_R_elbow,
                f"{name}_state:interaction_wrench_R_wrist": wrench_R_wrist,
                f"{name}_state:interaction_wrench_R_gripper": wrench_R_gripper,

                f"{name}_state:err_R_shoulder": err_R_shoulder,
                f"{name}_state:err_R_elbow": err_R_elbow,
                f"{name}_state:err_R_wrist": err_R_wrist,
                f"{name}_state:err_R_gripper": err_R_gripper,
                
                f"{name}_state:interaction_wrench_L_shoulder": wrench_L_shoulder,
                f"{name}_state:interaction_wrench_L_elbow": wrench_L_elbow,
                f"{name}_state:interaction_wrench_L_wrist": wrench_L_wrist,
                f"{name}_state:interaction_wrench_L_gripper": wrench_L_gripper,
                
                f"{name}_state:err_L_shoulder": err_L_shoulder,
                f"{name}_state:err_L_elbow": err_L_elbow,
                f"{name}_state:err_L_wrist": err_L_wrist,
                f"{name}_state:err_L_gripper": err_L_gripper
            }
            output_batch_maxlen = {key: history_length_redis for key in output_batch.keys()}
            client.stream_add_batch(output_batch, output_batch_maxlen)
            
            reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=f"{name} solver")
            if reg.iter_idx == 1:
                print("openarm follower controller ready")


if __name__ == "__main__":
    typer.run(main)

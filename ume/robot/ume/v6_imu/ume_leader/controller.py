import numpy as np

import typer
from ume.redis_client import RedisClient
from ume.tools.precise_sleep import FrequencyRegulator

import scipy.spatial.transform as st
from ume.robot.ume.v6_imu.solver import UMESolver as UME_IMU_Solver
from ume.robot.ume.v6_bimanual.solver import UMESolver as UME_Solver
from ume.robot.ume.v6_imu.ume_mujoco_env import UMEPlayEnv

from ume.robot.openarm1.teleop_leader_tuning import tau_friction_compensation, tau_stiction_compensation
FRICTION_COEFF    = np.array([1, 1, 1, 1, 0.02, 0.02, 0.02, 0.02]) * 1.6
FRICTION_MAX_COMP = np.array([0.4, 0.4, 0.4, 0.4, 0.1,  0.1,  0.1, 0.1])
STICTION_THRESHOLD_MIN = np.deg2rad([1, 1, 1, 1, 1, 1, 1, 1]) * 1
STICTION_THRESHOLD_MAX = np.deg2rad([1, 1, 1, 1, 1, 1, 1, 1]) * 10
STICTION_COMP = np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0])

torque_feedback_tolerance = np.deg2rad(1)
tanh_sharpness = 10.0

def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    right: str = "ume_leader_right",
    left: str = "ume_leader_left",
    name: str = "ume_leader_controller",
    follower_controller_name: str = "openarm_follower_controller"
):
    
    sim = UMEPlayEnv()
    tx_base_imu = sim.get_site_pose("imu")
    tx_imu_base = np.linalg.inv(tx_base_imu)
    del sim
    
    client = RedisClient(host=redis_host, port=redis_port)
    reg = FrequencyRegulator(frequency=2000)

    ume_solver = UME_Solver()
    ume_imu_solver = UME_IMU_Solver()

    q_zeros_arm = np.zeros(8)
    q_zeros_sim = np.zeros(16)
    q_zeros_base_vel = np.zeros(6)

    init_input_interaction_wrench_data = {
        f"{follower_controller_name}_state:interaction_wrench_R_shoulder": np.zeros(3),
        f"{follower_controller_name}_state:interaction_wrench_R_elbow":    np.zeros(1),
        f"{follower_controller_name}_state:interaction_wrench_R_wrist":    np.zeros(3),
        f"{follower_controller_name}_state:interaction_wrench_R_gripper":  np.zeros(1),
        
        f"{follower_controller_name}_state:interaction_wrench_L_shoulder": np.zeros(3),
        f"{follower_controller_name}_state:interaction_wrench_L_elbow":    np.zeros(1),
        f"{follower_controller_name}_state:interaction_wrench_L_wrist":    np.zeros(3),
        f"{follower_controller_name}_state:interaction_wrench_L_gripper":  np.zeros(1),
        
        f"{follower_controller_name}_state:err_R_shoulder": np.zeros(3),
        f"{follower_controller_name}_state:err_R_elbow":    np.zeros(1),
        f"{follower_controller_name}_state:err_R_wrist":    np.zeros(3),
        f"{follower_controller_name}_state:err_R_gripper":  np.zeros(1),
        
        f"{follower_controller_name}_state:err_L_shoulder": np.zeros(3),
        f"{follower_controller_name}_state:err_L_elbow":    np.zeros(1),
        f"{follower_controller_name}_state:err_L_wrist":    np.zeros(3),
        f"{follower_controller_name}_state:err_L_gripper":  np.zeros(1)
    }
    init_input_interaction_wrench_data_maxlen = {key: 1 for key in init_input_interaction_wrench_data.keys()}
    client.stream_add_batch(init_input_interaction_wrench_data, init_input_interaction_wrench_data_maxlen)
    
    while True:
        input_data = client.stream_get_batch(
            {

                f"ybimu_state:imu_pose": np.ndarray,

                f"{right}_state:actual_angle_rad": np.ndarray,
                f"{right}_state:actual_velocity_radps": np.ndarray,

                f"{left}_state:actual_angle_rad": np.ndarray,
                f"{left}_state:actual_velocity_radps": np.ndarray,

                f"{follower_controller_name}_state:interaction_wrench_R_shoulder": np.ndarray,
                f"{follower_controller_name}_state:interaction_wrench_R_elbow":    np.ndarray,
                f"{follower_controller_name}_state:interaction_wrench_R_wrist":    np.ndarray,
                f"{follower_controller_name}_state:interaction_wrench_R_gripper":  np.ndarray,

                f"{follower_controller_name}_state:interaction_wrench_L_shoulder": np.ndarray,
                f"{follower_controller_name}_state:interaction_wrench_L_elbow":    np.ndarray,
                f"{follower_controller_name}_state:interaction_wrench_L_wrist":    np.ndarray,
                f"{follower_controller_name}_state:interaction_wrench_L_gripper":  np.ndarray,

                f"{follower_controller_name}_state:err_R_shoulder": np.ndarray,
                f"{follower_controller_name}_state:err_R_elbow":    np.ndarray,
                f"{follower_controller_name}_state:err_R_wrist":    np.ndarray,
                f"{follower_controller_name}_state:err_R_gripper":  np.ndarray,

                f"{follower_controller_name}_state:err_L_shoulder": np.ndarray,
                f"{follower_controller_name}_state:err_L_elbow":    np.ndarray,
                f"{follower_controller_name}_state:err_L_wrist":    np.ndarray,
                f"{follower_controller_name}_state:err_L_gripper":  np.ndarray
            }
        )

        tx_world_imu = input_data[f"ybimu_state:imu_pose"]
        tx_world_base = tx_world_imu @ tx_imu_base
        tx_base_world = np.linalg.inv(tx_world_base)
        # rx_world_base = tx_world_base[:3, :3]
        rx_base_world = tx_base_world[:3, :3]
        print(f"tx_world_base:\n{tx_world_base}")
        
        qpos_base_xyzw = st.Rotation.from_matrix(tx_world_base[:3, :3]).as_quat()
        qpos_base_xyz = tx_world_base[:3, 3]

        qpos_R = input_data[f"{right}_state:actual_angle_rad"]
        qvel_R = input_data[f"{right}_state:actual_velocity_radps"]
        
        qpos_L = input_data[f"{left}_state:actual_angle_rad"]
        qvel_L = input_data[f"{left}_state:actual_velocity_radps"]

        qpos_pin_imu = np.concatenate([qpos_base_xyz, qpos_base_xyzw, qpos_R, qpos_L])
        qpos_pin = np.concatenate([qpos_R, qpos_L])
        qvel_pin_imu = np.concatenate([q_zeros_base_vel, qvel_R, qvel_L])
        qvel_pin = np.concatenate([qvel_R, qvel_L])
        desired_qacc_imu = np.concatenate([q_zeros_base_vel, q_zeros_arm, q_zeros_arm])
        
        ume_solver.forward_kinematics(qpos_pin)
        ume_imu_solver.forward_kinematics(qpos_pin_imu)
        tau_gravity_comp = ume_imu_solver.rnea(qpos_pin_imu, qvel_pin_imu, desired_qacc_imu)
        
        tau_friction_comp_R = tau_friction_compensation(qvel_R, FRICTION_COEFF, FRICTION_MAX_COMP)
        tau_stiction_comp_R = tau_stiction_compensation(qvel_R, STICTION_THRESHOLD_MIN, STICTION_THRESHOLD_MAX, STICTION_COMP)
        
        tau_friction_comp_L = tau_friction_compensation(qvel_L, FRICTION_COEFF, FRICTION_MAX_COMP)
        tau_stiction_comp_L = tau_stiction_compensation(qvel_L, STICTION_THRESHOLD_MIN, STICTION_THRESHOLD_MAX, STICTION_COMP)
        
        tau_ff_R = tau_gravity_comp[ 6:14] + tau_friction_comp_R + tau_stiction_comp_R
        tau_ff_L = tau_gravity_comp[14:22] + tau_friction_comp_L + tau_stiction_comp_L

        follower_wrench_R_shoulder = input_data[f"{follower_controller_name}_state:interaction_wrench_R_shoulder"]
        follower_wrench_R_elbow    = input_data[f"{follower_controller_name}_state:interaction_wrench_R_elbow"]
        follower_wrench_R_wrist    = input_data[f"{follower_controller_name}_state:interaction_wrench_R_wrist"]
        follower_wrench_R_gripper  = input_data[f"{follower_controller_name}_state:interaction_wrench_R_gripper"]

        interaction_tau_R_shoulder = ume_solver.get_ball_joint_torque(qpos_pin, "R_shoulder", follower_wrench_R_shoulder)
        interaction_tau_R_elbow = follower_wrench_R_elbow
        interaction_tau_R_wrist = ume_solver.get_ball_joint_torque(qpos_pin, "R_wrist", follower_wrench_R_wrist)
        interaction_tau_R_gripper = -follower_wrench_R_gripper

        err_R_shoulder = np.linalg.norm(input_data[f"{follower_controller_name}_state:err_R_shoulder"])
        err_R_elbow    = np.linalg.norm(input_data[f"{follower_controller_name}_state:err_R_elbow"])
        err_R_wrist    = np.linalg.norm(input_data[f"{follower_controller_name}_state:err_R_wrist"])
        err_R_gripper  = np.linalg.norm(input_data[f"{follower_controller_name}_state:err_R_gripper"])
        
        follower_wrench_L_shoulder = input_data[f"{follower_controller_name}_state:interaction_wrench_L_shoulder"]
        follower_wrench_L_elbow    = input_data[f"{follower_controller_name}_state:interaction_wrench_L_elbow"]
        follower_wrench_L_wrist    = input_data[f"{follower_controller_name}_state:interaction_wrench_L_wrist"]
        follower_wrench_L_gripper  = input_data[f"{follower_controller_name}_state:interaction_wrench_L_gripper"]
        
        interaction_tau_L_shoulder = ume_solver.get_ball_joint_torque(qpos_pin, "L_shoulder", follower_wrench_L_shoulder)
        interaction_tau_L_elbow = -follower_wrench_L_elbow
        interaction_tau_L_wrist = ume_solver.get_ball_joint_torque(qpos_pin, "L_wrist", follower_wrench_L_wrist)
        interaction_tau_L_gripper = follower_wrench_L_gripper

        err_L_shoulder = np.linalg.norm(input_data[f"{follower_controller_name}_state:err_L_shoulder"])
        err_L_elbow    = np.linalg.norm(input_data[f"{follower_controller_name}_state:err_L_elbow"])
        err_L_wrist    = np.linalg.norm(input_data[f"{follower_controller_name}_state:err_L_wrist"])
        err_L_gripper  = np.linalg.norm(input_data[f"{follower_controller_name}_state:err_L_gripper"])

        SCALE = 0.5
        scale_R_shoulder = SCALE * (np.tanh(tanh_sharpness * (np.abs(err_R_shoulder) - torque_feedback_tolerance)) + 1) / 2
        scale_R_elbow    = SCALE * (np.tanh(tanh_sharpness * (np.abs(err_R_elbow)    - torque_feedback_tolerance)) + 1) / 2
        scale_R_wrist    = SCALE * (np.tanh(tanh_sharpness * (np.abs(err_R_wrist)    - torque_feedback_tolerance)) + 1) / 2
        scale_R_gripper  = SCALE * (np.tanh(tanh_sharpness * (np.abs(err_R_gripper)  - torque_feedback_tolerance)) + 1) / 2

        scale_L_shoulder = SCALE * (np.tanh(tanh_sharpness * (np.abs(err_L_shoulder) - torque_feedback_tolerance)) + 1) / 2
        scale_L_elbow    = SCALE * (np.tanh(tanh_sharpness * (np.abs(err_L_elbow)    - torque_feedback_tolerance)) + 1) / 2
        scale_L_wrist    = SCALE * (np.tanh(tanh_sharpness * (np.abs(err_L_wrist)    - torque_feedback_tolerance)) + 1) / 2
        scale_L_gripper  = SCALE * (np.tanh(tanh_sharpness * (np.abs(err_L_gripper)  - torque_feedback_tolerance)) + 1) / 2

        MAX_TORQUE_4310 = 1
        MAX_TORQUE_4340 = 4

        tau_ff_R[0:3] -= np.clip(scale_R_shoulder * interaction_tau_R_shoulder, -MAX_TORQUE_4340, MAX_TORQUE_4340)
        tau_ff_R[3:4] -= np.clip(scale_R_elbow    * interaction_tau_R_elbow,    -MAX_TORQUE_4340, MAX_TORQUE_4340)
        tau_ff_R[4:7] -= np.clip(scale_R_wrist    * interaction_tau_R_wrist,    -MAX_TORQUE_4310, MAX_TORQUE_4310)
        tau_ff_R[7:8] -= np.clip(scale_R_gripper  * interaction_tau_R_gripper,  -MAX_TORQUE_4310, MAX_TORQUE_4310)

        tau_ff_L[0:3] -= np.clip(scale_L_shoulder * interaction_tau_L_shoulder, -MAX_TORQUE_4340, MAX_TORQUE_4340)
        tau_ff_L[3:4] -= np.clip(scale_L_elbow    * interaction_tau_L_elbow,    -MAX_TORQUE_4340, MAX_TORQUE_4340)
        tau_ff_L[4:7] -= np.clip(scale_L_wrist    * interaction_tau_L_wrist,    -MAX_TORQUE_4310, MAX_TORQUE_4310)
        tau_ff_L[7:8] -= np.clip(scale_L_gripper  * interaction_tau_L_gripper,  -MAX_TORQUE_4310, MAX_TORQUE_4310)
        
        tx_base_R_shoulder = ume_solver.get_ball_joint_pose("R_shoulder")
        tx_base_R_wrist = ume_solver.get_ball_joint_pose("R_wrist")
        ume_R_gripper_pos = qpos_R[7:8]

        xdot_R_shoulder = ume_solver.get_ball_joint_xdot(qpos_pin, "R_shoulder", qvel_pin)
        xdot_R_elbow = qvel_R[3:4]
        xdot_R_wrist = ume_solver.get_ball_joint_xdot(qpos_pin, "R_wrist", qvel_pin)
        xdot_R_gripper = qvel_R[7:8]

        tx_base_L_shoulder = ume_solver.get_ball_joint_pose("L_shoulder")
        tx_base_L_wrist = ume_solver.get_ball_joint_pose("L_wrist")
        ume_L_gripper_pos = qpos_L[7:8]
        
        xdot_L_shoulder = ume_solver.get_ball_joint_xdot(qpos_pin, "L_shoulder", qvel_pin)
        xdot_L_elbow = qvel_L[3:4]
        xdot_L_wrist = ume_solver.get_ball_joint_xdot(qpos_pin, "L_wrist", qvel_pin)
        xdot_L_gripper = qvel_L[7:8]

        output_batch = {
            f"{name}_state:mujoco_qpos": np.concatenate([qpos_base_xyz, qpos_base_xyzw[[3, 0, 1, 2]], qpos_R, qpos_L]),
            f"{name}_state:pinocchio_qpos": qpos_pin,

            f"{name}_state:desired_tx_base_R_shoulder": tx_base_R_shoulder,
            f"{name}_state:desired_tx_base_R_wrist":    tx_base_R_wrist,
            f"{name}_state:desired_R_gripper_pos": -ume_R_gripper_pos - 1.17,

            f"{name}_state:xdot_R_shoulder": xdot_R_shoulder,
            f"{name}_state:xdot_R_elbow":    xdot_R_elbow,
            f"{name}_state:xdot_R_wrist":    xdot_R_wrist,
            f"{name}_state:xdot_R_gripper":  xdot_R_gripper,

            f"{name}_state:desired_tx_base_L_shoulder": tx_base_L_shoulder,
            f"{name}_state:desired_tx_base_L_wrist":    tx_base_L_wrist,
            f"{name}_state:desired_L_gripper_pos":  ume_L_gripper_pos - 1.12,

            f"{name}_state:xdot_L_shoulder": xdot_L_shoulder,
            f"{name}_state:xdot_L_elbow":    xdot_L_elbow,
            f"{name}_state:xdot_L_wrist":    xdot_L_wrist,
            f"{name}_state:xdot_L_gripper":  xdot_L_gripper,

            f"{right}_mit_command:kp" : q_zeros_arm,
            f"{right}_mit_command:kd" : q_zeros_arm,
            f"{right}_mit_command:q"  : q_zeros_arm,
            f"{right}_mit_command:dq" : q_zeros_arm,
            f"{right}_mit_command:tau": tau_ff_R,

            f"{left}_mit_command:kp" : q_zeros_arm,
            f"{left}_mit_command:kd" : q_zeros_arm,
            f"{left}_mit_command:q"  : q_zeros_arm,
            f"{left}_mit_command:dq" : q_zeros_arm,
            f"{left}_mit_command:tau": tau_ff_L
        }

        output_maxlen = {key: 1 for key in output_batch.keys()}
        client.stream_add_batch(output_batch, output_maxlen)

        reg.sleep(verbose=True, verbose_interval=1, verbose_prefix=f"{name} solver")
        if reg.iter_idx == 1:
            print("ume leader controller ready")

if __name__ == "__main__":
    typer.run(main)
import numpy as np
import time
import canopen
from ume.hexfellow_chain import HexFellowChain
from ume.robot.mobile_base.power_caster_base import PowerCasterBaseKinematics

class HexFellowBaseController:

    def __init__(self, hex_chain_L: HexFellowChain, hex_chain_R: HexFellowChain):
        self.hex_chain_L = hex_chain_L
        self.hex_chain_R = hex_chain_R
        self.pcw_kinematics = PowerCasterBaseKinematics(
            n_casters=4, 
            wheel_bx=0.02, 
            wheel_by=0.0, 
            wheel_r=0.123/2,
            wheel_hx=np.array([ 1, -1,  1, -1]) * 0.185,
            wheel_hy=np.array([ 1,  1, -1, -1]) * 0.185
        )
    
    def get_actual_position_rad(self):
        rev_offset = np.array([0.15, 0.0, -0.15, 0.0, 0.37, 0.0, 0.375, 0.0])  # Adjust based on your hardware setup
        rev_offset[0::2] += 0.5
        position_rev = np.concatenate((self.hex_chain_L.get_motor_state()["position"], self.hex_chain_R.get_motor_state()["position"]))
        position_rev = position_rev + rev_offset
        # print(f"Actual position (rev): {position_rev}")
        position_rad = (position_rev % 1) * 2 * np.pi
        return position_rad
    
    def get_actual_velocity_radps(self):
        velocity_revps = np.concatenate((self.hex_chain_L.get_motor_state()["velocity"], self.hex_chain_R.get_motor_state()["velocity"]))
        velocity_radps = velocity_revps * 2 * np.pi
        return velocity_radps
    
    def update_kinematics_model(self, dt):
        qpos = self.get_actual_position_rad()
        qvel = self.get_actual_velocity_radps()
        self.pcw_kinematics.update_state(qpos=qpos, qvel=qvel, dt=dt)

    def osc_velocity_control(self, desired_vel_xyt):
        q_vel = self.pcw_kinematics.joint_velocity(desired_vel_xyt)
        # q_vel[1::2] *= -1  # Flip the drive motor velocity to match the actual motor direction
        # print(f"Desired joint velocity (rad/s): {q_vel}")
        # print(q_vel)
        if self.hex_chain_L.mode == "velocity":
            self.hex_chain_L.velocity_control(torque_limit_Nm=5.0, desired_revps=q_vel[:4] / (2 * np.pi))
            self.hex_chain_R.velocity_control(torque_limit_Nm=5.0, desired_revps=q_vel[4:] / (2 * np.pi))
        elif self.hex_chain_L.mode == "mit":
            q_zeros = np.zeros(4)
            kd_steer = 0.5
            kd_drive = 0.5
            kd_gains = np.array([kd_steer, kd_drive, kd_steer, kd_drive])
            self.hex_chain_L.mit_control(
                positions=q_zeros,  # Not used in MIT mode
                velocities=q_vel[:4] / (2 * np.pi),
                torques=q_zeros,  # No feedforward torque
                kp_gains=q_zeros,
                kd_gains=kd_gains
            )
            self.hex_chain_R.mit_control(
                positions=q_zeros,  # Not used in MIT mode
                velocities=q_vel[4:] / (2 * np.pi),
                torques=q_zeros,  # No feedforward torque
                kp_gains=q_zeros,
                kd_gains=kd_gains
            )

    def osc_impedance_control(self, desired_pos_xyt):
        current_pos_xyt = self.pcw_kinematics.x
        err_pos_xyt = desired_pos_xyt - current_pos_xyt
        Kp = np.ones(3) * 0.25  # Proportional gains for (x, y, theta)
        Kv = np.ones(3) * 0.05  # Derivative gains for (x, y, theta)
        F_task_global = Kp * err_pos_xyt - Kv * self.pcw_kinematics.dx
        rx_local_global = self.pcw_kinematics.rx_local_global(current_pos_xyt[2])
        F_task_local = rx_local_global.T @ F_task_global  # Transform to local frame
        tau = self.pcw_kinematics.C @ F_task_local
        
        # print(f"err {err_pos_xyt}, ") # F_task {F_task_global}, tau {tau}
        
        q_zeros = np.zeros(4)
        self.hex_chain_L.mit_control(q_zeros, q_zeros, tau[:4], q_zeros, q_zeros)
        self.hex_chain_R.mit_control(q_zeros, q_zeros, tau[4:], q_zeros, q_zeros)


import typer
from ume.tools.precise_sleep import FrequencyRegulator
from ruckig import InputParameter, OutputParameter, Result, Ruckig, ControlInterface
from ume.redis_client import RedisClient

import numpy as np
import scipy.spatial.transform as st
from ume.robot.ybimu.ybimu_simple import YbImu



class IMUMobileBaseTeleop:
    
    def __init__(self, initial_pose):
        self.mount_transform = np.eye(4)
        self.mount_transform[:3, :3] = st.Rotation.from_rotvec([0, -np.pi/2, 0]).as_matrix()
        self.initial_pose = initial_pose @ self.mount_transform

    def compute_delta_euler(self, pose):
        current_pose = pose @ self.mount_transform
        delta_pose = np.linalg.inv(self.initial_pose) @ current_pose
        delta_euler = st.Rotation.from_matrix(delta_pose[:3, :3]).as_euler('xyz')
        return delta_euler

    def compute_velocity_xyt(self, pose, translation_scale=2, rotation_scale=2, translation_deadzone=0.05, rotation_deadzone=0.05):
        delta_euler = self.compute_delta_euler(pose)
        # match pitch to dx, roll to dy
        velocity_xyt = np.zeros(3)
        velocity_xyt[0] =  delta_euler[1] * translation_scale  # pitch
        velocity_xyt[1] =  delta_euler[0] * translation_scale  # roll
        velocity_xyt[2] = -delta_euler[2] * rotation_scale  # yaw
        # deadzone
        velocity_xyt[0] = 0 if np.abs(velocity_xyt[0]) < translation_deadzone else velocity_xyt[0] - np.sign(velocity_xyt[0]) * translation_deadzone
        velocity_xyt[1] = 0 if np.abs(velocity_xyt[1]) < translation_deadzone else velocity_xyt[1] - np.sign(velocity_xyt[1]) * translation_deadzone
        velocity_xyt[2] = 0 if np.abs(velocity_xyt[2]) < rotation_deadzone else velocity_xyt[2] - np.sign(velocity_xyt[2]) * rotation_deadzone
        return velocity_xyt


def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    name: str = "hexfellow_base_imu_teleop",
    history_length_seconds: int = 30
):
    client = RedisClient(host=redis_host, port=redis_port)
    control_mode = "mit"  # Change to "mit" for MIT control mode
    with \
        canopen.Network().connect(channel='can0', bustype='socketcan', fd=True) as network_L, \
        canopen.Network().connect(channel='can1', bustype='socketcan', fd=True) as network_R, \
        HexFellowChain(network_L, node_ids=[2, 1, 4, 3], mode=control_mode) as hex_chain_L, \
        HexFellowChain(network_R, node_ids=[2, 1, 4, 3], mode=control_mode) as hex_chain_R:

        controller = HexFellowBaseController(hex_chain_L, hex_chain_R)
        
        DOF = 3
        CONTROL_FREQUENCY = 1000
        CONTROL_PERIOD = 1 / CONTROL_FREQUENCY
        MAX_VEL = np.ones(DOF) * 0.5
        MAX_ACC = np.ones(DOF) * 0.8
        otg = Ruckig(DOF, CONTROL_PERIOD)
        otg_input = InputParameter(DOF)
        otg_output = OutputParameter(DOF)
        otg_input.max_velocity = MAX_VEL
        otg_input.max_acceleration = MAX_ACC
        otg_input.control_interface = ControlInterface.Velocity
        last_desired_vel_xyt = np.zeros(DOF)
        
        max_redis_history_length = int(CONTROL_FREQUENCY * history_length_seconds)

        reg = FrequencyRegulator(CONTROL_FREQUENCY)  # 1000 Hz control loop
        
        
        input_data = client.stream_get_batch({f"ybimu_state:imu_pose": np.ndarray})
        imu_pose = input_data[f"ybimu_state:imu_pose"]
        imu_teleop = IMUMobileBaseTeleop(imu_pose)
        
        while True:
            t_now = time.time()
            actual_position_rad = controller.get_actual_position_rad()
            actual_velocity_radps = controller.get_actual_velocity_radps()

            controller.update_kinematics_model(reg.dt)
            actual_velocity_xyt = controller.pcw_kinematics.dx
            actual_position_xyt = controller.pcw_kinematics.x

            scale = MAX_VEL[0]
            input_data = client.stream_get_batch(
                {
                    f"ybimu_state:imu_pose": np.ndarray
                }
            )
            imu_pose = input_data[f"ybimu_state:imu_pose"]
            desired_vel_xyt = np.clip(imu_teleop.compute_velocity_xyt(imu_pose), -1, 1) * scale

            otg_input.current_velocity = last_desired_vel_xyt
            otg_input.target_velocity = desired_vel_xyt
            
            result = otg.update(otg_input, otg_output)
            if result == Result.Working:
                desired_vel_xyt_with_acc_limit = np.array(otg_output.new_velocity)
                # print(f"Current velocity: {otg_output.new_velocity}")
            elif result == Result.Finished:
                desired_vel_xyt_with_acc_limit = np.array(otg_output.new_velocity)
                # print("Reached target velocity.")
            else:
                # print("Error in Ruckig update!")
                pass
            controller.osc_velocity_control(desired_vel_xyt_with_acc_limit)
            last_desired_vel_xyt = desired_vel_xyt_with_acc_limit
            
            output_batch = {
                f"{name}:timestamp": t_now,
                f"{name}:actual_position_rad": actual_position_rad,
                f"{name}:actual_velocity_radps": actual_velocity_radps,
                f"{name}:actual_velocity_xyt": actual_velocity_xyt,
                f"{name}:actual_position_xyt": actual_position_xyt,
                f"{name}:desired_vel_xyt": desired_vel_xyt,
                f"{name}:desired_vel_xyt_with_acc_limit": desired_vel_xyt_with_acc_limit
            }
            # for key, value in output_batch.items():
            #     print(f"{key}: {value} {type(value)}")
            output_batch_maxlen = {
                f"{name}:timestamp": max_redis_history_length,
                f"{name}:actual_position_rad": max_redis_history_length,
                f"{name}:actual_velocity_radps": max_redis_history_length,
                f"{name}:actual_velocity_xyt": max_redis_history_length,
                f"{name}:actual_position_xyt": max_redis_history_length,
                f"{name}:desired_vel_xyt": max_redis_history_length,
                f"{name}:desired_vel_xyt_with_acc_limit": max_redis_history_length
            }
            client.stream_add_batch(output_batch, output_batch_maxlen)
            
            reg.sleep(verbose=True)
            
            if reg.iter_idx == 1:
                print("HexFellow base IMU teleop ready")

if __name__ == "__main__":
    typer.run(main)

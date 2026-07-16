import numpy as np
from ume.damiao_chain import DamiaoChain, MotorType
from ume.tools.precise_sleep import FrequencyRegulator
from ume.robot.ume.v6_imu.solver import UMESolver
from ume.robot.ybimu.ybimu_simple import YbImu
from ume.robot.ume.v6_imu.ume_mujoco_env import UMEPlayEnv
import scipy.spatial.transform as st


from ume.robot.openarm1.teleop_leader_tuning import tau_friction_compensation, tau_stiction_compensation
FRICTION_COEFF    = np.array([1, 1, 1, 1, 0.02, 0.02, 0.02, 0.02]) * 1.6
FRICTION_MAX_COMP = np.array([0.4, 0.4, 0.4, 0.4, 0.1,  0.1,  0.1, 0.1])
STICTION_THRESHOLD_MIN = np.deg2rad([1, 1, 1, 1, 1, 1, 1, 1]) * 1
STICTION_THRESHOLD_MAX = np.deg2rad([1, 1, 1, 1, 1, 1, 1, 1]) * 10
STICTION_COMP = np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0])


def pretty_arr(arr):
    return "".join([f"  ({i+1}){arr[i]:8.4f}  " for i in range(len(arr))])


if __name__ == "__main__":
    motor_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    sim = UMEPlayEnv()
    use_sim = True
    n_motors = len(motor_ids)
    q_zeros = np.zeros(n_motors)
    frequency = 400

    if use_sim:
        sim_viewer = sim.launch_viewer()
        frequency = 100

    reg = FrequencyRegulator(frequency=frequency)
    solver = UMESolver()
    
    ybimu = YbImu()
    tx_base_imu = sim.get_site_pose("imu")
    tx_imu_base = np.linalg.inv(tx_base_imu)

    with \
        DamiaoChain(
            motor_ids=motor_ids,
            motor_types=[
                MotorType.DM4340,
                MotorType.DM4340,
                MotorType.DM4340,
                MotorType.DM4340,
                MotorType.DM4310,
                MotorType.DM4310,
                MotorType.DM4310,
                MotorType.DM4310
            ], 
            can_channel="can4", can_fd=True, timeout=2
        ) as ume_R, \
        DamiaoChain(
            motor_ids=motor_ids,
            motor_types=[
                MotorType.DM4340,
                MotorType.DM4340,
                MotorType.DM4340,
                MotorType.DM4340,
                MotorType.DM4310,
                MotorType.DM4310,
                MotorType.DM4310,
                MotorType.DM4310
            ], 
            can_channel="can5", can_fd=True, timeout=2
        ) as ume_L:

        data_R = ume_R.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=q_zeros)
        data_L = ume_L.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=q_zeros)
        
        while True:
            
            tx_world_imu = ybimu.get_pose()
            tx_world_base = tx_world_imu @ tx_imu_base
            
            qpos_base_xyzw = st.Rotation.from_matrix(tx_world_base[:3, :3]).as_quat()
            qpos_base_xyz = tx_world_base[:3, 3]
            qpos_base_wxyz = qpos_base_xyzw[[3, 0, 1, 2]]
            
            qpos_R = data_R["actual_angle_rad"]
            qvel_R = data_R["actual_velocity_radps"]

            qpos_L = data_L["actual_angle_rad"]
            qvel_L = data_L["actual_velocity_radps"]
            
            qpos_pin = np.concatenate([qpos_base_xyz, qpos_base_xyzw, qpos_R, qpos_L])
            qpos_mjc = np.concatenate([qpos_base_xyz, qpos_base_wxyz, qpos_R, qpos_L])
            qvel = np.concatenate([np.zeros(6), qvel_R, qvel_L])
            desired_qacc = np.zeros_like(qvel)

            solver.forward_kinematics(qpos_pin)
            tau_gravity_comp = solver.rnea(qpos_pin, qvel, desired_qacc)
            tau_gravity_comp_R = tau_gravity_comp[6:14]
            tau_gravity_comp_L = tau_gravity_comp[14:22]

            tau_friction_comp_R = tau_friction_compensation(qvel_R, FRICTION_COEFF, FRICTION_MAX_COMP)
            tau_stiction_comp_R = tau_stiction_compensation(qvel_R, STICTION_THRESHOLD_MIN, STICTION_THRESHOLD_MAX, STICTION_COMP)
            tau_ff_R = tau_gravity_comp_R + tau_friction_comp_R + tau_stiction_comp_R

            tau_friction_comp_L = tau_friction_compensation(qvel_L, FRICTION_COEFF, FRICTION_MAX_COMP)
            tau_stiction_comp_L = tau_stiction_compensation(qvel_L, STICTION_THRESHOLD_MIN, STICTION_THRESHOLD_MAX, STICTION_COMP)
            tau_ff_L = tau_gravity_comp_L + tau_friction_comp_L + tau_stiction_comp_L

            data_R = ume_R.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=1 * tau_ff_R)
            data_L = ume_L.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=1 * tau_ff_L)

            pretty = f"{pretty_arr(qpos_pin)}\n{pretty_arr(tau_gravity_comp)}"

            if use_sim:
                sim.set_mocap_pose("R_shoulder", sim.get_site_pose("R_shoulder"))
                sim.set_mocap_pose("R_wrist", sim.get_site_pose("R_wrist"))
                sim.set_mocap_pose("L_shoulder", sim.get_site_pose("L_shoulder"))
                sim.set_mocap_pose("L_wrist", sim.get_site_pose("L_wrist"))

                sim.set_qpos(qpos_mjc)
                sim.update()
                
                sim_viewer.sync()
            reg.sleep(verbose=True, verbose_prefix=f"real_teleop_sim.py {pretty}")

import numpy as np
import time
import canopen
from ume.robot.mobile_openarm.hexfellow_base_imu_teleop_redis import HexFellowBaseController, HexFellowChain

if __name__ == "__main__":
    import cv2
    from ume.tools.precise_sleep import FrequencyRegulator
    from ume.xbox import XboxController
    from ruckig import InputParameter, OutputParameter, Result, Ruckig, ControlInterface
    
    xbox = XboxController()
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

        reg = FrequencyRegulator(CONTROL_FREQUENCY)  # 1000 Hz control loop
        while True:
            actual_position_rad = controller.get_actual_position_rad()
            # actual_torque_nm = np.concatenate((hex_chain_L.get_motor_state()["torque"], hex_chain_R.get_motor_state()["torque"]))

            controller.update_kinematics_model(reg.dt)
            actual_velocity_xyt = controller.pcw_kinematics.dx
            # print(f"{time.time():.4f} Actual Velocity (m/s, rad/s): {actual_velocity_xyt}")
            actual_position_xyt = controller.pcw_kinematics.x
            # print(f"{time.time():.4f} Actual Position (m, m, rad): {actual_position_xyt}")

            scale = MAX_VEL[0]  # Max speed scaling factor (0.5 means max 0.5 m/s)
            desired_vel_x = -xbox.LeftJoystickY * scale
            desired_vel_y = -xbox.LeftJoystickX * scale
            desired_vel_theta = -xbox.RightJoystickX * scale  # Max angular speed scaling
            print(f"{time.time():.4f} Desired Velocity (m/s, rad/s): ({desired_vel_x:.2f}, {desired_vel_y:.2f}, {desired_vel_theta:.2f})")
            desired_vel_xyt = np.array([desired_vel_x, desired_vel_y, desired_vel_theta])
            # print(desired_vel_xyt, actual_velocity_xyt)
            # controller.osc_velocity_control(desired_vel_xyt)

            otg_input.current_velocity = last_desired_vel_xyt
            otg_input.target_velocity = desired_vel_xyt
            
            result = otg.update(otg_input, otg_output)
            if result == Result.Working:
                desired_vel_xyt_with_acc_limit = otg_output.new_velocity
                # print(f"Current velocity: {otg_output.new_velocity}")
            elif result == Result.Finished:
                desired_vel_xyt_with_acc_limit = otg_output.new_velocity
                # print("Reached target velocity.")
            else:
                # print("Error in Ruckig update!")
                pass
            controller.osc_velocity_control(desired_vel_xyt_with_acc_limit)
            last_desired_vel_xyt = desired_vel_xyt_with_acc_limit
            
            reg.sleep(verbose=True)

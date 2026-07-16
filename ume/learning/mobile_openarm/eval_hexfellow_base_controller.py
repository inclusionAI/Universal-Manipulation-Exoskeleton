import numpy as np
import time
import canopen
from ume.hexfellow_chain import HexFellowChain
from ume.robot.mobile_base.power_caster_base import PowerCasterBaseKinematics
from ume.robot.mobile_openarm.hexfellow_base_imu_teleop_redis import HexFellowBaseController

import typer
from ume.tools.pose_interp import get_interp1d
from ume.tools.precise_sleep import FrequencyRegulator
from ume.xbox import XboxController
from ume.tools.keycounter import KeystrokeCounter, Key, KeyCode
from ruckig import InputParameter, OutputParameter, Result, Ruckig, ControlInterface
from ume.redis_client import RedisClient

def main(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    name: str = "hexfellow_base_controller",
    history_length_seconds: int = 30,
    model_inference_worker_name: str = "model_inference",
    chunk_renew_every_s: float = 2.0,
    stitch_duration: float = 0.0
):
    client = RedisClient(host=redis_host, port=redis_port)
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
        
        max_redis_history_length = int(CONTROL_FREQUENCY * history_length_seconds)

        reg = FrequencyRegulator(CONTROL_FREQUENCY)  # 1000 Hz control loop
        
        eval_state = "idle"
        desired_base_velocity_xyt_interp = None
        
        with KeystrokeCounter() as key_counter:
            while True:
                
                # get current state
                t_now = time.time()
                actual_position_rad = controller.get_actual_position_rad()
                actual_velocity_radps = controller.get_actual_velocity_radps()
                
                data_batch = client.stream_get_batch(
                    {
                        f"{model_inference_worker_name}:action_timestamp": np.ndarray,
                        f"{model_inference_worker_name}:action_desired_base_velocity_xyt": np.ndarray
                    }
                )
                model_action_timestamp = data_batch[f"{model_inference_worker_name}:action_timestamp"]
                model_desired_base_velocity_xyt = data_batch[f"{model_inference_worker_name}:action_desired_base_velocity_xyt"]
                
                # handle keyboard input for state transitions
                for press_events in key_counter.get_press_events():
                    if press_events == KeyCode(char='`'):
                        if model_action_timestamp is not None:
                            eval_state = "evaluating"
                            desired_base_velocity_xyt_interp = get_interp1d(model_action_timestamp, model_desired_base_velocity_xyt)
                            print("Evaluating... Interpolating desired base velocity from model inference.")
                        else:
                            print("No model inference data received yet. Cannot start evaluating.")
                    elif press_events == KeyCode(char='s'):
                        eval_state = "idle"
                        desired_base_velocity_xyt_interp = None
                        print("Stopping. Returning to idle state.")
                
                if eval_state == "evaluating" and desired_base_velocity_xyt_interp is not None and t_now > desired_base_velocity_xyt_interp.x[0] + chunk_renew_every_s + stitch_duration:
                    # if we are evaluating and close to the end of the model predicted trajectory
                    # recompute the interpolator with latest model output
                    
                    last_interp_end_time = desired_base_velocity_xyt_interp.x[0] + chunk_renew_every_s + stitch_duration
                    last_interp_end_desired_vel_xyt = np.zeros(3)
                    # stitch_duration = 0.0
                    stitched_model_action_timestamp = np.concatenate([[last_interp_end_time], model_action_timestamp + stitch_duration])
                    stitched_model_desired_base_velocity_xyt = np.concatenate([[last_interp_end_desired_vel_xyt], model_desired_base_velocity_xyt], axis=0)
                    desired_base_velocity_xyt_interp = get_interp1d(stitched_model_action_timestamp, stitched_model_desired_base_velocity_xyt)

                controller.update_kinematics_model(reg.dt)
                actual_velocity_xyt = controller.pcw_kinematics.dx
                actual_position_xyt = controller.pcw_kinematics.x

                scale = MAX_VEL[0]  # Max speed scaling factor (0.5 means max 0.5 m/s)
                
                if eval_state == "idle":
                    desired_vel_x = -xbox.LeftJoystickY * scale
                    desired_vel_y = -xbox.LeftJoystickX * scale
                    desired_vel_theta = -xbox.RightJoystickX * scale  # Max angular speed scaling
                    print(f"{time.time():.4f} Desired Velocity (m/s, rad/s): ({desired_vel_x:.2f}, {desired_vel_y:.2f}, {desired_vel_theta:.2f})")
                    desired_vel_xyt = np.array([desired_vel_x, desired_vel_y, desired_vel_theta])
                elif eval_state == "evaluating":
                    desired_vel_xyt = desired_base_velocity_xyt_interp(t_now)
                    print(f"{time.time():.4f} Desired Velocity from Model (m/s, rad/s): ({desired_vel_xyt[0]:.2f}, {desired_vel_xyt[1]:.2f}, {desired_vel_xyt[2]:.2f})")
                else:
                    raise ValueError(f"Unknown eval_state: {eval_state}")

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
                controller.velocity_control(desired_vel_xyt_with_acc_limit)
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
                    print("HexFellow base ready")

if __name__ == "__main__":
    typer.run(main)

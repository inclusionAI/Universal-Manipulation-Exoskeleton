import time
from typing import Dict

import cv2
import torch
import typer
import numpy as np

import yaml
from ume.learning.mobile_openarm_ume.model import ACT_Abs
from ume.redis_client import RedisClient
from ume.tools.fisheye_camera import jieruiweitong_fisheye_center_crop
from ume.tools.pose_interp import get_interp1d


def load_model(model_path):
    # 04 26 model: 1 obs (img, actual joints, desired joints), 20 actions (base velocity, desired joints)
    act_model, _ = ACT_Abs.build_model_and_optimizer(
        n_obs_steps=1,
        action_chunk_size=20,
        seed=42
    )
    act_model.load_state_dict(torch.load(model_path))
    act_model.cuda()
    act_model.eval()
    return act_model

def image_crop_and_resize(image_bgr):
    cropped_image = jieruiweitong_fisheye_center_crop(image_bgr)
    return cv2.resize(cropped_image, (224, 224))

def main(
    model_path: str = "data/experiments/mobile_openarm_ume/fridge/ckpt_step_40000/model.pth",
    perception_redis_host: str = "localhost",
    perception_redis_port: int = 6380,
    control_redis_host: str = "localhost",
    control_redis_port: int = 6379,
    reference_camera_key: str = "camera_head",
    name: str = "model_inference"
):
    print("Loading model", model_path)
    act_model = load_model(model_path)
    print(act_model)
    print("Model loaded successfully.")
    
    perception_client = RedisClient(host=perception_redis_host, port=perception_redis_port)
    control_client = RedisClient(host=control_redis_host, port=control_redis_port)
    
    while True:
        t_loop_start = time.time()
        print("loop start", t_loop_start)
        
        # prepare image observations
        image_inputs = perception_client.stream_get_batch(
            stream_keys={
                "camera_head:timestamp": float,
                "camera_head:image": np.ndarray,
                "camera_wrist_left:timestamp": float,
                "camera_wrist_left:image": np.ndarray,
                "camera_wrist_right:timestamp": float,
                "camera_wrist_right:image": np.ndarray,
            }
        )
        # prepare lowdim observations (current qpos)
        lowdim_inputs = control_client.stream_get_batch(
            stream_keys={
                "openarm_follower_left_state:timestamp": np.ndarray,
                "openarm_follower_left_state:actual_angle_rad": np.ndarray,
                
                "openarm_follower_right_state:timestamp": np.ndarray,
                "openarm_follower_right_state:actual_angle_rad": np.ndarray,
                
                "openarm_controller_state:timestamp": float,
                "openarm_controller_state:tau_interaction_L": np.ndarray,
                "openarm_controller_state:tau_interaction_R": np.ndarray,
            }
        )
        
        
        obs_image_head_bgr = image_inputs["camera_head:image"]
        obs_image_wrist_left_bgr = image_inputs["camera_wrist_left:image"]
        obs_image_wrist_right_bgr = image_inputs["camera_wrist_right:image"]
        reference_timestamp = image_inputs[f"{reference_camera_key}:timestamp"]
        
        obs_actual_qpos_left = lowdim_inputs["openarm_follower_left_state:actual_angle_rad"]
        # obs_desired_qpos_left = lowdim_inputs["openarm_follower_left_mit_command:q"]
        obs_actual_qpos_right = lowdim_inputs["openarm_follower_right_state:actual_angle_rad"]
        # obs_desired_qpos_right = lowdim_inputs["openarm_follower_right_mit_command:q"]
        obs_tau_interaction_L = lowdim_inputs["openarm_controller_state:tau_interaction_L"]
        obs_tau_interaction_R = lowdim_inputs["openarm_controller_state:tau_interaction_R"]
        
        obs_image_head_bgr_cropped = image_crop_and_resize(obs_image_head_bgr)
        obs_image_wrist_left_bgr_cropped = image_crop_and_resize(obs_image_wrist_left_bgr)
        obs_image_wrist_right_bgr_cropped = image_crop_and_resize(obs_image_wrist_right_bgr)
        print("obs_image_head_bgr_cropped shape:", obs_image_head_bgr_cropped.shape)
        print("obs_image_wrist_left_bgr_cropped shape:", obs_image_wrist_left_bgr_cropped.shape)
        print("obs_image_wrist_right_bgr_cropped shape:", obs_image_wrist_right_bgr_cropped.shape)
        
        obs_image_vis = np.hstack([obs_image_wrist_left_bgr_cropped, obs_image_head_bgr_cropped, obs_image_wrist_right_bgr_cropped])
        
        cv2.imshow("Observations (Wrist Left | Head | Wrist Right)", cv2.resize(obs_image_vis, (2400, 800)))
        cv2.waitKey(1)
        
        # get RGB images for model input
        obs_image_head_rgb = obs_image_head_bgr_cropped[..., [2, 1, 0]]
        obs_image_wrist_left_rgb = obs_image_wrist_left_bgr_cropped[..., [2, 1, 0]]
        obs_image_wrist_right_rgb = obs_image_wrist_right_bgr_cropped[..., [2, 1, 0]]
        
        joint_state = np.concatenate([obs_actual_qpos_left, obs_tau_interaction_L, obs_actual_qpos_right, obs_tau_interaction_R], axis=-1)
        obs_dict = {
            "camera_head": torch.from_numpy(obs_image_head_rgb[None, ...]).permute(0, 3, 1, 2).float().cuda().unsqueeze(0) / 255.0,  # (1, 1, C, H, W)
            "camera_wrist_left": torch.from_numpy(obs_image_wrist_left_rgb[None, ...]).permute(0, 3, 1, 2).float().cuda().unsqueeze(0) / 255.0,  # (1, 1, C, H, W)
            "camera_wrist_right": torch.from_numpy(obs_image_wrist_right_rgb[None, ...]).permute(0, 3, 1, 2).float().cuda().unsqueeze(0) / 255.0,  # (1, 1, C, H, W)
            "joint_state": torch.from_numpy(joint_state[None, ...]).float().cuda().unsqueeze(0)  # (1, 1, 16)
        }
        
        for key, value in obs_dict.items():
            print(f"{key}: {value.shape}")
        
        action_predicted = act_model.predict_action(obs_dict)
        action_timestamp = reference_timestamp + np.linspace(0, 2, 20)

        output_data = {
            f"{name}:action_timestamp": action_timestamp,
            f"{name}:action_desired_qpos_left": action_predicted["openarm_left_action"].cpu().numpy()[0],  # (16, 8)
            f"{name}:action_desired_qpos_right": action_predicted["openarm_right_action"].cpu().numpy()[0],  # (16, 8)
            f"{name}:action_desired_base_velocity_xyt": action_predicted["base_action"].cpu().numpy()[0]  # (16, 3)
        }
        output_data_maxlen = {
            f"{name}:action_timestamp": 1,
            f"{name}:action_desired_qpos_left": 1,
            f"{name}:action_desired_qpos_right": 1,
            f"{name}:action_desired_base_velocity_xyt": 1
        }
        control_client.stream_add_batch(output_data, batch_maxlen=output_data_maxlen)
        
        t_loop_end = time.time()
        print("model inference worker ready")
        print(f"Inference loop took {t_loop_end - t_loop_start:.3f} seconds.")
        print(f"Inference frequency: {1.0 / (t_loop_end - t_loop_start):.2f} Hz.")
        print(f"desired base velocity l2 norm", np.linalg.norm(action_predicted["base_action"].cpu().numpy(), axis=-1))
        print(f"desired qpos left delta max abs", np.rad2deg(np.abs(action_predicted["openarm_left_action"].cpu().numpy()[0, -1] - obs_actual_qpos_left).max()))
        print(f"desired qpos right delta max abs", np.rad2deg(np.abs(action_predicted["openarm_right_action"].cpu().numpy()[0, -1] - obs_actual_qpos_right).max()))

if __name__ == "__main__":
    typer.run(main)


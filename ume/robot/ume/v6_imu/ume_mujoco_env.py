import numpy as np
from ume.damiao_chain import DamiaoChain, MotorType
from ume.tools.precise_sleep import FrequencyRegulator

from ume.sim.mujoco_env import MujocoBaseEnv, mujoco, mink
from dm_control import mjcf
from ume.robot.ume.v6_imu.models import UME_MODEL_PATH


class UMEPlayEnv(MujocoBaseEnv):
    
    def construct_model(self):
        
        self.robot_model = mjcf.from_path(UME_MODEL_PATH)
        
        root = self.robot_model
        self.add_coordinate_frame_mocap(root, "world_origin", size=0.05)
        # self.add_coordinate_frame_mocap(root, "imu", size=0.05)
        
        self.add_coordinate_frame_mocap(root, "R_shoulder", size=0.05)
        self.add_coordinate_frame_mocap(root, "R_wrist", size=0.05)
        
        self.add_coordinate_frame_mocap(root, "L_shoulder", size=0.05)
        self.add_coordinate_frame_mocap(root, "L_wrist", size=0.05)

        model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())
        configuration = mink.Configuration(model)
        return model, configuration

    def get_base_pose(self):
        return self.get_body_pose("dm_j4340_2ec")

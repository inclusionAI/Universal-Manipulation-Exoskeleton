import numpy as np
import pinocchio as pin
from ume.robot.openarm1.models import OPENARM_MODEL_PATH


class OpenArm1Solver:
    
    q_slice_arm = np.concatenate([np.arange(7), np.arange(10, 17)])         # 14
    q_slice_arm_gripper = np.concatenate([np.arange(8), np.arange(9, 17)])  # 16
    
    ee_L_name = "openarm_left_hand"
    ee_R_name = "openarm_right_hand"

    def __init__(self):
        self.model = pin.buildModelFromMJCF(OPENARM_MODEL_PATH)
        self.data = self.model.createData()
        self.ee_L_id = self.model.getFrameId(self.ee_L_name)
        self.ee_R_id = self.model.getFrameId(self.ee_R_name)

    def forward_kinematics(self, qpos):
        pin.framesForwardKinematics(self.model, self.data, qpos)
    
    def rnea(self, qpos, qvel, qacc):
        tau = pin.rnea(self.model, self.data, qpos, qvel, qacc)
        return tau

    def get_ee_L_pose(self):
        return self.data.oMf[self.ee_L_id].homogeneous

    def get_ee_R_pose(self):
        return self.data.oMf[self.ee_R_id].homogeneous

    def get_gravity_compensation_torques(self, qpos, qvel, qacc):
        tau_gravity = self.rnea(qpos, qvel, qacc)[self.q_slice_arm_gripper]
        return tau_gravity

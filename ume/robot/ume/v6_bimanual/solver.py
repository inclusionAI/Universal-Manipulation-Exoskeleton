import numpy as np
import pinocchio as pin
from ume.robot.ume.v6_bimanual.models import UME_MODEL_PATH


class UMESolver:

    def __init__(
        self
    ):
        self.model = pin.buildModelFromMJCF(UME_MODEL_PATH)
        self.data = self.model.createData()
        # for i, frame in enumerate(self.model.frames):
        #     print(f"ID {i}: {frame.name}")
        self.ball_joint_name_to_id = {
            "R_shoulder": self.model.getFrameId("R_shoulder"),
            "R_wrist":    self.model.getFrameId("R_wrist"),
            "L_shoulder": self.model.getFrameId("L_shoulder"),
            "L_wrist":    self.model.getFrameId("L_wrist"),
        }
        self.ball_joint_q_slice = {
            "R_shoulder": slice(0, 3),
            "R_wrist":    slice(4, 7),
            "L_shoulder": slice(8, 11),
            "L_wrist":    slice(12, 15)
        }

    def forward_kinematics(self, qpos):
        pin.framesForwardKinematics(self.model, self.data, qpos)

    def rnea(self, qpos, qvel, qacc):
        tau = pin.rnea(self.model, self.data, qpos, qvel, qacc)
        return tau

    def get_ball_joint_pose(self, joint_name):
        assert joint_name in self.ball_joint_name_to_id
        link_id = self.ball_joint_name_to_id[joint_name]
        return self.data.oMf[link_id].homogeneous
    
    def get_ball_joint_xdot(self, qpos, joint_name, qvel):
        assert joint_name in self.ball_joint_name_to_id
        link_id = self.ball_joint_name_to_id[joint_name]
        J_mat = pin.computeFrameJacobian(self.model, self.data, qpos, link_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_rot_mat = J_mat[3:, self.ball_joint_q_slice[joint_name]]
        xdot = J_rot_mat @ qvel[self.ball_joint_q_slice[joint_name]]
        return xdot

    def get_ball_joint_torque(self, qpos, joint_name, wrench):
        # print("Computing wrench in world frame")
        assert joint_name in self.ball_joint_name_to_id
        link_id = self.ball_joint_name_to_id[joint_name]
        J_mat = pin.computeFrameJacobian(self.model, self.data, qpos, link_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_rot_mat = J_mat[3:, self.ball_joint_q_slice[joint_name]]
        tau = J_rot_mat.T @ wrench
        return tau


if __name__ == "__main__":
    pass


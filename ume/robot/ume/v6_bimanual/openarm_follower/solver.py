import numpy as np
import pinocchio as pin
from ume.robot.openarm1.models import OPENARM_MODEL_PATH


class OpenArm313Solver:

    def __init__(self):
        self.model = pin.buildModelFromMJCF(OPENARM_MODEL_PATH)
        self.data = self.model.createData()
        for i, frame in enumerate(self.model.frames):
            print(f"ID {i}: {frame.name}")
        
        self.ball_joint_name_to_id = {
            "R_shoulder": self.model.getFrameId("R_shoulder"),
            "R_wrist":    self.model.getFrameId("R_wrist"),
            
            "L_shoulder": self.model.getFrameId("L_shoulder"),
            "L_wrist":    self.model.getFrameId("L_wrist"),
        }
        
        self.ball_joint_q_slice = {
            "R_shoulder":  np.arange(9, 12),
            "R_elbow":     np.arange(12, 13),
            "R_wrist":     np.arange(13, 16),
            
            "L_shoulder":  np.arange(0, 3),
            "L_elbow":     np.arange(3, 4),
            "L_wrist":     np.arange(4, 7)
        }

    def forward_kinematics(self, qpos):
        pin.framesForwardKinematics(self.model, self.data, qpos)
    
    def rnea(self, qpos, qvel, qacc):
        tau = pin.rnea(self.model, self.data, qpos, qvel, qacc)
        return tau
    
    def get_ball_joint_pose(self, joint_name):
        assert joint_name in self.ball_joint_name_to_id
        frame_id = self.ball_joint_name_to_id[joint_name]
        return self.data.oMf[frame_id].homogeneous

    def ik_ball_joint(self, qpos, joint_name, desired_pose, step_size=0.5):
        # identify the frame and joint slices
        frame_id = self.ball_joint_name_to_id[joint_name]
        q_slice = self.ball_joint_q_slice[joint_name]
        # compute jacobian matrix
        J_mat = pin.computeFrameJacobian(self.model, self.data, qpos, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_rot_mat = J_mat[3:, q_slice]
        # compute rotation error in local world aligned frame
        curr_ball_joint_pose = self.data.oMf[frame_id].homogeneous
        curr_ball_joint_rot = curr_ball_joint_pose[:3, :3]
        desired_ball_joint_rot = desired_pose[:3, :3]
        err_rot = curr_ball_joint_rot @ pin.log3(curr_ball_joint_rot.T @ desired_ball_joint_rot)
        # compute delta
        inv_J_rot = np.linalg.pinv(J_rot_mat)
        delta_qpos = step_size * inv_J_rot @ err_rot
        return qpos[self.ball_joint_q_slice[joint_name]] + delta_qpos, err_rot
    
    def qvel_ball_joint(self, qpos, joint_name, xdot):
        assert joint_name in self.ball_joint_name_to_id
        frame_id = self.ball_joint_name_to_id[joint_name]
        J_mat = pin.computeFrameJacobian(self.model, self.data, qpos, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_rot_mat = J_mat[3:, self.ball_joint_q_slice[joint_name]]
        qvel = np.linalg.pinv(J_rot_mat) @ xdot
        return qvel

    def wrench_ball_joint(self, qpos, joint_name, tau):
        # print("Computing wrench in world frame")
        assert joint_name in self.ball_joint_name_to_id
        frame_id = self.ball_joint_name_to_id[joint_name]
        J_mat = pin.computeFrameJacobian(self.model, self.data, qpos, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_rot_mat = J_mat[3:, self.ball_joint_q_slice[joint_name]]
        return np.linalg.pinv(J_rot_mat.T) @ tau


if __name__ == "__main__":
    solver = OpenArm313Solver()
    qpos = np.zeros(18)
    L_shoulder_pose = solver.get_ball_joint_pose("L_shoulder")
    print("L_shoulder_pose:\n", L_shoulder_pose)

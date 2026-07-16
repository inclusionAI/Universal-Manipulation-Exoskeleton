import numpy as np
import pinocchio as pin
from ume.robot.openarm1.models import OPENARM_MODEL_PATH


class OpenArmIKSolver:
    
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

    def get_ik(self, desired_L_ee_pose, desired_R_ee_pose, q_init=None, max_iter=100, eps=1e-4, damping=1e-2, max_step=0.05):
        oMf_L_des = pin.SE3(desired_L_ee_pose)
        oMf_R_des = pin.SE3(desired_R_ee_pose)
        
        q = q_init if q_init is not None else pin.neutral(self.model)
        
        # joint limits
        q_min = self.model.lowerPositionLimit
        q_max = self.model.upperPositionLimit
        
        success = False
        last_error_norm = float('inf')
        
        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            oMf_L_curr = self.data.oMf[self.ee_L_id]
            oMf_R_curr = self.data.oMf[self.ee_R_id]
            
            dL_Mf = oMf_L_curr.actInv(oMf_L_des)
            dR_Mf = oMf_R_curr.actInv(oMf_R_des)
            
            error_stacked = np.concatenate([pin.log6(dL_Mf).vector, pin.log6(dR_Mf).vector])
            error_norm = np.linalg.norm(error_stacked)
            
            if error_norm < eps:
                success = True
                break
                
            if abs(last_error_norm - error_norm) < 1e-6:
                break
            
            last_error_norm = error_norm
                
            J_L = pin.computeFrameJacobian(self.model, self.data, q, self.ee_L_id, pin.ReferenceFrame.LOCAL)
            J_R = pin.computeFrameJacobian(self.model, self.data, q, self.ee_R_id, pin.ReferenceFrame.LOCAL)
            J_stacked = np.vstack([J_L, J_R])
            
            JJT = J_stacked @ J_stacked.T
            damping_matrix = (damping ** 2) * np.eye(JJT.shape[0])
            
            delta_q = J_stacked.T @ np.linalg.solve(JJT + damping_matrix, error_stacked)
            delta_q = np.clip(delta_q, -max_step, max_step)
            q = pin.integrate(self.model, q, delta_q)
            q = np.clip(q, q_min, q_max)
            
        return q, success
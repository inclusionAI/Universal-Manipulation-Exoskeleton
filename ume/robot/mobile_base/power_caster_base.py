import math
import numpy as np

import cv2
import numpy as np

class PowerCasterBaseKinematics:
    def __init__(self, n_casters, wheel_bx, wheel_by, wheel_hx, wheel_hy, wheel_r):
        
        # Constants
        self.n_casters = n_casters
        self.wheel_bx = wheel_bx
        self.wheel_by = wheel_by
        self.wheel_hx = wheel_hx
        self.wheel_hy = wheel_hy
        self.wheel_r = wheel_r
        
        # Joint space
        num_motors = 2 * self.n_casters
        self.q = np.zeros(num_motors)
        self.dq = np.zeros(num_motors)
        self.tau = np.zeros(num_motors)

        # Operational space (global frame)
        num_dofs = 3  # (x, y, theta)
        self.x = np.zeros(num_dofs)
        self.dx = np.zeros(num_dofs)

        # C matrix relating operational space velocities to joint velocities
        self.C = np.zeros((num_motors, num_dofs))
        self.C_steer = self.C[::2]
        self.C_drive = self.C[1::2]

        # C_p matrix relating operational space velocities to wheel velocities at the contact points
        self.C_p = np.zeros((num_motors, num_dofs))
        self.C_p_steer = self.C_p[::2]
        self.C_p_drive = self.C_p[1::2]
        self.C_p_steer[:, :2] = [1.0, 0.0]
        self.C_p_drive[:, :2] = [0.0, 1.0]

        # C_qp^# matrix relating joint velocities to operational space velocities
        self.C_pinv = np.zeros((num_motors, num_dofs))
        self.CpT_Cqinv = np.zeros((num_dofs, num_motors))
        self.CpT_Cqinv_steer = self.CpT_Cqinv[:, ::2]
        self.CpT_Cqinv_drive = self.CpT_Cqinv[:, 1::2]
    
    @classmethod
    def rx_local_global(cls, theta):
        R_3X3 = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [            0,              0, 1]
        ])
        return R_3X3
    
    def update_state(self, qpos, qvel, dt):
        # Joint positions and velocities
        self.q = qpos
        
        q_steer = self.q[::2]
        s = np.sin(q_steer)
        c = np.cos(q_steer)

        # C matrix
        self.C_steer[:, 0] = s / self.wheel_bx
        self.C_steer[:, 1] = -c / self.wheel_bx
        self.C_steer[:, 2] = (-self.wheel_hx * c - self.wheel_hy * s) / self.wheel_bx - 1.0
        self.C_drive[:, 0] = c / self.wheel_r - self.wheel_by * s / (self.wheel_bx * self.wheel_r)
        self.C_drive[:, 1] = s / self.wheel_r + self.wheel_by * c / (self.wheel_bx * self.wheel_r)
        self.C_drive[:, 2] = (self.wheel_hx * s - self.wheel_hy * c) / self.wheel_r + self.wheel_by * (
            self.wheel_hx * c + self.wheel_hy * s
        ) / (self.wheel_bx * self.wheel_r)

        # C_p matrix
        self.C_p_steer[:, 2] = -self.wheel_bx * s - self.wheel_by * c - self.wheel_hy
        self.C_p_drive[:, 2] = self.wheel_bx * c - self.wheel_by * s + self.wheel_hx

        # C_qp^# matrix
        self.CpT_Cqinv_steer[0] = self.wheel_bx * s + self.wheel_by * c
        self.CpT_Cqinv_steer[1] = -self.wheel_bx * c + self.wheel_by * s
        self.CpT_Cqinv_steer[2] = (
              self.wheel_bx * (-self.wheel_hx * c - self.wheel_hy * s - self.wheel_bx)
            + self.wheel_by * ( self.wheel_hx * s - self.wheel_hy * c - self.wheel_by)
        )
        self.CpT_Cqinv_drive[0] = self.wheel_r * c
        self.CpT_Cqinv_drive[1] = self.wheel_r * s
        self.CpT_Cqinv_drive[2] = self.wheel_r * (self.wheel_hx * s - self.wheel_hy * c - self.wheel_by)
        self.C_pinv = np.linalg.solve(self.C_p.T @ self.C_p, self.CpT_Cqinv)
        
        # odometry (velocity integration)
        self.dq = qvel
        dx_local = self.C_pinv @ self.dq
        theta_rk2 = self.x[2] + dx_local[2] * 0.5 * dt
        R_3X3 = self.rx_local_global(theta_rk2)
        self.dx = R_3X3 @ dx_local
        self.x += self.dx * dt

    def operational_space_velocity(self, joint_velocity: np.ndarray) -> np.ndarray:
        dx_local = self.C_pinv @ joint_velocity
        return dx_local

    def joint_velocity(self, desired_velocity: np.ndarray) -> np.ndarray:
        q_vel = self.C @ desired_velocity
        return q_vel

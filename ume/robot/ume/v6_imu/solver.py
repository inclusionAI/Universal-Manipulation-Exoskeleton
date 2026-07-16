import numpy as np
import pinocchio as pin
from ume.robot.ume.v6_imu.models import UME_MODEL_PATH


class UMESolver:

    def __init__(
        self
    ):
        self.model = pin.buildModelFromMJCF(UME_MODEL_PATH)
        self.data = self.model.createData()

    def forward_kinematics(self, qpos):
        pin.framesForwardKinematics(self.model, self.data, qpos)

    def rnea(self, qpos, qvel, qacc):
        tau = pin.rnea(self.model, self.data, qpos, qvel, qacc)
        return tau


if __name__ == "__main__":
    pass


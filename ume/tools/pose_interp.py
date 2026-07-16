import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_interp1d(t, x):
    gripper_interp = si.interp1d(t, x, axis=0, bounds_error=False, fill_value=(x[0], x[-1]))
    return gripper_interp


class PoseInterpolator:
    def __init__(self, t, x, repr="mat44"):
        if isinstance(x, list):
            x = np.stack(x)

        pos = None
        rot = None
        if repr == "vec6":
            pos = x[:, :3]
            rot = st.Rotation.from_rotvec(x[:, 3:])
        elif repr == "mat44":
            pos = x[:, :3, 3]
            rot = st.Rotation.from_matrix(x[:, :3, :3])
        else:
            raise NotImplementedError()

        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
        self.repr = repr
        self.y = x

    @property
    def x(self):
        return self.pos_interp.x

    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        if self.repr == "vec6":
            rvec = rot.as_rotvec()
            pose = np.concatenate([pos, rvec], axis=-1)
            return pose
        elif self.repr == "mat44":
            mat = np.zeros((pos.shape[0], 4, 4), dtype=pos.dtype)
            mat[:, :3, :3] = rot.as_matrix()
            mat[:, :3, 3] = pos
            mat[:, 3, 3] = 1
            return mat
        else:
            return RuntimeError()


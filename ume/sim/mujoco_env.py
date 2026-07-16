import mink
import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf


def add_coordinate_frame_mocap(root, name, size=0.05):
    width, length = size / 10, size
    body = root.worldbody.add("body", name=name, mocap=True)
    body.add(
        "geom",
        name=f"{name}_geom_x",
        type="cylinder",
        size=f"{width} {length}",
        quat="0 1 0 1",
        pos=f"{length} 0 0",
        rgba="1 0 0 1",
        contype="0",
        conaffinity="0",
    )
    body.add(
        "geom",
        name=f"{name}_geom_y",
        type="cylinder",
        size=f"{width} {length}",
        quat="0 0 1 1",
        pos=f"0 {length} 0",
        rgba="0 1 0 1",
        contype="0",
        conaffinity="0",
    )
    body.add(
        "geom",
        name=f"{name}_geom_z",
        type="cylinder",
        size=f"{width} {length}",
        quat="1 0 0 0",
        pos=f"0 0 {length}",
        rgba="0 0 1 1",
        contype="0",
        conaffinity="0",
    )


def add_cube_mocap(root, name, rgba="1 0 0 1", size=0.05):
    body = root.worldbody.add("body", name=name, mocap=True)
    body.add(
        "geom",
        name=f"{name}_geom",
        type="box",
        size=f"{size} {size} {size}",
        rgba=rgba,
        contype="0",
        conaffinity="0",
    )


def move_mocap_to_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    pose: np.ndarray,
) -> None:
    mocap_id = model.body(mocap_name).mocapid[0]
    if mocap_id == -1:
        raise NotImplementedError(f"mocap {mocap_name} not found")
    xpos = pose[:3, 3]
    xmat = pose[:3, :3].reshape(-1)
    data.mocap_pos[mocap_id] = xpos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], xmat)


class MujocoBaseEnv:
    def __init__(self):
        self.coord_frame_mocap_names = []
        self.cube_mocap_names = []
        self.model, self.configuration = self.construct_model()
        self.viewer = None

    @staticmethod
    def get_chessboard_floor():
        return mjcf.from_path("ume/sim/assets/chessboard.xml")

    def add_coordinate_frame_mocap(self, root, name, size=0.05):
        add_coordinate_frame_mocap(root, name, size)
        self.coord_frame_mocap_names.append(name)

    def add_cube_mocap(self, root, name, rgba="1 0 0 1", size=0.05):
        add_cube_mocap(root, name, rgba, size)
        self.cube_mocap_names.append(name)

    def construct_model(self):
        root = self.get_chessboard_floor()
        self.add_coordinate_frame_mocap(root, "world_origin", size=0.05)
        self.add_coordinate_frame_mocap(root, "spacemouse", size=0.05)
        # self.add_cube_mocap(root, "cube", size=0.05)
        model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())
        configuration = mink.Configuration(model)
        return model, configuration

    def clear_cubes(self):
        far_away_pose = np.eye(4)
        far_away_pose[:3, 3] = [0, 0, 10000]
        for cube_name in self.cube_mocap_names:
            self.set_mocap_pose(cube_name, far_away_pose)

    def get_mocap_pose(self, name):
        self.update()
        pose = mink.SE3.from_mocap_name(self.model, self.configuration.data, name).as_matrix()
        return pose

    def get_qpos(self):
        return self.configuration.data.qpos.copy()
    
    def set_qpos(self, qpos):
        self.configuration.data.qpos[:] = qpos
    
    def set_tau(self, tau):
        self.configuration.data.ctrl[:] = tau

    def get_qvel(self):
        return self.configuration.data.qvel.copy()
    
    def get_qacc(self):
        return self.configuration.data.qacc.copy()

    def get_body_pose(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        # print(body_id)
        if body_id == -1:
            raise ValueError(f"body {name} not found")
        pos = self.configuration.data.xpos[body_id]
        rot = self.configuration.data.xmat[body_id].reshape(3, 3)
        pose_mat44 = np.eye(4)
        pose_mat44[:3, 3] = pos
        pose_mat44[:3, :3] = rot
        return pose_mat44
    
    def get_site_pose(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        # print(site_id)
        if site_id == -1:
            raise ValueError(f"site {name} not found")
        pos = self.configuration.data.site_xpos[site_id]
        rot = self.configuration.data.site_xmat[site_id].reshape(3, 3)
        pose_mat44 = np.eye(4)
        pose_mat44[:3, 3] = pos
        pose_mat44[:3, :3] = rot
        return pose_mat44
    
    def get_neutral_qpos(self):
        return self.model.qpos0

    def update(self):
        mujoco.mj_fwdPosition(self.model, self.configuration.data)

    def step(self):
        mujoco.mj_step(self.model, self.configuration.data)

    def set_mocap_pose(self, name, pose):
        assert pose.shape == (4, 4)
        move_mocap_to_pose(self.model, self.configuration.data, name, pose)
        self.update()

    def launch_viewer(self):
        viewer = mujoco.viewer.launch_passive(
            model=self.model, data=self.configuration.data, show_left_ui=False, show_right_ui=False
        )
        return viewer


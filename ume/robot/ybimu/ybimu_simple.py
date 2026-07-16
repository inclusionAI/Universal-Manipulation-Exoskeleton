import time
import numpy as np
import scipy.spatial.transform as st
from ume.robot.ybimu.YbImuSerialLib import YbImuSerial

class YbImu:
    
    def __init__(self):
        import platform
        device = platform.system()
        print("Read device:", device)
        bot = None
        if device == 'Windows':
            com_index = 1
            while True:
                com_index = com_index + 1
                try:
                    print("try COM%d" % com_index)
                    port = 'COM%d' % com_index
                    bot = YbImuSerial(port, debug=True)
                    break
                except:
                    if com_index > 256:
                        print("-----------------------No COM Open--------------------------")
                        exit()
                    continue
            print("--------------------Open %s---------------------" % port)
        else:
            port_list = ["/dev/myserial", "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyTHS1", "/dev/ttyAMA0"]
            for port in port_list:
                try:
                    bot = YbImuSerial(port, debug=True)
                    print("Open Ybimu port OK:%s" % port)
                    break
                except:
                    pass
        if bot is None:
            print("Fail To Open Serial")
            exit()
        
        bot.create_receive_threading()

        version = bot.get_version()
        print("version=", version)
        
        self.bot = bot
        self.base_transform = st.Rotation.from_rotvec([0, np.pi, 0]).as_matrix()
    
    def get_pose(self):
        quat = self.bot.get_imu_quaternion_data()
        quat_wxyz = np.array(quat)
        quat_xyzw = quat_wxyz[[3, 0, 1, 2]]
        quat_xyzw[0] *= -1
        quat_xyzw[1] *= -1
        R = st.Rotation.from_quat(quat_xyzw).as_matrix()
        tx_world_imu = np.eye(4)
        tx_world_imu[:3, :3] = self.base_transform @ R
        return tx_world_imu
    
    def get_accelerometer(self):
        accel = self.bot.get_accelerometer_data()
        return np.array(accel)


if __name__ == "__main__":
    
    from ume.sim.mujoco_env import MujocoBaseEnv, mujoco, mink
    import scipy.spatial.transform as st
    
    class IMUPlayEnv(MujocoBaseEnv):
    
        def construct_model(self):
            root = self.get_chessboard_floor()
            self.add_coordinate_frame_mocap(root, "world_origin", size=0.05)
            self.add_coordinate_frame_mocap(root, "imu", size=0.05)

            model = mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())
            configuration = mink.Configuration(model)
            return model, configuration
    
    sim = IMUPlayEnv()
    sim_viewer = sim.launch_viewer()
    
    rot_x_90 = st.Rotation.from_rotvec([np.pi/2, 0, 0])
    rot_z_90 = st.Rotation.from_rotvec([0, 0, np.pi/2])
    mount_transform = np.eye(4)
    mount_transform[:3, :3] = rot_x_90.as_matrix() @ rot_z_90.as_matrix()

    yb_imu = YbImu()
    while True:
        pose = yb_imu.get_pose()
        pose = pose @ mount_transform
        pose[2, 3] = 0.2
        print("Current Pose:\n", pose)
        sim.set_mocap_pose("imu", pose)
        sim.update()
        sim_viewer.sync()
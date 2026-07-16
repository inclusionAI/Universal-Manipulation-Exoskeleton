#!/usr/bin/env python3
# coding: utf-8

import struct
import time
import serial
import threading
import math


# V1.0.0
class YbImuSerial(object):

    def __init__(self, port, debug=False):
        # port = "COM30"
        # port="/dev/ttyTHS1"
        # port="/dev/ttyUSB0"
        # port="/dev/ttyAMA0"

        self._dev = serial.Serial(str(port), 115200)

        self._debug = debug

        self._HEAD1 = 0x7E
        self._HEAD2 = 0x23

        self.FUNC_VERSION = 0x01

        self.FUNC_REPORT_IMU_RAW = 0x04
        self.FUNC_REPORT_IMU_QUAT = 0x16
        self.FUNC_REPORT_IMU_EULER = 0x26
        self.FUNC_REPORT_BARO = 0x32

        self.FUNC_REPORT_RATE = 0x60
        self.FUNC_ALGO_TYPE = 0x61

        self.FUNC_CALIB_IMU = 0x70
        self.FUNC_CALIB_MAG = 0x71
        self.FUNC_CALIB_BARO = 0x72
        self.FUNC_CALIB_TEMP = 0x73

        self.FUNC_REQUEST_DATA = 0x80
        self.FUNC_RETURN_STATE = 0x81

        self.FUNC_RESET_FLASH = 0xA0

        self._RX_MAX_LEN = 40

        
        self._ax = 0.0
        self._ay = 0.0
        self._az = 0.0
        self._gx = 0.0
        self._gy = 0.0
        self._gz = 0.0
        self._mx = 0.0
        self._my = 0.0
        self._mz = 0.0

        self._yaw = 0.0
        self._roll = 0.0
        self._pitch = 0.0

        self._q0 = 0.0
        self._q1 = 0.0
        self._q2 = 0.0
        self._q3 = 0.0

        self._height = 0.0
        self._temperature = 0.0
        self._pressure = 0.0
        self._pressure_contrast = 0.0

        self._rx_func = 0
        self._rx_state = 0
        self.rx_flag = 0


        self._version_H = -1
        self._version_M = -1
        self._version_L = -1


        if self._dev.isOpen():
            print("YbImu Serial Opened! Baudrate=115200")
        else:
            print("Open YbImu Serial Failed!")


    def __del__(self):
        try:
            self._dev.close()
            print("YbImu Serial Close!")
        except:
            pass


    def _print_log(self, cmd_data, log="Send"):
        if not self._debug:
            return
        # print ("Send: [0x" + ', 0x'.join('{:02X}'.format(x) for x in cmd_data) + "]")
        print (str(log) + ": [" + ''.join('{:02X}'.format(x) for x in cmd_data) + "]")

    def _send_data(self, cmd_data):
        self._dev.write(cmd_data)

    # 根据数据帧的类型来做出对应的解析
    # According to the type of data frame to make the corresponding parsing
    def _parse_data(self, ext_type, ext_data):
        # print("parse_data:", ext_data, ext_type)
        
        # 解析原始陀螺仪、加速度计、磁力计数据
        # the original gyroscope, accelerometer, magnetometer data
        if ext_type == self.FUNC_REPORT_IMU_RAW:
            # 转化单位为g
            accel_ratio = 16 / 32767.0
            self._ax = struct.unpack('h', bytearray(ext_data[0:2]))[0]*accel_ratio
            self._ay = struct.unpack('h', bytearray(ext_data[2:4]))[0]*accel_ratio
            self._az = struct.unpack('h', bytearray(ext_data[4:6]))[0]*accel_ratio

            # 转化单位为rad/s
            AtoR = math.pi / 180.0
            gyro_ratio = (2000 / 32767.0) * AtoR
            self._gx = struct.unpack('h', bytearray(ext_data[6:8]))[0]*gyro_ratio
            self._gy = struct.unpack('h', bytearray(ext_data[8:10]))[0]*gyro_ratio
            self._gz = struct.unpack('h', bytearray(ext_data[10:12]))[0]*gyro_ratio

            # 转化单位为uT
            mag_ratio = 800.0 / 32767.0
            self._mx = struct.unpack('h', bytearray(ext_data[12:14]))[0]*mag_ratio
            self._my = struct.unpack('h', bytearray(ext_data[14:16]))[0]*mag_ratio
            self._mz = struct.unpack('h', bytearray(ext_data[16:18]))[0]*mag_ratio
        # 解析板子的姿态角
        # the attitude Angle of the board
        elif ext_type == self.FUNC_REPORT_IMU_EULER:
            self._roll = struct.unpack('f', bytearray(ext_data[0:4]))[0]
            self._pitch = struct.unpack('f', bytearray(ext_data[4:8]))[0]
            self._yaw = struct.unpack('f', bytearray(ext_data[8:12]))[0]
        # 解析IMU的四元数
        # the quaternion of IMU
        elif ext_type == self.FUNC_REPORT_IMU_QUAT:
            self._q0 = struct.unpack('f', bytearray(ext_data[0:4]))[0]
            self._q1 = struct.unpack('f', bytearray(ext_data[4:8]))[0]
            self._q2 = struct.unpack('f', bytearray(ext_data[8:12]))[0]
            self._q3 = struct.unpack('f', bytearray(ext_data[12:16]))[0]
        # 解析气压计数据
        elif ext_type == self.FUNC_REPORT_BARO:
            self._height = round(struct.unpack('f', bytearray(ext_data[0:4]))[0], 2)
            self._temperature = round(struct.unpack('f', bytearray(ext_data[4:8]))[0], 2)
            self._pressure = round(struct.unpack('f', bytearray(ext_data[8:12]))[0], 5)
            self._pressure_contrast = round(struct.unpack('f', bytearray(ext_data[12:16]))[0], 5)
        elif ext_type == self.FUNC_VERSION:
            self._version_H = struct.unpack('B', bytearray(ext_data[0:1]))[0]
            self._version_M = struct.unpack('B', bytearray(ext_data[1:2]))[0]
            self._version_L = struct.unpack('B', bytearray(ext_data[2:3]))[0]
        elif ext_type == self.FUNC_RETURN_STATE:
            self._rx_func = struct.unpack('B', bytearray(ext_data[0:1]))[0]
            self._rx_state = struct.unpack('B', bytearray(ext_data[1:2]))[0]


    # 接收数据 receive data
    def _receive_data(self, data):
        if self.rx_flag == 0:
            if data == self._HEAD1:
                self.rx_flag = 1
        elif self.rx_flag == 1:
            if data == self._HEAD2:
                self.rx_flag = 2
            else:
                self.rx_flag = 0
        elif self.rx_flag == 2:
            self.data_len = data
            if self.data_len <= self._RX_MAX_LEN:
                self.rx_flag = 3
            else:
                self.data_len = 0
                self.rx_flag = 0
        elif self.rx_flag == 3:
            self.data_func = data
            self.rx_data = []
            self.rx_count = 4
            self.rx_flag = 4
        elif self.rx_flag == 4:
            self.rx_data.append(data)
            self.rx_count = self.rx_count + 1
            if self.rx_count >= self.data_len - 1:
                self.rx_flag = 5
        elif self.rx_flag == 5:
            self.rx_flag = 0
            rx_check_num = data
            check_sum = self._HEAD1 + self._HEAD2 + self.data_len + self.data_func
            for a in self.rx_data:
                check_sum = a + check_sum
            check_sum = check_sum % 256
            if rx_check_num == check_sum:
                # print("parse_data", self.data_func, self.rx_data)
                self._parse_data(self.data_func, self.rx_data)
            else:
                if self._debug:
                    print("check sum error:", rx_check_num, check_sum)
                    print("data:", self.data_len, self.data_func, self.rx_data)


    def _data_handle(self):
        # 清空缓冲区
        self._dev.flushInput()
        while True:
            data_count = self._dev.inWaiting()
            if data_count <= 0:
                time.sleep(0.001)
                continue
            data_array = self._dev.read_all()
            for data in data_array:
                self._receive_data(data)


    # 请求数据， function：对应要返回数据的功能字，parm：传入的参数。
    # Request data, function: corresponding function word to return data, parm: parameter passed in
    def _request_data(self, function, param=0):
        cmd = [self._HEAD1, self._HEAD2, 0x00, self.FUNC_REQUEST_DATA, int(function) & 0xFF, int(param) & 0xFF]
        cmd[2] = len(cmd) + 1
        checksum = sum(cmd) & 0xFF
        cmd.append(checksum)
        self._send_data(cmd)
        self._print_log(cmd, "request")

    # 开启接收和处理数据的线程
    # Start the thread that receives and processes data
    def create_receive_threading(self):
        try:
            name1 = "task_serial_receive"
            task_receive = threading.Thread(target=self._data_handle, name=name1)
            task_receive.daemon=True
            task_receive.start()
            print("----------------create receive threading--------------")
            time.sleep(.05)
        except:
            print('---create_receive_threading error!---')

    

    # 清除单片机自动发送过来的缓存数据
    # Clear the cache data automatically sent by the MCU
    def clear_auto_report_data(self):
        self._ax = 0.0
        self._ay = 0.0
        self._az = 0.0
        self._gx = 0.0
        self._gy = 0.0
        self._gz = 0.0
        self._mx = 0.0
        self._my = 0.0
        self._mz = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._q0 = 0.0
        self._q1 = 0.0
        self._q2 = 0.0
        self._q3 = 0.0
        self._height = 0.0
        self._temperature = 0.0
        self._pressure = 0.0
        self._pressure_contrast = 0.0

    # 设置输出数据的频率。rate=[10, 100]
    def set_report_rate(self, rate):
        if rate < 10:
            rate = 10
        if rate > 100:
            rate = 100
        cmd = [self._HEAD1, self._HEAD2, 0x00, self.FUNC_REPORT_RATE, int(rate), 0x5F]
        cmd[2] = len(cmd) + 1
        checksum = sum(cmd) & 0xFF
        cmd.append(checksum)
        self._send_data(cmd)
        self._print_log(cmd, "report rate")
        time.sleep(1)

    # 设置融合算法为六轴算法或者九轴算法。algo=6或9
    def set_algo_type(self, algo):
        if algo != 6 and algo != 9:
            return
        cmd = [self._HEAD1, self._HEAD2, 0x00, self.FUNC_ALGO_TYPE, int(algo), 0x5F]
        cmd[2] = len(cmd) + 1
        checksum = sum(cmd) & 0xFF
        cmd.append(checksum)
        self._send_data(cmd)
        self._print_log(cmd, "algo type")
        time.sleep(1)


    def wait_calibration(self, func, name, timeout_ms=None):
        count = 0
        while True:
            if self._rx_func == func:
                if self._debug:
                    print("rx", name, "state:", self._rx_state)
                return self._rx_state
            time.sleep(0.001)
            if timeout_ms is not None:
                count = count + 1
                if count > timeout_ms:
                    if self._debug:
                        print(name, "timeout")
                    return None


    # 校准IMU（陀螺仪和加速度计）
    def calibration_imu(self):
        self._rx_func = 0
        self._rx_state = 0
        cmd = [self._HEAD1, self._HEAD2, 0x00, self.FUNC_CALIB_IMU, 0x01, 0x5F]
        cmd[2] = len(cmd) + 1
        checksum = sum(cmd) & 0xFF
        cmd.append(checksum)
        self._send_data(cmd)
        self._print_log(cmd, "cali imu")
        self.wait_calibration(self.FUNC_CALIB_IMU, "cali imu", 7000)


    # 校准磁力计
    def calibration_mag(self):
        self._rx_func = 0
        self._rx_state = 0
        cmd = [self._HEAD1, self._HEAD2, 0x00, self.FUNC_CALIB_MAG, 0x01, 0x5F]
        cmd[2] = len(cmd) + 1
        checksum = sum(cmd) & 0xFF
        cmd.append(checksum)
        self._send_data(cmd)
        self._print_log(cmd, "cali mag")
        self.wait_calibration(self.FUNC_CALIB_MAG, "cali mag")


    # 校准温度
    def calibration_temperature(self, now_temperature):
        self._rx_func = 0
        self._rx_state = 0
        if now_temperature > 50 or now_temperature < -50:
            return
        value = bytearray(struct.pack('h', int(now_temperature*100)))
        cmd = [self._HEAD1, self._HEAD2, 0x00, self.FUNC_CALIB_TEMP, value[0], value[1], 0x5F]
        cmd[2] = len(cmd) + 1
        checksum = sum(cmd) & 0xFF
        cmd.append(checksum)
        self._send_data(cmd)
        self._print_log(cmd, "cali temp")
        self.wait_calibration(self.FUNC_CALIB_TEMP, "cali temp", 2000)


    # 获取加速度计三轴数据，返回accel=[a_x, a_y, a_z]
    # Get accelerometer triaxial data, return accel=[a_x, a_y, a_z]
    def get_accelerometer_data(self):
        a_x, a_y, a_z = self._ax, self._ay, self._az
        accel = [a_x, a_y, a_z]
        # self._ax, self._ay, self._az = 0.0, 0.0, 0.0
        return accel

    # 获取陀螺仪三轴数据，返回gyro=[g_x, g_y, g_z]
    # Get the gyro triaxial data, return gyro=[g_x, g_y, g_z]
    def get_gyroscope_data(self):
        g_x, g_y, g_z = self._gx, self._gy, self._gz
        gyro = [g_x, g_y, g_z]
        # self._gx, self._gy, self._gz = 0.0, 0.0, 0.0
        return gyro

    # 获取磁力计三轴数据，返回mag=[m_x, m_y, m_z]
    def get_magnetometer_data(self):
        m_x, m_y, m_z = self._mx, self._my, self._mz
        mag = [m_x, m_y, m_z]
        # self._mx, self._my, self._mz = 0.0, 0.0, 0.0
        return mag

    # 获取板子姿态角，返回euler=[roll, pitch, yaw]
    # ToAngle=True返回角度，ToAngle=False返回弧度。
    def get_imu_attitude_data(self, ToAngle=True):
        if ToAngle:
            RtA = 57.2957795
            roll = self._roll * RtA
            pitch = self._pitch * RtA
            yaw = self._yaw * RtA
        else:
            roll, pitch, yaw = self._roll, self._pitch, self._yaw
        euler = [roll, pitch, yaw]
        # self._roll, self._pitch, self._yaw = 0.0, 0.0, 0.0
        return euler

    # 获取IMU的四元数，返回quat=[w, x, y, z]
    def get_imu_quaternion_data(self):
        quat = [self._q0, self._q1, self._q2, self._q3]
        # self._q0, self._q1, self._q2, self._q3 = 0.0, 0.0, 0.0, 0.0
        return quat

    # 获取气压计的数据，返回baro=[height, temperature, pressure, pressure_contrast]
    def get_baro_data(self):
        baro = [self._height, self._temperature, self._pressure, self._pressure_contrast]
        # self._height, self._temperature, self._pressure, self._pressure_contrast = 0.0, 0.0, 0.0, 0.0
        return baro

    # 获取底层单片机版本号，如1.1
    # Get the underlying microcontroller version number, such as 1.1
    def get_version(self):
        self._version_H = -1
        self._version_M = -1
        self._version_L = -1
        self._request_data(self.FUNC_VERSION)
        for i in range(0, 20):
            if self._version_H != -1:
                time.sleep(.001)
                version = "V%d.%d.%d" % (self._version_H, self._version_M, self._version_L)
                if self._debug:
                    print("get_version:{0}".format(version))
                return version
            time.sleep(.01)
        return None

    # 重置用户数据。请谨慎操作，会将所有校准值清零。
    def reset_user_data(self):
        cmd = [self._HEAD1, self._HEAD2, 0x00, self.FUNC_RESET_FLASH, 0x01, 0x5F]
        cmd[2] = len(cmd) + 1
        checksum = sum(cmd) & 0xFF
        cmd.append(checksum)
        self._send_data(cmd)
        self._print_log(cmd, "reset user data")
        time.sleep(1)



if __name__ == '__main__':
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

    # time.sleep(1)
    # bot.calibration_imu()

    # bot.calibration_mag()

    # bot.calibration_temperature(27.1)

    # bot.set_report_rate(25)

    # bot.set_algo_type(9)

    # bot.reset_user_data()

    time.sleep(1)

    try:
        while True:
            accel = bot.get_accelerometer_data()
            gyro = bot.get_gyroscope_data()
            mag = bot.get_magnetometer_data()
            euler = bot.get_imu_attitude_data()
            quat = bot.get_imu_quaternion_data()
            baro = bot.get_baro_data()

            print("raw_accel:", accel)
            print("raw_gyro:", gyro)
            print("raw_mag:", mag)
            print("rpy:", euler)
            print("quat:", quat)
            print("baro:", baro)
            print("")

            time.sleep(.1)
    except KeyboardInterrupt:
        pass


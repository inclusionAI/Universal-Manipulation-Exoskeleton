import time

import can
import numpy as np

from ume.damiao_util import (
    create_mode_set_message,
    create_MIT_control_message,
    parse_can_reply,
    to_can_fd_msg,
    MotorType
)

class DamiaoChain:

    def __init__(self, motor_ids=[1], motor_types=[MotorType.DM4310], can_channel="can0", can_fd=False, timeout=2):
        self.motor_ids = motor_ids
        self.motor_types = motor_types
        self.motor_id_to_data_idx = {motor_id: idx for idx, motor_id in enumerate(motor_ids)}
        self.can_channel = can_channel
        self.can_fd = can_fd
        self.timeout = timeout
    
    def __enter__(self):
        self.bus = can.Bus(interface="socketcan", channel=self.can_channel, fd=self.can_fd, receive_own_messages=False)
        for motor_id, motor_type in zip(self.motor_ids, self.motor_types):
            tx_msg = create_mode_set_message(motor_id, mode="clear_error")
            tx_msg = to_can_fd_msg(tx_msg, is_fd=self.can_fd)
            print(tx_msg)
            self.bus.send(tx_msg, timeout=self.timeout)
            rx_msg = self.bus.recv()
            print(rx_msg)
            tx_msg = create_mode_set_message(motor_id, mode="enable")
            tx_msg = to_can_fd_msg(tx_msg, is_fd=self.can_fd)
            print(tx_msg)
            self.bus.send(tx_msg, timeout=self.timeout)
            rx_msg = self.bus.recv()
            print(rx_msg)
            data = parse_can_reply(rx_msg, motor_type=motor_type)
            print(f"damiao motor {motor_id} initial reply", data, data["actual_angle_rad"], np.rad2deg(data["actual_angle_rad"]))
        return self
    
    def set_zero(self):
        for motor_id, motor_type in zip(self.motor_ids, self.motor_types):
            tx_msg = create_mode_set_message(motor_id, mode="set_zero")
            tx_msg = to_can_fd_msg(tx_msg, is_fd=self.can_fd)
            self.bus.send(tx_msg, timeout=self.timeout)
            rx_msg = self.bus.recv()
            data = parse_can_reply(rx_msg, motor_type=motor_type)
            print(f"damiao motor {motor_id} set zero reply", data)
        return data
    
    def mit_control(self, kp: np.ndarray, kd: np.ndarray, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> dict:
        for x in ["kp", "kd", "q", "dq", "tau"]:
            assert eval(x).shape[0] == len(self.motor_ids), f"Dimension mismatch for {x}"
        data_dict_of_list = {}
        for i, (motor_id, motor_type) in enumerate(zip(self.motor_ids, self.motor_types)):
            tx_msg = create_MIT_control_message(motor_id, kp=kp[i], kd=kd[i], q=q[i], dq=dq[i], tau=tau[i], motor_type=motor_type)
            tx_msg = to_can_fd_msg(tx_msg, is_fd=self.can_fd)
            self.bus.send(tx_msg, timeout=self.timeout)
        replied_motor_ids = list()
        for _ in range(len(self.motor_ids)):
            rx_msg = self.bus.recv(timeout=0.1)
            if rx_msg is None:
                print("Timeout while waiting for motor replies")
                print("replied ids", replied_motor_ids)
                raise TimeoutError("Timeout while waiting for motor replies", "replied motor_ids", replied_motor_ids, "motor_ids that haven't replied", set(self.motor_ids) - set(replied_motor_ids))
            motor_id = rx_msg.arbitration_id & 0x0F
            replied_motor_ids.append(motor_id)
            motor_type = self.motor_types[self.motor_id_to_data_idx[motor_id]]
            data = parse_can_reply(rx_msg, motor_type=motor_type)
            for key, value in data.items():
                if key not in data_dict_of_list:
                    data_dict_of_list[key] = np.zeros(len(self.motor_ids))
                data_idx = self.motor_id_to_data_idx[motor_id]
                data_dict_of_list[key][data_idx] = value
                if "timestamp" not in data_dict_of_list:
                    data_dict_of_list["timestamp"] = np.zeros(len(self.motor_ids))
                data_dict_of_list["timestamp"][data_idx] = rx_msg.timestamp
        return data_dict_of_list
    
    def __exit__(self, exc_type, exc_value, traceback):
        zeros = np.zeros(len(self.motor_ids))
        self.mit_control(kp=zeros, kd=zeros, q=zeros, dq=zeros, tau=zeros)
        for motor_id, motor_type in zip(self.motor_ids, self.motor_types):
            tx_msg = create_mode_set_message(motor_id, mode="disable")
            tx_msg = to_can_fd_msg(tx_msg, is_fd=self.can_fd)
            print(tx_msg)
            self.bus.send(tx_msg, timeout=self.timeout)
            rx_msg = self.bus.recv()
            print(rx_msg)
        self.bus.shutdown()

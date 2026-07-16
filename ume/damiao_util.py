import can
import numpy as np
from typing import Dict, List, Literal, Optional, Union
from dataclasses import dataclass
from enum import Enum

def to_can_fd_msg(msg: can.Message, is_fd: bool) -> can.Message:
    if not is_fd:
        return msg
    else:
        return can.Message(
            timestamp=msg.timestamp,
            arbitration_id=msg.arbitration_id,
            is_extended_id=msg.is_extended_id,
            dlc=msg.dlc,
            data=msg.data,
            is_rx=msg.is_rx,
            is_error_frame=msg.is_error_frame,
            is_remote_frame=msg.is_remote_frame,
            is_fd=True,
            bitrate_switch=True
        )

def create_mode_set_message(arbitration_id: int, mode: Literal["enable", "disable", "set_zero", "clear_error"]):
    data = bytearray(8)
    data[0] = 0xFF
    data[1] = 0xFF
    data[2] = 0xFF
    data[3] = 0xFF
    data[4] = 0xFF
    data[5] = 0xFF
    data[6] = 0xFF
    if mode == "enable":
        data[7] = 0xFC
    elif mode == "disable":
        data[7] = 0xFD
    elif mode == "set_zero":
        data[7] = 0xFE
    elif mode == "clear_error":
        data[7] = 0xFB
    else:
        raise RuntimeError()
    
    msg = can.Message(
        arbitration_id=arbitration_id,
        data=data,
        is_extended_id=False,
        dlc=8,
        is_fd=False,
        is_rx=False,
    )
    return msg

def uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    span = x_max - x_min
    offset = x_min
    return (float(x_int + 1) * span / float((1 << bits))) + offset

def LIMIT_MIN_MAX(x, min, max):
    if x <= min:
        return min
    elif x > max:
        return max
    return x

def float_to_uint(x: float, x_min: float, x_max: float, bits):
    x = LIMIT_MIN_MAX(x, x_min, x_max)
    span = x_max - x_min
    data_norm = (x - x_min) / span
    return np.uint16(data_norm * ((1 << bits) - 1))

def int_can_val_to_float_val(value: int, val_min: float, val_max: float, bits: int) -> float:
        return uint_to_float(value, val_min, val_max, bits)

class DamiaoConstants:
    KP_MAX = 500
    KD_MAX = 5
    POSITION_REPLY_BITS = 16
    VELOCITY_REPLY_BITS = 12
    TORQUE_REPLY_BITS = 12
    KP_REPLY_BITS = 12
    KD_REPLY_BITS = 12

@dataclass
class MotorLimits:
    """Motor physical limits for parameter scaling.

    Reference: DM_CAN.py Limit_Param array structure
    """

    q_max: float  # Maximum position in radians
    dq_max: float  # Maximum velocity in radians/second
    tau_max: float  # Maximum torque in Nm

class MotorType(str, Enum):
    """Enumeration of Damiao motor types.

    Reference: DM_CAN.py DM_Motor_Type enum and Limit_Param array lines 65-69
    """

    DM4310 = "DM4310"
    DM4310_48V = "DM4310_48V"
    DM4340 = "DM4340"
    DM4340_48V = "DM4340_48V"
    DM6006 = "DM6006"
    DM8006 = "DM8006"
    DM8009 = "DM8009"
    DM10010L = "DM10010L"
    DM10010 = "DM10010"
    DMH3510 = "DMH3510"
    DMH6215 = "DMH6215"
    DMG6220 = "DMG6220"

MOTOR_LIMITS = {
    MotorType.DM4310: MotorLimits(q_max=12.5, dq_max=30.0, tau_max=10.0),
    MotorType.DM4310_48V: MotorLimits(q_max=12.5, dq_max=50.0, tau_max=10.0),
    MotorType.DM4340: MotorLimits(q_max=12.5, dq_max=8.0, tau_max=28.0),
    MotorType.DM4340_48V: MotorLimits(q_max=12.5, dq_max=10.0, tau_max=28.0),
    MotorType.DM6006: MotorLimits(q_max=12.5, dq_max=45.0, tau_max=20.0),
    MotorType.DM8006: MotorLimits(q_max=12.5, dq_max=45.0, tau_max=40.0),
    MotorType.DM8009: MotorLimits(q_max=12.5, dq_max=45.0, tau_max=54.0),
    MotorType.DM10010L: MotorLimits(q_max=12.5, dq_max=25.0, tau_max=200.0),
    MotorType.DM10010: MotorLimits(q_max=12.5, dq_max=20.0, tau_max=200.0),
    MotorType.DMH3510: MotorLimits(q_max=12.5, dq_max=280.0, tau_max=1.0),
    MotorType.DMH6215: MotorLimits(q_max=12.5, dq_max=45.0, tau_max=10.0),
    MotorType.DMG6220: MotorLimits(q_max=12.5, dq_max=45.0, tau_max=10.0),
}

def parse_can_reply(msg: can.Message, motor_type: MotorType) -> Dict:
    data = msg.data
    pos_int = (data[1] << 8) | data[2]
    spd_int = (data[3] << 4) | (data[4] >> 4)
    tor_int = ((data[4] & 0x0F) << 8) | data[5]
    motor_limits = MOTOR_LIMITS[motor_type]
    motor_data = {
        "actual_angle_rad": uint_to_float(pos_int, -motor_limits.q_max, motor_limits.q_max, DamiaoConstants.POSITION_REPLY_BITS),
        "actual_velocity_radps": uint_to_float(spd_int, -motor_limits.dq_max, motor_limits.dq_max, DamiaoConstants.VELOCITY_REPLY_BITS),
        "actual_torque_nm": uint_to_float(tor_int, -motor_limits.tau_max, motor_limits.tau_max, DamiaoConstants.TORQUE_REPLY_BITS)
    }
    return motor_data


def create_MIT_control_message(arbitration_id, kp: float, kd: float, q: float, dq: float, tau: float, motor_type: MotorType = MotorType.DM4310):
    motor_limits = MOTOR_LIMITS[motor_type]
    kp_uint = float_to_uint(kp, 0, DamiaoConstants.KP_MAX, DamiaoConstants.KP_REPLY_BITS)
    kd_uint = float_to_uint(kd, 0, DamiaoConstants.KD_MAX, DamiaoConstants.KD_REPLY_BITS)
    q_uint = float_to_uint(q, -motor_limits.q_max, motor_limits.q_max, DamiaoConstants.POSITION_REPLY_BITS)
    dq_uint = float_to_uint(dq, -motor_limits.dq_max, motor_limits.dq_max, DamiaoConstants.VELOCITY_REPLY_BITS)
    tau_uint = float_to_uint(tau, -motor_limits.tau_max, motor_limits.tau_max, DamiaoConstants.TORQUE_REPLY_BITS)
    data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
    data_buf[0] = (q_uint >> 8) & 0xff
    data_buf[1] = q_uint & 0xff
    data_buf[2] = dq_uint >> 4
    data_buf[3] = ((dq_uint & 0xf) << 4) | ((kp_uint >> 8) & 0xf)
    data_buf[4] = kp_uint & 0xff
    data_buf[5] = kd_uint >> 4
    data_buf[6] = ((kd_uint & 0xf) << 4) | ((tau_uint >> 8) & 0xf)
    data_buf[7] = tau_uint & 0xff
    can_msg = can.Message(
        arbitration_id=arbitration_id,
        data=data_buf,
        is_extended_id=False,
        dlc=8,
        is_fd=False,
        is_rx=False,
    )
    return can_msg

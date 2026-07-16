import time
import numpy as np
from ume.damiao_chain import DamiaoChain, MotorType

if __name__ == "__main__":
    # sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on
    motor_ids = [1,2,3,4,5,6,7,8] # ,2,3,4,5,6
    q_zeros = np.zeros(len(motor_ids))
    with DamiaoChain(
        motor_ids=motor_ids,
        motor_types=[
            MotorType.DM4340,
            MotorType.DM4340,
            MotorType.DM4340,
            MotorType.DM4340,
            MotorType.DM4310,
            MotorType.DM4310,
            MotorType.DM4310,
            MotorType.DM4310
        ], 
        can_channel="can1", can_fd=True, timeout=2
    ) as damiao_chain:
        # damiao_chain.set_zero()
        while True:
            data = damiao_chain.mit_control(kp=q_zeros, kd=q_zeros, q=q_zeros, dq=q_zeros, tau=q_zeros)
            print(data)
            time.sleep(0.1)

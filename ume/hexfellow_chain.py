import time
import numpy as np
from typing import List

import can
import canopen
import struct


class HexFellowChain:
    
    def __init__(self, network: canopen.Network, node_ids: list[int], mode="velocity"):
        self.mode = mode
        self.network = network
        self.node_ids = node_ids
        self.nodes = [network.add_node(node_id) for node_id in node_ids]
        self.peak_torques = self.read_peak_torques()
        self.mit_factors = self.read_mit_factors()  # Read 0x2003:07
        self.check_motor_compatibility()           # Verify firmware/type
        # Store comprehensive state for each motor
        self.motor_states = {
            node_id: {
                "position": 0.0,
                "timestamp": 0,
                "torque": 0.0,
                "error_code": 0,
                "last_update": 0
            } for node_id in node_ids
        }
        for node_id in node_ids:
            # TPDO1 uses COB-ID 0x180 + NodeID
            network.subscribe(0x180 + node_id, self._on_tpdo)
        self.last_received = {node_id: 0 for node_id in node_ids}
    
    def read_mit_factors(self) -> np.ndarray:
        """Reads Object 0x2003:07 to get the MIT conversion factor"""
        # Upload returns bytes; MIT factor is typically a 32-bit integer (i32)
        factor_bytes = self.sdo_upload(0x2003, 7)
        factors = [struct.unpack('<i', fb)[0] for fb in factor_bytes]
        return np.array(factors)
    
    def check_motor_compatibility(self):
        """Checks firmware versions and motor types for safety"""
        versions = self.read_firmware_versions()
        for i, v in enumerate(versions):
            # Documentation notes V7 is current; warn if unexpected
            if v < 7:
                print(f"Warning: Node {self.node_ids[i]} firmware version {v} is older than V7.")
            elif v > 7:
                print(f"Note: Node {self.node_ids[i]} firmware version {v} is newer than expected V7.")
    
    def nmt_operational(self):
        self.network.nmt.send_command(0x01)
    
    def nmt_pre_operational(self):
        self.network.nmt.send_command(0x80)
    
    def sdo_upload(self, index: int, subindex: int) -> List[bytes]:
        results = []
        for node in self.nodes:
            result = node.sdo.upload(index, subindex)
            results.append(result)
        return results

    def read_firmware_versions(self) -> List[int]:
        version_bytes = self.sdo_upload(0x1018, 3)
        return [int.from_bytes(v, 'little') for v in version_bytes]

    def read_peak_torques(self) -> List[float]:
        torque_bytes = self.sdo_upload(0x6076, 0)
        torque_milli_nm = [struct.unpack('<I', torque_bytes[i])[0] for i in range(len(torque_bytes))]
        return np.array(torque_milli_nm) / 1000.0  # Convert mNm to Nm

    def _on_tpdo(self, can_id, data, recv_timestamp):
        # Bytes 0-3: Position (i32)
        # Bytes 4-7: Timestamp (u32)
        # Bytes 8-9: Torque (i16)
        # Bytes 10-11: Error Code (u16)
        node_id = can_id - 0x180
        raw_pos = int.from_bytes(data[0:4], 'little', signed=True)
        raw_time = int.from_bytes(data[4:8], 'little', signed=False)
        raw_torque = int.from_bytes(data[8:10], 'little', signed=True)
        raw_err = int.from_bytes(data[10:12], 'little', signed=False)
        state = self.motor_states[node_id]
        state["velocity"]    = ((raw_pos / (2**21) - state["position"]) / ((raw_time - state["timestamp"]) / 1e6)) if state["last_update"] > 0 else 0.0
        state["position"]    = raw_pos / (2**21)  # Q21 format
        state["timestamp"]   = raw_time
        state["torque"]      = raw_torque / 1000.0 * self.peak_torques[node_id - 1]
        state["error_code"]  = raw_err
        state["this_update"] = recv_timestamp
        state["last_update"] = self.last_received[node_id]
        self.last_received[node_id] = recv_timestamp
        # print(f"Node {node_id} - Position: {state['position']:.4f} rev, Time Stamp: {state['timestamp']} ms, Torque: {state['torque']:.2%} of peak, Error Code: {state['error_code']}, TPDO Frequency: {frequency:.2f} Hz")
    
    def enable_motors(self):
        for node in self.nodes:
            # CiA402 Enable: 
            # Shutdown -> Switch On -> Enable 
            # 0x06 -> 0x07 -> 0x0F
            node.sdo.download(0x6040, 0, (0x06).to_bytes(2, 'little'))
            time.sleep(0.02)
            node.sdo.download(0x6040, 0, (0x07).to_bytes(2, 'little'))
            time.sleep(0.02)
            node.sdo.download(0x6040, 0, (0x0F).to_bytes(2, 'little'))
        print("Motors enabled.")
    
    def disable_motors(self):
        for node in self.nodes:
            node.sdo.download(0x6040, 0, (0x00).to_bytes(2, 'little'))
        print("Motors disabled.")
    
    def shutdown(self):
        for node in self.nodes:
            node.sdo.download(0x6040, 0, (0x06).to_bytes(2, 'little'))
        print("Motors shutdown.")
    
    def enable_velocity_mode(self):
        for node in self.nodes:
            node.sdo.download(0x6060, 0, b'\x03')
            # velocity gain 0x2002, 1-1000
            # node.sdo.download(0x2002, 1, (1000).to_bytes(2, 'little'))
        print("Velocity mode enabled.")
    
    def enable_mit_mode(self):
        for node in self.nodes:
            node.sdo.download(0x6060, 0, b'\x05')
        print("MIT mode enabled.")
    
    def get_motor_state(self) -> dict:
        res = dict()
        for node_idx, node_id in enumerate(self.node_ids):
            state = self.motor_states[node_id]
            for key in state:
                res.setdefault(key, np.zeros(len(self.node_ids)))
                res[key][node_idx] = state[key]
        res["frequency"] = 1 / (res["this_update"] - res["last_update"])
        return res
    
    def setup_tpdo(self, inhibit_time_us=0):
        # No inhibit time by default to allow max frequency
        for node in self.nodes:
            # 1. Disable TPDO to allow parameter changes
            node.sdo.download(0x1800, 1, (0xC0000180 + node.id).to_bytes(4, 'little'))
            time.sleep(0.1) # Litian HACK: Short delay to ensure TPDO is disabled before reconfiguring
            # 2. Clear existing mapping (Set count to 0)
            node.sdo.download(0x1A00, 0, b'\x00') # Clear existing mappings
            # 3. Define TPDO1 Mapping (4 variables: Position, Timestamp, Torque, Error Code)
            # Sub 1: Position   (0x6064)
            node.sdo.download(0x1A00, 1, (0x60640020).to_bytes(4, 'little'))
            # Sub 2: Timestamp  (0x1013)
            node.sdo.download(0x1A00, 2, (0x10130020).to_bytes(4, 'little'))
            # Sub 3: Torque     (0x6077)
            node.sdo.download(0x1A00, 3, (0x60770010).to_bytes(4, 'little'))
            # Sub 4: Error Code (0x603F)
            node.sdo.download(0x1A00, 4, (0x603F0010).to_bytes(4, 'little'))
            # 4. Set Mapping Count back to 4
            node.sdo.download(0x1A00, 0, b'\x04')
            # 5. Set Transmission Type to 255 (Asynchronous/Event-driven)
            node.sdo.download(0x1800, 2, b'\xFF')
            # 6. Set Inhibit Time in microseconds
            node.sdo.download(0x1800, 3, (inhibit_time_us).to_bytes(2, 'little'))
            # 7. Set Event Timer to 1ms
            node.sdo.download(0x1800, 5, (1).to_bytes(2, 'little'))
            # 8. Re-enable TPDO1
            node.sdo.download(0x1800, 1, (0x40000180 + node.id).to_bytes(4, 'little'))

    def setup_rpdo_velocity_mode(self):
        # Using a group COB-ID (e.g., 0x190) for the entire frame
        group_cob_id = 0x190  # Base COB-ID for RPDO1

        for i, node in enumerate(self.nodes):
            # 1. Disable RPDO1, set transmission type to 255, reset count
            node.sdo.download(0x1400, 1, (0x80000000 | group_cob_id).to_bytes(4, 'little'))
            node.sdo.download(0x1400, 2, b'\xFF')  # Asynchronous
            node.sdo.download(0x1600, 0, b'\x00')  # Clear existing mappings
            # Define our target objects
            MAX_TORQUE   = 0x60720010  # 2 bytes
            TARGET_VEL   = 0x60FF0020  # 4 bytes
            FILL_4_BYTES = 0x30000320  # 4 bytes
            FILL_2_BYTES = 0x30000210  # 2 bytes
            # 2. Build mapping per motor
            mapping = []
            for motor_idx in range(len(self.nodes)):
                if motor_idx == i:
                    mapping.extend([MAX_TORQUE, TARGET_VEL])
                else:
                    # fill 6 bytes for the other motors
                    mapping.extend([FILL_4_BYTES, FILL_2_BYTES])
            # 3. Write the mapping to the node
            for sub, obj in enumerate(mapping, start=1):
                node.sdo.download(0x1600, sub, obj.to_bytes(4, 'little'))
            # 4?. Set final mapping count (7 objects total)
            node.sdo.download(0x1600, 0, len(mapping).to_bytes(1, 'little'))
            # 5. Enable RPDO1 and set mode to Velocity (3)
            node.sdo.download(0x1400, 1, group_cob_id.to_bytes(4, 'little'))
            node.sdo.download(0x6060, 0, b'\x03')
            print(f"Node {node.id} Group Velocity/Torque mapping successful.")
    
    def setup_rpdo_mit_mode(self):
        # Using a group COB-ID for the 64-byte CAN-FD frame
        group_cob_id = 0x190

        for i, node in enumerate(self.nodes):
            try:
                # 1. Disable RPDO1 and clear mapping
                node.sdo.download(0x1400, 1, (0x80000000 | group_cob_id).to_bytes(4, 'little'))
                node.sdo.download(0x1400, 2, b'\xFF')  # Asynchronous
                node.sdo.download(0x1600, 0, b'\x00')
                
                # Define MIT Objects (Object Index + Sub + BitLength)
                MIT_POS = 0x20030120  # 4 bytes (Q21 Rev)
                MIT_VEL = 0x20030220  # 4 bytes (Q21 Rev/s)
                MIT_TRQ = 0x20030320  # 4 bytes (Q21 Nm)
                MIT_KP  = 0x20030410  # 2 bytes (0-10000)
                MIT_KD  = 0x20030510  # 2 bytes (0-10000)
                FILL_4  = 0x30000320  # 4-byte placeholder
                
                # 2. Build the 64-byte frame mapping (16 bytes per motor)
                mapping = []
                for motor_idx in range(len(self.nodes)):
                    if motor_idx == i:
                        mapping.extend([MIT_POS, MIT_VEL, MIT_TRQ, MIT_KP, MIT_KD])
                    else:
                        # Fill 16 bytes for other motors using 4x 4-byte placeholders
                        mapping.extend([FILL_4, FILL_4, FILL_4, FILL_4])

                # 3. Download mapping objects
                for sub, obj in enumerate(mapping, start=1):
                    node.sdo.download(0x1600, sub, obj.to_bytes(4, 'little'))
                
                # 4. Set mapping count (Total objects: 5 from target + 12 from fills = 17)
                node.sdo.download(0x1600, 0, len(mapping).to_bytes(1, 'little'))
                
                # 5. Set mode to MIT (5) and re-enable RPDO
                node.sdo.download(0x6060, 0, b'\x05')
                node.sdo.download(0x1400, 1, group_cob_id.to_bytes(4, 'little'))
                
                print(f"Node {node.id} MIT Mode RPDO mapping successful.")
            except Exception as e:
                print(f"Node {node.id} MIT setup failed: {e}")

    def mit_control(self, 
                    positions: np.ndarray,  # In Revolutions (Rev)
                    velocities: np.ndarray, # In Rev/s
                    torques: np.ndarray,    # In Nm
                    kp_gains: np.ndarray,   # Standard Kp units
                    kd_gains: np.ndarray):  # Standard Kd units
        """
        Sends a 64-byte CAN-FD frame for MIT Mode control of 4 motors.
        Calculations follow the HexFellow V7 firmware rules.
        """
        import struct
        
        # 1. Constants and Offsets
        # Firmware V7 uses [0, 2pi), so we add 2^20 to shift to [-pi, pi)
        POS_OFFSET = 2**20
        TWO_POWER_21 = 2**21
        GROUP_COB_ID = 0x190

        payload = b""
        
        for i in range(len(self.node_ids)):
            # 2. Conversion to Motor Units
            # Position: Rev -> Q21 + Offset
            pos_int = int(positions[i] * TWO_POWER_21) + POS_OFFSET
            
            # Velocity: Rev/s -> Q21
            vel_int = int(velocities[i] * TWO_POWER_21)
            
            # Torque: Nm -> Q21
            trq_int = int(torques[i] * TWO_POWER_21)
            
            # 3. Kp/Kd Gain Conversion
            # Gain factor derived from mit_factors
            mit_gain = self.mit_factors[i] / TWO_POWER_21

            # Convert float gains to per-mille integer format (0-10000)
            # Formula: (Gain * 2*pi) / mit_gain
            kp_int = int((kp_gains[i] * 2 * np.pi) / mit_gain)
            kd_int = int((kd_gains[i] * 2 * np.pi) / mit_gain)

            # 4. Pack 16 bytes per motor: i32, i32, i32, u16, u16
            payload += struct.pack('<iiiHH', 
                                   pos_int, 
                                   vel_int, 
                                   trq_int, 
                                   max(0, min(10000, kp_int)), 
                                   max(0, min(10000, kd_int)))

        # 5. Send as a single CAN-FD Frame
        # Ensure is_fd=True and bitrate_switch=True for the 64-byte payload
        msg = can.Message(
            arbitration_id=GROUP_COB_ID,
            data=payload,
            is_fd=True,
            bitrate_switch=True,
            is_extended_id=False
        )
        self.network.bus.send(msg)
    
    def __enter__(self):
        self.nmt_pre_operational()
        print("Firmware versions", self.read_firmware_versions())
        print("Peak torques (Nm)", self.peak_torques)
        
        self.setup_tpdo()
        mode = self.mode
        if mode == "velocity":
            self.setup_rpdo_velocity_mode()
            self.enable_velocity_mode()
        elif mode == "mit":
            self.setup_rpdo_mit_mode()
            self.enable_mit_mode()
        self.enable_motors()
        self.nmt_operational()
        time.sleep(0.1) # Wait for NMT state change
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.nmt_pre_operational()
        self.disable_motors()  # always disable motors on exit for safety
        time.sleep(0.1) # Wait for NMT state change


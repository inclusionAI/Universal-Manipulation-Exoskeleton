import canopen
import time

# sudo ip link set can0 up type can bitrate 1000000 sample-point 0.8 dbitrate 5000000 sample-point 0.75 sjw 5 dsjw 3 fd on

network = canopen.Network()
network.connect(channel='can1', bustype='socketcan', fd=True)

# add node
node_id = 4
node = network.add_node(node_id)

# read firmware
version_bytes = node.sdo.upload(0x1018, 3)
version = int.from_bytes(version_bytes, 'little')
print(f"Firmware Version: {version}")

# check current node id
current_node_id_bytes = node.sdo.upload(0x2001, 1)
current_node_id = int.from_bytes(current_node_id_bytes, 'little')
print(f"Current Node ID: {current_node_id}")

network.disconnect()

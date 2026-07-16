# sudo ip link set can0 up type can bitrate 1000000 sample-point 0.8 dbitrate 5000000 sample-point 0.75 sjw 5 dsjw 3 fd on

import canopen

network = canopen.Network()
network.connect(channel='can1', bustype='socketcan', fd=True)

node_id = 1
node = network.add_node(node_id)
node.sdo.RESPONSE_TIMEOUT = 0.5
version_bytes = node.sdo.upload(0x1018, 3)
version = int.from_bytes(version_bytes, 'little')

print(f"Firmware Version: {version}")
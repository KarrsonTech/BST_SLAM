# after SLAM works use with steamvr and opentrack.
# research opentrack and find my fork on my github.
# optional BTW. also needs patch for rotation. 
# use steamvr_use_after_testing_only.bat to pipe pose quickly using special cli hack.

import socket
import struct
import time
import numpy as np
import quaternion

OPENTRACK_IP = "127.0.0.1"
OPENTRACK_PORT = 4242

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_tracking_data(x, y, z, yaw, pitch, roll):
    """Pack and send tracking data to OpenTrack over UDP."""
    data = struct.pack('dddddd', x, y, z, yaw, pitch, roll)
    sock.sendto(data, (OPENTRACK_IP, OPENTRACK_PORT))

while True:
    _in = input().split(' ')
    x, y, z = float(_in[0]) * 100.0, float(_in[2]) * -100.0, float(_in[1]) * 100.0
    r = quaternion.as_rotation_vector(np.quaternion(float(_in[6]), float(_in[4]), -float(_in[3]), float(_in[5])))
    pitch, yaw, roll = r[1] / np.pi * 180.0, r[0] / np.pi * 180.0, r[2] / np.pi * 180.0
    
    send_tracking_data(x, y, z, yaw, pitch, roll)
    print(f"Sent: X={x}, Y={y}, Z={z}, Yaw={yaw}, Pitch={pitch}, Roll={roll}")

    time.sleep(0.01)  # 100Hz update rate

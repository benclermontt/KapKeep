
import numpy as np
import socket
import sys
import pickle
import struct
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 10
raw_capture = PiRGBArray(camera, size=(1280, 720))

clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('192.168.0.22',8089))

time.sleep(0.1)

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    image = frame.array
    # Serialize frame
    data = pickle.dumps(image)

    # Send message length first
    message_size = struct.pack("L", len(data)) ### CHANGED

    # Then data
    clientsocket.sendall(message_size + data)

    raw_capture.truncate(0)

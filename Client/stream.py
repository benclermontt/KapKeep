"""
Images are just a bunch of numbers, in a certain order, which says the intensity of each color at each location
If you use a specific ordering, you can just write down the bytes representing that data and send it as a string of bits, over a socket
So for example, if I have the array:
[[ 0, 1 ],
   [ 1, 0]]
And I want to represent that in a byte string
Maybe first I say, ok when I pack this I'll go left to right, then start at the top row and go down
So, my string representation of that array is:
"0110"
If on the other side we knew this was a 2x2 array, and the order it was packed, we can read 4 numbers from the socket which maybe also contains the next frame:
011011001011
Then, re-pack it with that order:
[[ 0, 1 ],
   [ 1, 0]]
In your case the big difference is an extra step. You have 3D arrays R, G, B. But if you tell numpy to pack it in 'C' order, it decides all that crap for you
Then it converts whatever datatype to binary and you have a "byte length" to read on the other side!
But the argument for a delimiter comes when maybe you lose a number. You choose a delimiter which is long enough and never appears in your message, so maybe "abc" delimits a new frame. Every time you take a character off, you check if the next 3 are "abc":
Say I was reading a frame with our order from before and I read ahead into the socket and see:
"011abc1100"
That first part is obviously a fucked up frame since it's 3 long not four! So we can throw it out and move on with our lives unlike before, where without the delimiter you lose track of what the start of the message even is
This is much less of a requirement in fixed length messaging but it's still a nice to have. It comes at computational cost

"""
import numpy as np
# import socket
import sys
# import pickle
import zmq
import struct
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import base64


def send_array(socket, A, flags=0, copy=True, track=False):
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )

    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=False)


def main():
    camera = PiCamera()
    camera.resolution = (1280, 720)
    camera.framerate = 10
    raw_capture = PiRGBArray(camera, size=(1280, 720))

    # clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # clientsocket.connect(('192.168.0.22', 8089))
    context = zmq.Context()
    print("Connecting to Server")
    s = context.socket(zmq.PUB)
    s.connect('tcp://192.168.0.11:8089')

    time.sleep(0.1)

    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        # Serialize frame
        # data = pickle.dumps(image)
        #data = image.tobytes(order='C')

        # Send message length first
        # message_size = struct.pack("L", len(data))

        buffer_encoded = base64.b64encode(image)
        # Then data
        s.send_string(buffer_encoded.decode('ascii'))

        raw_capture.truncate(0)


if __name__ == '__main__':
    main()

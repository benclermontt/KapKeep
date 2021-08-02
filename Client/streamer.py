import argparse

import cv2
import zmq

import sys
sys.path.append('..')
sys.path.append('../camera')

from camera.Camera import Camera
from constants import PORT, SERVER_ADDRESS
from utils import image_to_string


class Streamer:
    """
    @author Ben Clermont

    Class to connect to zmq socket server StreamViewer running on server

    Uses unconventional implementation of publisher-subscriber model.
    The subscriber is the host and publisher is the client
    """
    def __init__(self, server_address=SERVER_ADDRESS, port=PORT):
        """
        Tries to connect to the StreamViewer with supplied server_address and creates a socket for future use.
        :param server_address: Address of the computer on which the StreamViewer is running, default is `localhost`
        :param port: Port which will be used for sending the stream
        """

        print("Connecting to ", server_address, "at", port)
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.PUB)
        self.footage_socket.setsockopt(zmq.SNDHWM, 2)
        self.footage_socket.connect('tcp://' + server_address + ':' + port)
        self.keep_running = True

    def start(self):
        """
        Starts sending the stream to the Viewer.
        Creates a camera, takes a image frame converts the frame to string and sends the string across the network
        :return: None
        """
        print("Streaming Started...")
        camera = Camera()
        camera.start_capture()
        self.keep_running = True

        while self.footage_socket and self.keep_running:
            try:
                frame = camera.current_frame.read()  # grab the current frame
                image_as_string = image_to_string(frame)
                self.footage_socket.send(image_as_string)

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
        print("Streaming Stopped!")
        cv2.destroyAllWindows()

    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False


def main():
    port = PORT
    server_address = SERVER_ADDRESS

    streamer = Streamer(server_address, port)
    streamer.start()


if __name__ == '__main__':
    main()

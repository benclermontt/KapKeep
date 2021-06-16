import pickle
import socket
import struct
import numpy as np
import cv2

HOST = ''
PORT = 8089


"""
   Normalizes the gamma values of the passed frame, making brighter areas darker and darker areas brighter.
   
   It does this by creating an array of gamma values that maps the input pixel values to an output value.
   
   For Ex.
    The table could say that if the input gamma is 75, the output gamma will be 90 (brighter).
"""


def normalize_gamma(image, gamma=1.0):
    inverse_gamma = 1 / gamma

    gamma_table = np.array([((i / 255.0) ** inverse_gamma) * 255
                            for i in np.arange(0, 256)]).astype('uint8')

    return cv2.LUT(image, gamma_table)


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()

    data = b''
    payload_size = struct.calcsize("L")

    while True:

        # Retrieve message size
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)

        adjusted_frame = normalize_gamma(frame, 1.5)

        # Calculate Gradients

        adjusted_frame = np.float32(adjusted_frame) / 255.0

        """
        We might want to use an alternative to a Sobel in the future
        
        However the research paper said they got the best results with a simple 1x3 mask
        In this case the Sobel is creating a 1x3/3x1 mask (depending on x or y derivative) the contains
        
        [-1, 0, 1]
        
        We may also want to explore calculating the gradient invididually for each color channel to see which gives
        the lowest normal value and use that each frame. This depends on how computationally heavy later steps will be.
        """

        gradient_x = cv2.Sobel(adjusted_frame, cv2.CV_32F, 1, 0, ksize=1)
        gradient_y = cv2.Sobel(adjusted_frame, cv2.CV_32F, 0, 1, ksize=1)

        """
        To find magnitude and direction of the gradients
        Convert from cartesian to Polar with OpenCV function
        
        This may be a formula we implement ourselves later to look cool
        """

        magnitude, direction = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)



        # Display
        cv2.imshow('adjusted_frame', magnitude)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

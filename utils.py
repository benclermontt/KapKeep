import io


def is_raspberry_pi(raise_on_errors=False):
    """
    @author Ben Clermont
    Checks if Raspberry PI by checking the processor ID of the running machine

    List is not complete and will probably need to be updated/a better solution could be found
    :return: True if machine is raspberry pi
    """
    try:
        with io.open('/proc/cpuinfo', 'r') as cpuinfo:
            found = False
            for line in cpuinfo:
                if line.startswith('Hardware'):
                    found = True
                    label, value = line.strip().split(':', 1)
                    value = value.strip()
                    if value not in (
                            'BCM2708',
                            'BCM2709',
                            'BCM2711',
                            'BCM2835',
                            'BCM2836'
                    ):
                        if raise_on_errors:
                            raise ValueError(
                                'This system does not appear to be a '
                                'Raspberry Pi.'
                            )
                        else:
                            return False
            if not found:
                if raise_on_errors:
                    raise ValueError(
                        'Unable to determine if this system is a Raspberry Pi.'
                    )
                else:
                    return False
    except IOError:
        if raise_on_errors:
            raise ValueError('Unable to open `/proc/cpuinfo`.')
        else:
            return False

    return True


def image_to_string(image):
    """
    @author Ben Clermont

    Converts the passed image to a string and encodes with base 64 for sending over network

    :param image: Image to convert
    :return: base64 encoded string
    """
    import cv2
    import base64
    encoded, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer)


def string_to_image(string):
    """
    @author Ben Clermont

    Converts base64 string to jpg image after receiving it from network

    :param string: Base64 string received over network
    :return: decoded jpg image
    """
    import numpy as np
    import cv2
    import base64
    img = base64.b64decode(string)
    npimg = np.fromstring(img, dtype=np.uint8)
    return cv2.imdecode(npimg, 1)

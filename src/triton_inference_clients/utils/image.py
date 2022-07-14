import cv2
import numpy as np


def normalize(image: np.ndarray):
    return (image - image.min()) / image.ptp()


def bgr2rgb(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr2gray(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def set_min_max(image: np.ndarray, minval: float = 0.0, maxval: float = 1.0):
    return image * (maxval - minval) + minval


def set_mean_std(image: np.ndarray, means, stds):
    image[:, :, 0] -= means[0]
    image[:, :, 0] -= stds[0]
    image[:, :, 1] -= means[1]
    image[:, :, 1] -= stds[1]
    image[:, :, 2] -= means[2]
    image[:, :, 2] -= stds[2]
    
    return image


def to_channel_first_layout(image):
    return np.transpose(image, (2, 0, 1))


def cv2_imencode(image, quality = 50):
    return cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]


def cv2_imdecode(encodings):
    return cv2.imdecode(encodings, 1)

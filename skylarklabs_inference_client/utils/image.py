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


def resize_box(top_left, bottom_right, image_width_height):
    return (
        (
            int(top_left[0] * image_width_height[0]),
            int(top_left[1] * image_width_height[1])
        ),
        (
            int(bottom_right[0] * image_width_height[0]),
            int(bottom_right[1] * image_width_height[1])
        )
    )


def crop_image(image, top_left, bottom_right):
    return image[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]


def aspect_ratio_resize(image, resize_dim):
    h, w = image.shape[:2]

    if h > w:
        aspect_ratio = w / h
        h, w = resize_dim, int(aspect_ratio * resize_dim)
        image = cv2.resize(image, (w, h))
    else:
        aspect_ratio = h / w
        h, w = int(aspect_ratio * resize_dim), resize_dim
        image = cv2.resize(image, (w, h))

    return image    

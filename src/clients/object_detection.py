import cv2
import numpy as np
from .base_image_client import BaseImageGRPCClient
from .utils import to_channel_first_layout

from tritonclient.utils import triton_to_np_dtype

class ObjectDetectionGRPCClient(BaseImageGRPCClient):
    def __init__(self, model_name: str, url: str = "0.0.0.0:8001", model_version: int = 1, hot_reloads: int = 5, **kwargs) -> None:
        super().__init__(model_name, url, model_version, hot_reloads, **kwargs)

        if isinstance(self.resized_dim, int):
            self.resized_dim = (self.resized_dim, self.resized_dim)
            
    def letterbox(self, im, color = (114, 114, 114), auto = True, scaleFill = False, scaleup = True, stride = 32):
        shape = im.shape[:2]

        r = min(self.resized_dim[0] / shape[0], self.resized_dim[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.resized_dim[1] - new_unpad[0], self.resized_dim[0] - new_unpad[1]  
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  
        elif scaleFill:  
            dw, dh = 0.0, 0.0
            new_unpad = (self.resized_dim[1], self.resized_dim[0])

        dw /= 2  
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation = cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color) 

        return im

    def preprocess(self, image):
        self.original_height_width = image.shape[:2]
        
        if self.bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.letterbox(image, stride = 64, auto = False)

        image = np.ascontiguousarray(image)

        image = image / 255.0

        if image.ndim == 2:
            image = np.expand_dims(image, -1)

        image = self.cast_triton_dtype(image)
        image = to_channel_first_layout(image)

        return image
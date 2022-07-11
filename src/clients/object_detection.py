import cv2
import numpy as np
from .base_image_client import BaseImageGRPCClient
from .utils import to_channel_first_layout

from tritonclient.utils import triton_to_np_dtype

class ObjectDetectionGRPCClient(BaseImageGRPCClient):
    def __init__(self, model_name: str, url: str = "0.0.0.0:8001", model_version: int = 1, hot_reloads: int = 5, **kwargs) -> None:
        super().__init__(model_name, url, model_version, hot_reloads, **kwargs)

        # if isinstance(self.resized_dim, int):
        #     self.resized_dim = (self.resized_dim, self.resized_dim)
            
    # def preprocess(self, image):
    #     self.original_height_width = image.shape[:2]
        
    #     if self.bgr2rgb:
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     image = self.letterbox(image, stride = 64, auto = False)

    #     image = np.ascontiguousarray(image)

    #     image = image / 255.0

    #     if image.ndim == 2:
    #         image = np.expand_dims(image, -1)

    #     image = self.cast_triton_dtype(image)
    #     image = to_channel_first_layout(image)

    #     return image
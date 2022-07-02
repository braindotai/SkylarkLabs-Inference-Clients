from clients.base_image_client import BaseImageGRPCClient
from clients.utils import set_min_max, normalize

import numpy as np

class FaceRecognitionGRPCClient(BaseImageGRPCClient):
    def __init__(self, bgr2rgb, **kwargs):
        super().__init__(bgr2rgb = bgr2rgb, model_name = 'face_recognition', **kwargs)
    
    def preprocess(self, image):
        image = super().preprocess(image)
        image = normalize(image)
        image = set_min_max(image, self.minval, self.maxval)

        return image

if __name__ == '__main__':
    client = FaceRecognitionGRPCClient(minval = -1.0, maxval = 1.0, model_version = 1, bgr2rgb = True)
    preprocessed_inputs = client.preprocess((np.random.random((128, 584, 3)) * 255.0).astype('uint8'))

    with client.monitor_performance():
        for _ in range(100):
            output = client.perform_inference([[preprocessed_inputs]] * 128)
    
    print(output)
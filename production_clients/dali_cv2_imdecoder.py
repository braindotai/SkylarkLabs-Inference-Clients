from clients.base_client import BaseGRPCClient

import cv2
import numpy as np

class DaliResizeGRPCClient(BaseGRPCClient):
    def __init__(self, **kwargs):
        super().__init__(model_name = 'dali_cv2_imdecoder', **kwargs)
    
    def preprocess(self, *inputs):
        return super().preprocess(*inputs)

if __name__ == '__main__':
    client = DaliResizeGRPCClient()
    preprocessed_inputs = client.preprocess(cv2.imencode('.jpg', (np.random.random((512, 256, 3)) * 255.0).astype('uint8'))[1])

    with client.monitor_performance():
        for _ in range(1000):
            output = client.perform_inference([preprocessed_inputs])

    print(output[0].shape, output[0].dtype, output[0].min(), output[0].max())

    # import time

    # for I in range(100):
    #     s = time.time()
    #     cv2.imdecode(cv2.imencode('.jpg', (np.random.random((512, 256, 3)) * 255.0).astype('uint8'))[1], 1)
    #     print(time.time() - s)
    # client.benchmark_performance([preprocessed_inputs] * 128)
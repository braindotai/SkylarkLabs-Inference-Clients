from clients.base_client import BaseGRPCClient

import cv2
import numpy as np

class DaliDecoderGRPCClient(BaseGRPCClient):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(
            model_name = 'dali_decoder',
            **kwargs
        )
    
if __name__ == '__main__':
    client = DaliDecoderGRPCClient()
    preprocessed_input1 = client.preprocess(cv2.imencode('.jpg', cv2.imread('production_clients/samples/inputs/pedestrian_detection1.jpg'))[1])
    preprocessed_input2 = client.preprocess(cv2.imencode('.jpg', cv2.imread('production_clients/samples/inputs/face_detection2.jpg'))[1])

    with client.monitor_performance():
        for _ in range(100):
            output = client.perform_inference([preprocessed_input1, preprocessed_input2])

    print(output.shape, output.dtype, output.min(), output.max())
    cv2.imwrite(f'production_clients/samples/outputs/dali_decoder.jpg', output[0])

    # import time

    # for I in range(100):
    #     s = time.time()
    #     cv2.imdecode(cv2.imencode('.jpg', (np.random.random((512, 256, 3)) * 255.0).astype('uint8'))[1], 1)
    #     print(time.time() - s)
    # client.benchmark_performance([preprocessed_inputs] * 128)
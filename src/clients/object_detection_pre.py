import numpy as np
from clients.base_client import BaseGRPCClient
import cv2

class ObjectDetectionPreGRPCClient(BaseGRPCClient):
    def __init__(
        self,
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,
            resize_dim = 640,
        ),
        **kwargs
    ) -> None:
        super().__init__(triton_params = triton_params, model_name = 'object_detection_pre', **kwargs)

        remainder = (self.triton_params['resize_dim'] % 32)
        
        if remainder:
            self.triton_params['resize_dim'] = self.triton_params['resize_dim'] - remainder + 32
        else:
            self.triton_params['resize_dim'] = self.triton_params['resize_dim']
    
    def generate_request(self, *input_batches):
        self.inputs = []
        joined_encodings = []
        split_indices = []

        for cv2_image in input_batches[0]:
            encodings = cv2.imencode('.jpg', cv2_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1]
            joined_encodings.append(encodings)

            split_indices.append((split_indices[-1] if len(split_indices) else 0) + len(encodings))
        
        self.triton_params['joined_encodings'] = np.expand_dims(np.concatenate(joined_encodings, axis = 0), 0)
        self.triton_params['split_indices'] = np.expand_dims(np.array(split_indices), 0)

if __name__ == '__main__':
    import cv2

    client = ObjectDetectionPreGRPCClient(
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,
            resize_dim = 900,
        ),
    )
    cv2_image1 = cv2.resize(cv2.imread('production_clients/samples/inputs/face_detection1.jpg'), (1280, 720))
    cv2_image2 = cv2.resize(cv2.imread('production_clients/samples/inputs/face_detection2.jpg'), (1280, 720))
    cv2_image3 = cv2.resize(cv2.imread('production_clients/samples/inputs/pedestrian_detection1.jpg'), (1280, 720))

    with client.monitor_performance():
        for _ in range(100):
            for output in client.perform_inference([cv2_image1, cv2_image1, cv2_image1, cv2_image1, cv2_image1, cv2_image1, cv2_image1, cv2_image1, cv2_image1]):
                pass

            print(output.shape)
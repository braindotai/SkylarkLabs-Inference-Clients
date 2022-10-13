import cv2
import numpy as np
from .base_client import BaseGRPCClient


class BaseImageGRPCClient(BaseGRPCClient):
    def generate_triton_request(self, inputs, *input_batches):
        joined_encodings = []
        split_indices = []

        for cv2_image in input_batches[0]:
            encodings = cv2.imencode('.jpg', cv2_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.encoding_quality])[1]
            joined_encodings.append(encodings)

            split_indices.append((split_indices[-1] if len(split_indices) else 0) + len(encodings))
        
        self.inference_params['joined_encodings'] = np.expand_dims(np.concatenate(joined_encodings, axis = 0), 0)
        self.inference_params['split_indices'] = np.expand_dims(np.array(split_indices), 0)

        return len(input_batches[0])

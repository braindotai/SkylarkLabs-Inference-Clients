import cv2
import numpy as np
from .base_client import BaseGRPCClient


class ObjectDetectionGRPCClient(BaseGRPCClient):
    def __init__(
        self,
        model_name,
        encoding_quality = 50,
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 720,
            original_width = 1280,
            
            resize_dim = 640,

            iou_thres = 0.40,
            conf_thres = 0.2,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        ),
        **kwargs
    ):
        super().__init__(model_name = model_name, encoding_quality = encoding_quality, triton_params = triton_params, **kwargs)

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
            encodings = cv2.imencode('.jpg', cv2_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.encoding_quality])[1]
            joined_encodings.append(encodings)

            split_indices.append((split_indices[-1] if len(split_indices) else 0) + len(encodings))
        
        self.triton_params['joined_encodings'] = np.expand_dims(np.concatenate(joined_encodings, axis = 0), 0)
        self.triton_params['split_indices'] = np.expand_dims(np.array(split_indices), 0)

        self.batch_size = len(input_batches[0])
    
    
    def postprocess(self, batch_boxes, batch_labels):
        split_indices = np.where(batch_labels == -1)[0]
        
        return np.split(batch_boxes, split_indices)[:-1], np.split(batch_labels, split_indices)[:-1]

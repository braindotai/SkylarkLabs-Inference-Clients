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

            original_width = 1280,
            original_height = 720,
            
            resize_dim = 1280,

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
        
        self.original_height = triton_params['original_height']
        self.original_width = triton_params['original_width']


    def generate_request(self, inputs, *input_batches):
        joined_encodings = []
        split_indices = []

        for cv2_image in input_batches[0]:
            encodings = cv2.imencode('.jpg', cv2_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.encoding_quality])[1]
            joined_encodings.append(encodings)

            split_indices.append((split_indices[-1] if len(split_indices) else 0) + len(encodings))
        
        self.triton_params['joined_encodings'] = np.expand_dims(np.concatenate(joined_encodings, axis = 0), 0)
        self.triton_params['split_indices'] = np.expand_dims(np.array(split_indices), 0)

        return len(input_batches[0])
    
    
    def postprocess(self, batch_boxes, batch_split_indices):
        batch_boxes = np.split(batch_boxes, batch_split_indices)[:-1]

        # batch_boxes[:, 0, 0] *= self.original_width
        # batch_boxes[:, 0, 1] *= self.original_height
        
        # batch_boxes[:, 1, 0] *= self.original_width
        # batch_boxes[:, 1, 1] *= self.original_height

        return batch_boxes
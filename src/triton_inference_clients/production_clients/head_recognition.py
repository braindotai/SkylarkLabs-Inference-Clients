from ..standard_clients.object_detection import ObjectDetectionGRPCClient
import numpy as np


class HeadRecognitionGRPCClient(ObjectDetectionGRPCClient):
    def __init__(
        self,
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
        super().__init__(model_name = 'head_recognition', encoding_quality = encoding_quality, triton_params = triton_params, **kwargs)


    def postprocess(self, batch_features, batch_boxes, batch_split_indices):
        batch_boxes = np.split(batch_boxes, batch_split_indices)[:-1]
        batch_split_indices = np.split(batch_features, batch_split_indices)[:-1]

        return batch_features, batch_boxes
from ..standard_clients.object_detection import ObjectDetectionGRPCClient


class HeadDetectionGRPCClient(ObjectDetectionGRPCClient):
    def __init__(
        self,
        encoding_quality = 50,
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 720,
            original_width = 1280,
            
            resize_dim = 1280,

            iou_thres = 0.40,
            conf_thres = 0.2,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        ),
        **kwargs
    ):
        super().__init__(model_name = 'head_detection', encoding_quality = encoding_quality, triton_params = triton_params, **kwargs)

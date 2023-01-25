from ..standard_clients.object_detection import ObjectDetectionGRPCClient


class FaceDetectionGRPCClient(ObjectDetectionGRPCClient):
    def __init__(
        self,
        encoding_quality = 90,
        inference_params = dict(
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

            spatial_split = 0,

            class_indices = [0],
        ),
        **kwargs
    ):
        super().__init__(model_name = 'face_detection', encoding_quality = encoding_quality, inference_params = inference_params, **kwargs)

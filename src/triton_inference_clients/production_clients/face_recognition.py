from ..standard_clients.base_image_client import BaseImageGRPCClient


class FaceRecognitionGRPCClient(BaseImageGRPCClient):
    def __init__(
        self,
        encoding_quality = 100,
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,
        ),
        **kwargs
    ):
        super().__init__(model_name = 'face_recognition', encoding_quality = encoding_quality, triton_params = triton_params, **kwargs)

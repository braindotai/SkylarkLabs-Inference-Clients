from ..standard_clients.base_image_client import BaseImageGRPCClient


class TruckRecognitionGRPCClient(BaseImageGRPCClient):
    def __init__(
        self,
        encoding_quality = 100,
        inference_params = dict(
            joined_encodings = None,
            split_indices = None,
        ),
        **kwargs
    ):
        super().__init__(model_name = 'truck_recognition', encoding_quality = encoding_quality, inference_params = inference_params, **kwargs)

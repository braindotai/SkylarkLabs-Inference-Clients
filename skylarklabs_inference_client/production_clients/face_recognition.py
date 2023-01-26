import numpy as np
from ..standard_clients.base_image_client import BaseImageGRPCClient


class FaceRecognitionGRPCClient(BaseImageGRPCClient):
    def __init__(
        self,
        encoding_quality = 100,
        inference_params = dict(
            joined_encodings = None,
            split_indices = None,
        ),
        resize_dims = (160, 160),
        **kwargs
    ):
        super().__init__(
            model_name = 'face_recognition',
            encoding_quality = encoding_quality,
            inference_params = inference_params,
            resize_dims = resize_dims,
            **kwargs
        )

    def monolythic_preprocess(self, *input_batch):
        input_batch = super().monolythic_preprocess(input_batch[0])
        input_batch /= 255.0
        input_batch *= 2.0
        input_batch -= 1.0
        input_batch = np.transpose(input_batch, (0, 3, 1, 2))
        input_batch = input_batch[:, [2, 1, 0], :, :]

        return input_batch

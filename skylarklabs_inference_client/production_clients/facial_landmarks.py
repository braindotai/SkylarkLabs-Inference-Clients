import os
INFERENCE_TYPE = os.getenv('INFERENCE_TYPE', 'TRITON_SERVER')
if INFERENCE_TYPE == 'MONOLYTHIC_SERVER':
    import onnxruntime as ort
    import torch

import numpy as np
from ..standard_clients.base_image_client import BaseImageGRPCClient


class FacialLandmarksGRPCClient(BaseImageGRPCClient):
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
            model_name = 'facial_landmarks',
            encoding_quality = encoding_quality,
            inference_params = inference_params,
            resize_dims = resize_dims,
            **kwargs
        )

        if INFERENCE_TYPE == 'MONOLYTHIC_SERVER':
            self.onnxruntime_session = ort.InferenceSession(
                os.path.join(self.repository_root, f'{self.model_name}_model', self.model_version, 'model.onnx'),
                providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'],
            )
    

    def triton_postprocess(self, model_outptus):
        return model_outptus.reshape(model_outptus.shape[0], -1, 2) + 0.5


    def monolythic_preprocess(self, input_batch):
        input_batch = super().monolythic_preprocess(input_batch)
        input_batch /= 255.0
        input_batch *= 2.0
        input_batch -= 1.0
        input_batch = np.expand_dims(input_batch, axis = 1)
        return input_batch
    

    def monolythic_postprocess(self, model_outputs):
        return np.reshape(model_outputs, (-1, 2)) + 0.5
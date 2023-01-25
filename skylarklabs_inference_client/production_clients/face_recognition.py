import os
INFERENCE_TYPE = os.getenv('INFERENCE_TYPE', 'TRITON_SERVER')
if INFERENCE_TYPE == 'MONOLYTHIC_SERVER':
    import onnxruntime as ort
    import torch

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

        if INFERENCE_TYPE == 'MONOLYTHIC_SERVER':
            self.onnxruntime_session = ort.InferenceSession(
                os.path.join(self.repository_root, f'{self.model_name}_model', self.model_version, 'model.onnx'),
                providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'],
            )

    def monolythic_preprocess(self, input_batch):
        input_batch = super().monolythic_preprocess(input_batch)
        input_batch /= 255.0
        input_batch *= 2.0
        input_batch -= 1.0
        input_batch = np.transpose(input_batch, (0, 3, 1, 2))
        return input_batch


    # def monolythic_inference(self, *input_batches, instance_inference_params = None):
    #     input_batch = self.monolythic_preprocess(input_batches[0], self.inference_params['resize_dims'][0][0])

    #     model_outputs = self.onnxruntime_session.run(
    #         [self.onnxruntime_session.get_outputs()[0].name],
    #         {self.onnxruntime_session.get_inputs()[0].name: input_batch}
    #     )[0]

    #     return model_outputs
import cv2
import numpy as np
from .base_client import BaseGRPCClient


class BaseImageGRPCClient(BaseGRPCClient):
    def __init__(
        self,
        resize_dims = (160, 160),
        *args,
        **kwargs
    ):
        super().__init__(resize_dims = resize_dims, *args, **kwargs)

    def triton_generate_request(self, inputs, *input_batches):
        joined_encodings = []
        split_indices = []
        inference_params = self.inference_params.copy()

        for cv2_image in input_batches[0]:
            encodings = cv2.imencode('.jpg', cv2_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.encoding_quality])[1]
            joined_encodings.append(encodings)

            split_indices.append((split_indices[-1] if len(split_indices) else 0) + len(encodings))
        
        inference_params['joined_encodings'] = np.expand_dims(np.concatenate(joined_encodings, axis = 0), 0) 
        inference_params['split_indices'] = np.expand_dims(np.array(split_indices), 0)

        return len(input_batches[0]), inference_params

    
    def monolythic_preprocess(self, input_batch):
        return np.array([cv2.resize(image, self.resize_dims) for image in input_batch]).astype(np.float32)


    def monolythic_postprocess(self, model_outputs):
        return model_outputs

    def monolythic_inference(self, *input_batches, instance_inference_params = None):
        input_batch = self.monolythic_preprocess(input_batches[0])

        model_outputs = (self.onnxruntime_session.run(
            [self.onnxruntime_session.get_outputs()[0].name],
            {self.onnxruntime_session.get_inputs()[0].name: input_batch}
        )[0])

        outputs = self.monolythic_postprocess(model_outputs)

        return outputs
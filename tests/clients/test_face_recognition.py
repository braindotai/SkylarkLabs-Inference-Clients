import os
# os.environ['INFERENCE_TYPE'] = 'MONOLYTHIC_SERVER'

from triton_inference_clients.production_clients import FaceRecognitionGRPCClient
import numpy as np


def test_face_recognition():
    client = FaceRecognitionGRPCClient(
        repository_root = os.path.join('tests', 'assets', 'models'),
        inference_params = dict(
            joined_encodings = None,
            split_indices = None,
            resize_dims = (160, 160),
        )
    )

    input_batch = np.ones((16, 160, 160, 3)).astype('uint8') * 255
        
    embeddings = client.perform_inference(input_batch)
    
    assert embeddings.shape == (16, 512)

if __name__ == '__main__':
    test_face_recognition()
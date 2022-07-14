from triton_inference_clients.production_clients import FaceRecognitionGRPCClient
import numpy as np


def test_face_recognition():
    client = FaceRecognitionGRPCClient(
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,
        )
    )

    input_batch = np.random.random((16, 160, 160, 3)).astype('uint8')
        
    embeddings = client.perform_inference(input_batch)
    
    assert embeddings.shape == (16, 512)
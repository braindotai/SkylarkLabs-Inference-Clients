from triton_inference_clients.production_clients import BatchSimilarityGRPCClient
import numpy as np

def test_batch_similarity():
    client = BatchSimilarityGRPCClient()

    for _ in range(100):
        a, b, c = client.perform_inference([np.random.random((512))] * 10, [np.random.random((512))] * 15)

    assert a.shape == (10,)
    assert b.shape == (10,)
    assert c.shape == (10,)
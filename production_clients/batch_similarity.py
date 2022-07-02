from clients.base_client import BaseGRPCClient
import numpy as np

class BatchSimilarityGRPCClient(BaseGRPCClient):
    def __init__(self, **kwargs):
        super().__init__(model_name = 'batch_similarity', **kwargs)

if __name__ == '__main__':
    client = BatchSimilarityGRPCClient()
    preprocessed_input_batch1, preprocessed_input_batch2 = client.preprocess(np.random.random((512)), np.random.random((512)))

    with client.monitor_performance():
        for _ in range(1000):
            a, b, c = client.perform_inference([preprocessed_input_batch1] * 128, [preprocessed_input_batch2] * 128)

            print(a.shape, b.shape, c.shape)
                


    # client.benchmark_performance([preprocessed_inputs] * 128)
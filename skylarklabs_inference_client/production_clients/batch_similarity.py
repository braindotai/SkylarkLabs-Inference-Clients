import torch
import numpy as np
from ..standard_clients.base_client import BaseGRPCClient


class BatchSimilarityGRPCClient(BaseGRPCClient):
    def __init__(self, **kwargs):
        super().__init__(model_name = 'batch_similarity', **kwargs)


    def triton_postprocess(self, matched_indices, max_similarities, min_similarities):
        return (
            matched_indices.reshape(-1),
            max_similarities.reshape(-1),
            min_similarities.reshape(-1),
        )
    

    def monolythic_inference(self, reference_embeddings, query_embeddings, instance_inference_params = None):
        matched_indices, matched_similarities, mismatched_similarities = self.onnxruntime_session.run(
            [
                self.onnxruntime_session.get_outputs()[0].name,
                self.onnxruntime_session.get_outputs()[1].name,
                self.onnxruntime_session.get_outputs()[2].name,
            ],
            {
                self.onnxruntime_session.get_inputs()[0].name: reference_embeddings,
                self.onnxruntime_session.get_inputs()[1].name: query_embeddings,
            }
        )

        return matched_indices, matched_similarities, mismatched_similarities

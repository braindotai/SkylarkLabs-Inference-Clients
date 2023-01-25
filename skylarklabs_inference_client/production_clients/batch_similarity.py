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
    

    def _monolythic_batch_similarity_metric(self, dotproduct, embeddings1, embeddings2):
        embeddings1_norm = embeddings1.norm(2, dim = 1)
        embeddings2_norm = embeddings2.norm(2, dim = 1)

        cosine_similarity = (1.0 + (dotproduct / (embeddings1_norm * embeddings2_norm)))/2.0 # 0 - 1
        norm_similarity = ((embeddings1_norm - embeddings2_norm).abs()/embeddings2_norm)
        result = (cosine_similarity - norm_similarity).clip(0.0, 1.0)

        return result


    def monolythic_inference(self, query_embeddings, reference_embeddings, instance_inference_params = None):
        query_embeddings, reference_embeddings = torch.from_numpy(np.array(query_embeddings)), torch.from_numpy(np.array(reference_embeddings))
        dot = torch.mm(query_embeddings, reference_embeddings.t())

        similarity_matrix = dot
        matched_references_dotproducts, matched_indices = similarity_matrix.max(1) # [Q]
        matched_references_embeddings = reference_embeddings[matched_indices]

        mismatched_references_dotproducts, mismatched_indices = similarity_matrix.min(1)
        mismatched_references_embeddings = reference_embeddings[mismatched_indices]

        matched_similarities = self._monolythic_batch_similarity_metric(matched_references_dotproducts, query_embeddings, matched_references_embeddings)
        mismatched_similarities = self._monolythic_batch_similarity_metric(mismatched_references_dotproducts, query_embeddings, mismatched_references_embeddings)


        return matched_indices, matched_similarities, mismatched_similarities

from ..standard_clients.base_client import BaseGRPCClient


class BatchSimilarityGRPCClient(BaseGRPCClient):
    def __init__(self, **kwargs):
        super().__init__(model_name = 'batch_similarity', **kwargs)


    def postprocess(self, matched_indices, max_similarities, min_similarities):
        return (
            matched_indices.reshape(-1),
            max_similarities.reshape(-1),
            min_similarities.reshape(-1),
        )

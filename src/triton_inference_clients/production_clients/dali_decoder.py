from ..standard_clients.base_client import BaseGRPCClient


class DaliDecoderGRPCClient(BaseGRPCClient):
    def __init__(self, **kwargs) -> None:
        super().__init__(model_name = 'dali_decoder', **kwargs)

import numpy as np
import os

INFERENCE_TYPE = os.getenv('INFERENCE_TYPE', 'TRITON_SERVER')

if INFERENCE_TYPE == 'TRITON_SERVER':
    from attrdict import AttrDict
    from google.protobuf.json_format import MessageToDict

    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException, triton_to_np_dtype

from rich.table import Table
from rich.console import Console

import time


class BaseGRPCClient:
    def __init__(
        self,
        model_name: str,
        repository_root = None,
        url: str = "0.0.0.0:8001",
        model_version: int = 1,
        inference_params: dict = None,
        **kwargs
    ) -> None:
        
        self.console = Console()
        self.console.rule(f'⚡ Client Initialization Started ⚡', style = 'white')
        start = time.perf_counter()

        self.model_name = model_name
        self.model_version = str(model_version)
        self.request_id = -1
        
        self.inference_params = inference_params
        if inference_params:
            self.inference_params_dtypes = []
            
            for key, float_value in self.inference_params.items():
                self.inference_params[key] = np.array([[float_value]], dtype = np.float32)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        if INFERENCE_TYPE == 'TRITON_SERVER':
            try:
                self.console.log("Initializing GRPC Client...")
                self.triton_client = grpcclient.InferenceServerClient(url = url)
            except InferenceServerException as e:
                self.console.log(e)
                raise

            try:
                self.console.log("Retrieving Model Metadata...")
                model_metadata = self.triton_client.get_model_metadata(
                    model_name = self.model_name,
                    model_version = self.model_version
                )
            except InferenceServerException as e:
                self.console.log(e)
                raise

            try:
                self.console.log("Retrieving Model Config...")
                model_config = self.triton_client.get_model_config(
                    model_name = self.model_name,
                    model_version = self.model_version
                )
            except InferenceServerException as e:
                self.console.log(e)
                raise

            table = self.setup_metadata(model_metadata, model_config)

            self.console.print(table)
            self.console.rule(f'✅ Client Initialization Completed in {time.perf_counter() - start:.4f} ✅', style = 'white')
            print()
        else:
            assert repository_root, f'\n\nArgument `repository_root` must be provided for Monolythic inference type.\n'
            self.repository_root = repository_root


    def setup_metadata(self, model_metadata, model_config):
        model_metadata, model_config = AttrDict(MessageToDict(model_metadata)), AttrDict(MessageToDict(model_config)['config'])

        self.max_batch_size, self.input_config, self.output_config = self.parse_model(model_metadata, model_config)
        
        table = Table(title = '\nParsed Model Properties', min_width = 50, title_justify = 'left', title_style = 'purple3')
        table.add_column('Property', style = 'cyan', justify = 'right')
        table.add_column('Value', style = 'spring_green3')
        
        table.add_row('Model Name', f'{self.model_name}')
        table.add_row('Model Version', f'{self.model_version}')
        table.add_row('Max Batch Size', f'{self.max_batch_size}')
        table.add_row()

        for config in self.input_config:
            table.add_row('Input', f'{config["name"]}', style = 'grey89')
            table.add_row('      Dtype', f'{config["dataType"].replace("TYPE_", "")}')
            table.add_row('      Dims', f'{config["dims"]}')
            if 'reshape' in config:
                table.add_row('      Reshape', f'{config["reshape"]}')
            

            if self.inference_params and config['name'] in self.inference_params:
                self.inference_params_dtypes.append(config["dataType"].replace("TYPE_", ""))

        table.add_row()

        for config in self.output_config:
            table.add_row('Output', f'{config["name"]}', style = 'grey89')
            table.add_row('       Dtype', f'{config["dataType"].replace("TYPE_", "")}')
            table.add_row('       Dims', f'{config["dims"]}')
            if 'reshape' in config:
                table.add_row('       Reshape', f'{config["reshape"]}')

        return table


    def parse_model(self, model_metadata, model_config):
        return (
            model_config.maxBatchSize if hasattr(model_config, 'maxBatchSize') else 0,
            model_config.input,
            model_config.output,
        )


    def triton_preprocess(self, input_batch, input_batch_idx):
        return input_batch


    def generate_request(self, inputs, *input_batches):
        for input_batch_idx, (config, input_batch) in enumerate(zip(self.input_config, input_batches)):
            if not isinstance(input_batch, np.ndarray):
                input_batch = np.array(input_batch)
            
            self.triton_preprocess(input_batch, input_batch_idx)

            dtype = config["dataType"].replace("TYPE_", "")
            infer_inputs = grpcclient.InferInput(config['name'], input_batch.shape, dtype)
            infer_inputs.set_data_from_numpy(input_batch.astype(triton_to_np_dtype(dtype)))
            inputs.append(infer_inputs)

        return input_batch.shape[0]

    def add_inference_params(self, inputs, batch_size, inference_params, instance_inference_params):
        if self.inference_params:
            for (param_name, param_np_array_or_value), dtype in zip(inference_params.items(), self.inference_params_dtypes):
                if instance_inference_params is not None and param_name in instance_inference_params:
                    param_batch = np.array([[instance_inference_params[param_name]] * batch_size], dtype = np.float32)
                else:
                    param_batch = np.repeat(param_np_array_or_value, repeats = batch_size, axis = 0)
                
                infer_inputs = grpcclient.InferInput(param_name, param_batch.shape, dtype)
                infer_inputs.set_data_from_numpy(param_batch.astype(triton_to_np_dtype(dtype)))
                inputs.append(infer_inputs)
    

    def triton_inference(self, *input_batches, instance_inference_params = None):
        inputs = []
        batch_size, inference_params = self.generate_request(inputs, *input_batches)
        self.add_inference_params(inputs, batch_size, inference_params, instance_inference_params)
        self.request_id += 1

        response = self.triton_client.infer(
            self.model_name,
            inputs,
            request_id = str(self.request_id),
            model_version = self.model_version
        )

        outputs = []
        
        for config in self.output_config:
            outputs.append(response.as_numpy(config['name']))
        
        if len(outputs) == 1:
            return self.triton_postprocess(outputs[0])

        return self.triton_postprocess(*outputs)
    

    def monolythic_inference(self, *input_batches, instance_inference_params = None):
        raise NotImplementedError(f'\n\nMonolythic inference is not implemented for {self.model_name}.\n')
    

    def perform_inference(self, *input_batches, instance_inference_params = None):
        if INFERENCE_TYPE == 'TRITON_SERVER':
            result = self.triton_inference(*input_batches, instance_inference_params)
            
        elif INFERENCE_TYPE == 'MONOLYTHIC_SERVER':
            result = self.monolythic_inference(*input_batches, instance_inference_params)

        return result
    

    def triton_postprocess(self, *outputs):
        if len(outputs) == 1:
            return outputs[0]

        return outputs

from typing import List
import numpy as np
from attrdict import AttrDict
from google.protobuf.json_format import MessageToDict

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

import time

from rich.live import Live
from rich.table import Table
from rich.console import Console

class PerformanceMonitor:
    def __init__(self, client) -> None:
        self.client = client
        
    def __enter__(self):
        self.client.console.rule(f'⚡ Starting performence benchmark ⚡', style = 'white')
        self.client.time_profile = True

        self.start = time.perf_counter()
        benchmark_table = self.client._get_benchmark_table()
        self.client._live = Live(benchmark_table, refresh_per_second = 20)
        self.client._live.start(True)
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.client._live.stop()
        self.client.time_profile = False
        
        self.client.console.rule(f'✅ Benchmark Completed in: {time.perf_counter() - self.start:.4f} ✅', style = 'white')
        print()


class BaseGRPCClient:
    def __init__(
        self,
        model_name: str,
        url: str = "0.0.0.0:8001",
        model_version: int = 1,
        hot_reloads: int = 5,
        **kwargs
    ) -> None:
        
        self.console = Console()
        self.console.rule(f'⚡ Client Initialization Started ⚡', style = 'white')
        start = time.perf_counter()        

        self.model_name = model_name
        self.model_version = str(model_version)
        self.inputs = None
        self.request_id = -1
        self.hot_reloads = hot_reloads
        self.benchmark_performance_stats = None
        self.cumulative_runtime = 0.0
        self.runtimes = []
        self.time_profile = False
        
        for key, value in kwargs.items():
            setattr(self, key, value)

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
            model_config.maxBatchSize,
            model_config.input,
            model_config.output,
        )

    def preprocess(self, *inputs):
        preprocessed = []
        for x, config in zip(inputs, self.input_config):
            x = x.astype(triton_to_np_dtype(config["dataType"].replace("TYPE_", "")))
            preprocessed.append(x)

        return preprocessed[0] if len(preprocessed) == 1 else preprocessed
    
    def generate_request(self, *input_batches):
        self.inputs = []

        for config, input_batch in zip(self.input_config, input_batches):
            if not isinstance(input_batch, np.ndarray):
                input_batch = np.array(input_batch)
            
            infer_inputs = grpcclient.InferInput(config['name'], input_batch.shape, config['dataType'].replace("TYPE_", ""))
            infer_inputs.set_data_from_numpy(input_batch)
            self.inputs.append(infer_inputs)
    
    def _get_benchmark_table(self):
        benchmark_table = Table(title = f'{self.model_name.replace("_", " ").title()} Benchmarks Statistics', min_width = 50, title_justify = 'left', title_style = 'purple3')
        benchmark_table.add_column("Stats Metric", justify = 'right', style = 'cyan')
        benchmark_table.add_column("Seconds", style = 'spring_green3')

        return benchmark_table
    
    def perform_inference(self, *input_batches):
        if self.time_profile:
            start = time.perf_counter()

        self.generate_request(*input_batches)
        self.request_id += 1

        response = self.triton_client.infer(
            self.model_name,
            self.inputs,
            request_id = str(self.request_id),
            model_version = self.model_version
        )

        if self.time_profile and self.request_id >= self.hot_reloads:
            took = time.perf_counter() - start
            self.runtimes.append(took)

            a = np.array(self.runtimes)
            self.runtimes = self.runtimes[-1000:]
            self.cumulative_runtime = (0.9 * self.cumulative_runtime) + (0.1 * took)

            benchmark_table = self._get_benchmark_table()
            
            self.benchmark_performance_stats = ('Mean', f'{a.mean():.8f}'), ('Std', f'{a.std():.8f}'), ('95%', f'{np.percentile(a, 95):.8f}'), ('Min', f'{a.min():.8f}'), ('Max', f'{a.max():.8f}'), ('Median', f'{np.median(a):.8f}'), ('Weighted', f'{self.cumulative_runtime:.8f}'),

            for (metric_name, value) in self.benchmark_performance_stats:
                benchmark_table.add_row(metric_name, value)
            
            self._live.update(benchmark_table)
        
        outputs = []
        
        for config in self.output_config:
            outputs.append(response.as_numpy(config['name']))

        if len(outputs) == 1:
            return self.postprocess(outputs[0])
        
        return self.postprocess(*outputs)
    
    def postprocess(self, outputs):
        return outputs
    
    def monitor_performance(self):
        return PerformanceMonitor(self)
import numpy as np
from attrdict import AttrDict
from google.protobuf.json_format import MessageToDict

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

import time

from rich.live import Live
from rich.table import Table
from rich.console import Console, Group


rich_render_group = Group


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
        triton_params: dict = None,
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
        self.batch_size = 1
        
        self.triton_params = triton_params
        if triton_params:
            self.triton_params_dtypes = []
            
            for key, float_value in self.triton_params.items():
                self.triton_params[key] = np.array([[float_value]], dtype = np.float32)
        
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
            

            if self.triton_params and config['name'] in self.triton_params:
                self.triton_params_dtypes.append(config["dataType"].replace("TYPE_", ""))

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


    def preprocess(self, input_batch, input_batch_idx):
        return input_batch


    def generate_request(self, *input_batches):
        self.inputs = []

        for input_batch_idx, (config, input_batch) in enumerate(zip(self.input_config, input_batches)):
            if not isinstance(input_batch, np.ndarray):
                input_batch = np.array(input_batch)
            
            self.preprocess(input_batch, input_batch_idx)

            dtype = config["dataType"].replace("TYPE_", "")
            infer_inputs = grpcclient.InferInput(config['name'], input_batch.shape, dtype)
            infer_inputs.set_data_from_numpy(input_batch.astype(triton_to_np_dtype(dtype)))
            self.inputs.append(infer_inputs)

        self.batch_size = input_batch.shape[0]
    

    def add_triton_params(self, instance_triton_params):
        if self.triton_params:
            for (param_name, param_np_array_or_value), dtype in zip(self.triton_params.items(), self.triton_params_dtypes):
                if instance_triton_params is not None and param_name in instance_triton_params:
                    param_batch = np.array([[instance_triton_params[param_name]] * self.batch_size], dtype = np.float32)
                else:
                    param_batch = np.repeat(param_np_array_or_value, repeats = self.batch_size, axis = 0)
                
                infer_inputs = grpcclient.InferInput(param_name, param_batch.shape, dtype)
                infer_inputs.set_data_from_numpy(param_batch.astype(triton_to_np_dtype(dtype)))
                self.inputs.append(infer_inputs)
    

    def _get_benchmark_table(self):
        benchmark_table = Table(
            title = f'{self.model_name.replace("_", " ").title()} Benchmarks Statistics',
            min_width = 50,
            title_justify = 'left',
            title_style = 'purple3'
        )
        benchmark_table.add_column("Stats Metric", justify = 'right', style = 'cyan')
        benchmark_table.add_column("Seconds", style = 'spring_green3')

        return benchmark_table
    

    def perform_inference(self, *input_batches, instance_triton_params = None):
        if self.time_profile:
            start = time.perf_counter()

        self.generate_request(*input_batches)
        self.add_triton_params(instance_triton_params)
        self.request_id += 1

        response = self.triton_client.infer(
            self.model_name,
            self.inputs,
            request_id = str(self.request_id),
            model_version = self.model_version
        )

        # del self.inputs

        outputs = []
        
        for config in self.output_config:
            outputs.append(response.as_numpy(config['name']))
        
        if len(outputs) == 1:
            result = self.postprocess(outputs[0])
        else:
            result = self.postprocess(*outputs)

        if self.time_profile and self.request_id >= self.hot_reloads:
            took = time.perf_counter() - start
            self.runtimes.append(took)

            a = np.array(self.runtimes)
            self.runtimes = self.runtimes[-1000:]
            self.cumulative_runtime = (0.9 * self.cumulative_runtime) + (0.1 * took)

            benchmark_table = self._get_benchmark_table()
            
            self.benchmark_performance_stats = [
                ('Mean', f'{a.mean():.8f}'),
                ('Std', f'{a.std():.8f}'),
                ('95%', f'{np.percentile(a, 95):.8f}'),
                ('Min', f'{a.min():.8f}'),
                ('Max', f'{a.max():.8f}'),
                ('Median', f'{np.median(a):.8f}'),
                ('Weighted', f'{self.cumulative_runtime:.8f}'),
                ('Fps', f'{self.batch_size / took :.8f}'),
            ]

            for (metric_name, value) in self.benchmark_performance_stats:
                benchmark_table.add_row(metric_name, value)
            
            self._live.update(benchmark_table)
        
        return result
    

    def postprocess(self, *outputs):
        if len(outputs) == 1:
            return outputs[0]

        return outputs
    

    def monitor_performance(self):
        return PerformanceMonitor(self)

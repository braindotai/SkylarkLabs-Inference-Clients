import time
import types
import numpy as np
from rich.table import Table
from rich.console import Console
from rich.live import Live


class PerformanceMonitor:
    def __init__(self, pipeline, hot_reloads = 5) -> None:
        assert hasattr(pipeline, 'perform_inference')
        
        setattr(pipeline, 'perform_inference',  self._monitor(pipeline.perform_inference))

        self.hot_reloads = hot_reloads
        self.title = pipeline.__class__.__name__.title()
        
        self.console = Console()


    def start_monitoring(self):
        self.console.rule(f'⚡ Starting performence benchmark ⚡', style = 'white')
        self.time_profile = True

        self.start = time.perf_counter()
        self.runtimes = []
        self.cumulative_runtime = 0.0

        benchmark_table = self._get_benchmark_table()
        self._live = Live(benchmark_table, refresh_per_second = 20)
        self._live.start(True)
    

    def end_monitoring(self):
        self._live.stop()
        
        self.console.rule(f'✅ Benchmark Completed in: {time.perf_counter() - self.start:.4f} ✅', style = 'white')
        print()
    

    def _get_benchmark_table(self):
        benchmark_table = Table(
            title = f'{self.title.title()} Performance Results',
            min_width = 60,
            title_justify = 'left',
            title_style = 'purple3'
        )
        benchmark_table.add_column("Stats Metric", justify = 'right', style = 'cyan')
        benchmark_table.add_column("Seconds", style = 'spring_green3')
        
        return benchmark_table
    

    def _monitor(self, to_monitor):
        def decorated(*args, **kwargs):
            # if self.time_profile and self.request_id > self.hot_reloads:
            #     start = time.perf_counter()

            started_at = time.perf_counter()

            outputs = to_monitor(*args, **kwargs)
            if isinstance(outputs, types.GeneratorType):
                outputs = list(outputs)

            took = time.perf_counter() - started_at
            self.runtimes.append(took)

            a = np.array(self.runtimes)
            self.runtimes = self.runtimes[-1000:]
            self.cumulative_runtime = (0.9 * self.cumulative_runtime) + (0.1 * took)

            benchmark_table = self._get_benchmark_table()
            
            self.benchmark_performance_stats = [
                ('Took', f'{took:.8f}'),
                ('Mean', f'{a.mean():.8f}'),
                ('Std', f'{a.std():.8f}'),
                ('95%', f'{np.percentile(a, 95):.8f}'),
                ('Min', f'{a.min():.8f}'),
                ('Max', f'{a.max():.8f}'),
                ('Median', f'{np.median(a):.8f}'),
                ('Weighted', f'{self.cumulative_runtime:.8f}'),
                ('Fps', f'{1 / took :.8f}'),
            ]

            for (metric_name, value) in self.benchmark_performance_stats:
                benchmark_table.add_row(metric_name, value)
            
            self._live.update(benchmark_table)
        
            return outputs

        return decorated
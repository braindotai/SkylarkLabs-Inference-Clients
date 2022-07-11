import cv2

import numpy as np
from attrdict import AttrDict
from google.protobuf.json_format import MessageToDict

from tritonclient.utils import triton_to_np_dtype

from rich.table import Table
from .base_client import BaseGRPCClient
from .utils import to_channel_first_layout

class BaseImageGRPCClient(BaseGRPCClient):
    def setup_metadata(self, model_metadata, model_config):
        model_metadata, model_config = AttrDict(MessageToDict(model_metadata)), AttrDict(MessageToDict(model_config)['config'])
        self.input_config, self.output_config = model_config.input, model_config.output
        self.max_batch_size, self.input_name, self.output_name, c, h, w, self.input_format, self.input_dtype = self.parse_model(model_metadata, model_config)
        self.channels, self.height, self.width = int(c), int(h), int(w)
        
        table = Table(title = '\nParsed Model Properties', min_width = 40, title_justify = 'left', title_style = 'purple3')
        table.add_column('Property', style = 'cyan', justify = 'right')
        table.add_column('Value', style = 'spring_green3')
        
        table.add_row('Model Name', f'{self.model_name}')
        table.add_row('Model Version', f'{self.model_version}')
        table.add_row('Max Batch Size', f'{self.max_batch_size}')
        table.add_row('Input Name', f'{self.input_name}')
        table.add_row('Output Name', f'{self.output_name}')
        table.add_row('Channel', f'{c}')
        table.add_row('Height', f'{h}')
        table.add_row('Width', f'{w}')
        table.add_row('Input Format', f'{self.input_format}')
        table.add_row('Input Dtype', f'{self.input_dtype}')

        return table

    def parse_model(self, model_metadata, model_config):
        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = model_metadata.outputs[0]
        
        input_batch_dim = model_config.maxBatchSize > 0

        if input_config.format == 'FORMAT_NHWC':
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1 if input_batch_dim else 0]
            h = input_metadata.shape[2 if input_batch_dim else 1]
            w = input_metadata.shape[3 if input_batch_dim else 2]

        return (
            model_config.maxBatchSize,
            input_metadata.name,
            output_metadata.name,
            c,
            h,
            w,
            input_config.format,
            input_metadata.datatype
        )
    
    def cast_triton_shape(self, image):
        if self.height != -1 and self.width != -1:
            return cv2.resize(image, (self.width, self.height))
        return image
    
    def cast_triton_dtype(self, inputs):
        return inputs.astype(triton_to_np_dtype(self.input_dtype))
    
    def cast_triton_channels(self, image):
        c = image.shape[-1]
        assert c in (1, 3), f'Channels must be either 1 or 3, but got {self.channels}'
        
        if c == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.bgr2rgb else image
        elif c == 1:
            return np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0)

    def preprocess(self, input_batch, input_batch_idx):
        if input_batch_idx == 0:
            image = self.cast_triton_shape(image)
            image = self.cast_triton_channels(image)
            image = self.cast_triton_dtype(image)
            image = to_channel_first_layout(image)
        
        return image
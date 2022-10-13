import os
import cv2
import numpy as np
import onnxruntime as ort

from .base_client import BaseGRPCClient

INFERENCE_TYPE = os.getenv('INFERENCE_TYPE', 'TRITON_SERVER')
if INFERENCE_TYPE == 'MONOLYTHIC_SERVER':
    import torch
    import torchvision


class ObjectDetectionGRPCClient(BaseGRPCClient):
    def __init__(
        self,
        model_name,
        encoding_quality = 50,
        inference_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_width = 1280,
            original_height = 720,
            
            resize_dim = 1280,

            iou_thres = 0.40,
            conf_thres = 0.2,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        ),
        **kwargs
    ):
        super().__init__(model_name = model_name, encoding_quality = encoding_quality, inference_params = inference_params, **kwargs)

        if INFERENCE_TYPE == 'MONOLYTHIC_SERVER':
            self.onnxruntime_session = ort.InferenceSession(os.path.join(self.repository_root, f'{self.model_name}_model', self.model_version, 'model.onnx'), providers = ['CPUExecutionProvider'])

        remainder = (self.inference_params['resize_dim'] % 32)
        
        if remainder:
            self.inference_params['resize_dim'] = self.inference_params['resize_dim'] - remainder + 32
        else:
            self.inference_params['resize_dim'] = self.inference_params['resize_dim']
        
        self.original_height = inference_params['original_height']
        self.original_width = inference_params['original_width']


    def generate_request(self, inputs, *input_batches):
        joined_encodings = []
        split_indices = []

        for cv2_image in input_batches[0]:
            encodings = cv2.imencode('.jpg', cv2_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.encoding_quality])[1]
            joined_encodings.append(encodings)

            split_indices.append((split_indices[-1] if len(split_indices) else 0) + len(encodings))
        
        self.inference_params['joined_encodings'] = np.expand_dims(np.concatenate(joined_encodings, axis = 0), 0)
        self.inference_params['split_indices'] = np.expand_dims(np.array(split_indices), 0)

        return len(input_batches[0])
    

    def triton_postprocess(self, batch_boxes, batch_split_indices):
        batch_boxes = np.split(batch_boxes, batch_split_indices)[:-1]

        return batch_boxes


    def monolythic_preprocess(
        self,
        image,
        resize_dim,
    ):  
        shape = image.shape[:2]
        new_shape = (resize_dim, resize_dim)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  

        dw /= 2  
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation = cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (114, 114, 114)).astype('float32') / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(np.transpose(image, (2, 0, 1)), axis = 0)
        # batch = batch[:, :, :, [2, 1, 0]]
        # assert False, batch.shape

        return batch
    

    def _monolythic_non_max_suppression(
        self,
        outputs,
        iou_thres,
        conf_thres,
        max_det,
        agnostic_nms,
        multi_label,
    ):
        outputs = torch.tensor(outputs)
        
        nc = outputs.shape[2] - 5
        xc = outputs[..., 4] > conf_thres

        max_wh = 4096
        max_nms = 30000
        redundant = True

        multi_label &= nc > 1
        merge = False

        output = []

        for xi, x in enumerate(outputs):
            x = x[xc[xi]]

            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]

            box = self._monolythic_xywh2xyxy(x[:, :4])

            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple = False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:
                conf, j = x[:, 5:].max(1, keepdim = True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            n = x.shape[0]
            if not n:
                continue
            elif n > max_nms:
                x = x[x[:, 4].argsort(descending = True)[:max_nms]]

            c = x[:, 5:6] * (0 if agnostic_nms else max_wh) 
            boxes, scores = x[:, :4] + c, x[:, 4] 
            
            i = torchvision.ops.nms(boxes, scores, iou_thres)

            if i.shape[0] > max_det:
                i = i[:max_det]

            if merge and (1 < n < 3E3): 
                iou = self._monolythic_box_iou(boxes[i], boxes) > iou_thres  
                weights = iou * scores[None]  
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim = True) 
                if redundant:
                    i = i[iou.sum(1) > 1]  

            output.append(x[i])

        return output


    def _monolythic_xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2

        return y


    def _monolythic_clip_coords(self, boxes, shape):
        if isinstance(boxes, torch.Tensor): 
            boxes[:, 0].clamp_(0, shape[1])  
            boxes[:, 1].clamp_(0, shape[0])  
            boxes[:, 2].clamp_(0, shape[1])  
            boxes[:, 3].clamp_(0, shape[0])  
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


    def _monolythic_scale_coords(self, img1_shape, coords, img0_shape, ratio_pad = None):
        if ratio_pad is None:  
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  
        coords[:, [1, 3]] -= pad[1]  
        coords[:, :4] /= gain
        self._monolythic_clip_coords(coords, img0_shape)

        return coords


    def _monolythic_box_iou(self, box1, box2):
        area1 = self._monolythic_box_area(box1.T)
        area2 = self._monolythic_box_area(box2.T)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        
        return inter / (area1[:, None] + area2 - inter)


    def _monolythic_box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
    

    def monolythic_postprocess(
        self,
        outputs,
        original_height,
        original_width,
        resize_dim,
        iou_thres,
        conf_thres,
        max_det,
        agnostic_nms,
        multi_label,
    ):
        outputs_from_model = self._monolythic_non_max_suppression(
            outputs,
            iou_thres,
            conf_thres,
            max_det,
            agnostic_nms,
            multi_label,
        )
        batch_boxes = []
        batch_boxes_split_indices = []
        
        if original_height > original_width:
            aspect_ratio = original_width / original_height
            resized_height = resize_dim
            resized_width = int(aspect_ratio * resize_dim)
            
            if resized_width % 32 != 0:
                resized_width = resized_width - (resized_width % 32) + 32
        else:
            aspect_ratio = original_height / original_width 
            resized_width = resize_dim
            resized_height = int(aspect_ratio * resize_dim)
            
            if resized_height % 32 != 0:
                resized_height = resized_height - (resized_height % 32) + 32
                
        for output in outputs_from_model:
            if len(output):
                output[:, :4] = self._monolythic_scale_coords((resized_height, resized_width), output[:, :4], (original_height, original_width, 3)).round()                

                for *xyxy, _, cls_int in reversed(output):
                    c1, c2 = ((xyxy[0] / original_width), (xyxy[1] / original_height)), ((xyxy[2] / original_width), (xyxy[3] / original_height))
                    # if int(cls_int) == 0:
                    batch_boxes.append((c1, c2))
                        # batch_labels.append(0)
            
            batch_boxes_split_indices.append(len(batch_boxes))
        
        batch_boxes = np.split(batch_boxes, batch_boxes_split_indices)[:-1]

        return batch_boxes


    def monolythic_inference(self, *input_batches, instance_inference_params = None):
        input_batch = self.monolythic_preprocess(input_batches[0][0], self.inference_params['resize_dim'][0][0])

        model_outputs = self.onnxruntime_session.run(
            [self.onnxruntime_session.get_outputs()[0].name],
            {self.onnxruntime_session.get_inputs()[0].name: input_batch}
        )[0]

        batch_boxes = self.monolythic_postprocess(
            model_outputs,
            self.inference_params['original_height'][0][0],
            self.inference_params['original_width'][0][0],
            self.inference_params['resize_dim'][0][0],
            self.inference_params['iou_thres'][0][0],
            self.inference_params['conf_thres'][0][0],
            self.inference_params['max_det'][0][0],
            bool(self.inference_params['agnostic_nms'][0][0]),
            bool(self.inference_params['multi_label'][0][0]),
        )

        return batch_boxes
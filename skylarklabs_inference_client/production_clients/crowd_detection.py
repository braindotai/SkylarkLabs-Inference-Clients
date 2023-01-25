from ..standard_clients.object_detection import ObjectDetectionGRPCClient
import cv2
import numpy as np


class CrowdDetectionGRPCClient(ObjectDetectionGRPCClient):
    def __init__(
        self,
        encoding_quality = 90,
        inference_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 720,
            original_width = 1280,
            
            resize_dim = 1280,

            iou_thres = 0.40,
            conf_thres = 0.2,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,

            spatial_split = 0,
        ),
        **kwargs
    ):
        super().__init__(model_name = 'crowd_detection', encoding_quality = encoding_quality, inference_params = inference_params, **kwargs)
    

    def monolythic_preprocess(
        self,
        image,
        resize_dim,
    ):
        shape = image.shape[:2]
        new_shape = (1280, 1280)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
        dw, dh = np.mod(dw, 1280), np.mod(dh, 1280)  

        dw /= 2  
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation = cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (114, 114, 114)).astype('float32') / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(np.transpose(image, (2, 0, 1)), axis = 0)

        return batch


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
        batch_labels = []
        batch_boxes_split_indices = []
        
        if original_height > original_width:
            aspect_ratio = original_width / original_height
            resized_height = resize_dim
            resized_width = int(aspect_ratio * resize_dim)
            
            if resized_width % 1280 != 0:
                resized_width = resized_width - (resized_width % 1280) + 1280
        else:
            aspect_ratio = original_height / original_width 
            resized_width = resize_dim
            resized_height = int(aspect_ratio * resize_dim)
            
            if resized_height % 1280 != 0:
                resized_height = resized_height - (resized_height % 1280) + 1280
                
        for output in outputs_from_model:
            if len(output):
                output[:, :4] = self._monolythic_scale_coords((resized_height, resized_width), output[:, :4], (original_height, original_width, 3)).round()                

                for *xyxy, _, cls_int in reversed(output):
                    c1, c2 = ((xyxy[0] / original_width), (xyxy[1] / original_height)), ((xyxy[2] / original_width), (xyxy[3] / original_height))
                    class_index = int(cls_int)
                    if class_index in self.inference_params['class_indices']:
                        batch_boxes.append((c1, c2))
                        batch_labels.append(class_index)
            
            batch_boxes_split_indices.append(len(batch_boxes))
        
        batch_boxes = np.split(batch_boxes, batch_boxes_split_indices)[:-1]
        batch_labels = np.split(batch_labels, batch_boxes_split_indices)[:-1]

        return batch_boxes, batch_labels

    
    def monolythic_inference(self, *input_batches, instance_inference_params = None):
        inference_params = self.inference_params.copy()

        if instance_inference_params:
            for key, value in instance_inference_params.items():
                inference_params[key] = value

        input_batch = self.monolythic_preprocess(input_batches[0][0], 1280)
        
        model_outputs = self.onnxruntime_session.run(
            [self.onnxruntime_session.get_outputs()[0].name],
            {self.onnxruntime_session.get_inputs()[0].name: input_batch}
        )[0]

        batch_boxes = self.monolythic_postprocess(
            model_outputs,
            inference_params['original_height'][0][0],
            inference_params['original_width'][0][0],
            inference_params['resize_dim'][0][0],
            inference_params['iou_thres'][0][0],
            inference_params['conf_thres'][0][0],
            inference_params['max_det'][0][0],
            bool(inference_params['agnostic_nms'][0][0]),
            bool(inference_params['multi_label'][0][0]),
        )

        return batch_boxes
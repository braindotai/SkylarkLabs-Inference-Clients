import numpy as np
from clients.base_client import BaseGRPCClient

class FaceDetectionGRPCClient(BaseGRPCClient):
    def __init__(
        self,
        encoding_quality = 50,
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 720,
            original_width = 1280,
            
            resize_dim = 640,

            iou_thres = 0.40,
            conf_thres = 0.2,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        ),
        **kwargs
    ):
        super().__init__(encoding_quality = encoding_quality, triton_params = triton_params, model_name = 'face_detection', **kwargs)

        remainder = (self.triton_params['resize_dim'] % 32)
        
        if remainder:
            self.triton_params['resize_dim'] = self.triton_params['resize_dim'] - remainder + 32
        else:
            self.triton_params['resize_dim'] = self.triton_params['resize_dim']

    def generate_request(self, *input_batches):
        self.inputs = []
        joined_encodings = []
        split_indices = []

        for cv2_image in input_batches[0]:
            encodings = cv2.imencode('.jpg', cv2_image, [int(cv2.IMWRITE_JPEG_QUALITY), self.encoding_quality])[1]
            joined_encodings.append(encodings)

            split_indices.append((split_indices[-1] if len(split_indices) else 0) + len(encodings))
        
        self.triton_params['joined_encodings'] = np.expand_dims(np.concatenate(joined_encodings, axis = 0), 0)
        self.triton_params['split_indices'] = np.expand_dims(np.array(split_indices), 0)

        self.batch_size = len(input_batches[0])
    
    def postprocess(self, batch_boxes, batch_labels):
        split_indices = np.where(batch_labels == -1)[0]
        
        return np.split(batch_boxes, split_indices)[:-1], np.split(batch_labels, split_indices)[:-1]

if __name__ == '__main__':
    import os
    import cv2
    from glob import glob

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    client = FaceDetectionGRPCClient()

    input_batch = [cv2.resize(cv2.imread(path), (1280, 720)) for path in glob(os.path.join(BASE_DIR, 'samples', 'inputs', 'face*'))]
    # input_batch = [cv2.resize(cv2.imread(os.path.join(BASE_DIR, 'samples', 'inputs', 'face_detection3.jpg')), (1280, 720))]
    input_batch += input_batch
    print('Batch size:', len(input_batch))

    with client.monitor_performance():
        for _ in range(100):
            batch_boxes, batch_labels = client.perform_inference(input_batch)

    for idx, (boxes, labels, cv2_image) in enumerate(zip(batch_boxes, batch_labels, input_batch)):
        for (top_left, bottom_right) in boxes:
            cv2.rectangle(cv2_image, top_left, bottom_right, [0, 0, 255], 2)

        cv2.imwrite(os.path.join(BASE_DIR, 'samples', 'outputs', f'face_detection({idx}).jpg'), cv2_image)

# from clients.base_client import BaseGRPCClient

# class FaceDetectionGRPCClient(BaseGRPCClient):
#     def __init__(self, **kwargs):
#         super().__init__(model_name = 'face_detection_model', **kwargs)

# if __name__ == '__main__':
#     import os
#     import cv2
#     from glob import glob
#     from clients.object_detection_pre import ObjectDetectionPreGRPCClient
#     from clients.object_detection_post import ObjectDetectionPostGRPCClient

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#     client_pre = ObjectDetectionPreGRPCClient(
#         triton_params = dict(
#             joined_encodings = None,
#             split_indices = None,
#             resize_dim = 640,
#         ),
#     )
    
#     client_model = FaceDetectionGRPCClient()
    
#     client_post = ObjectDetectionPostGRPCClient(
#         triton_params = dict(
#             original_height = 720,
#             original_width = 1280,
#             iou_thres = 0.40,
#             conf_thres = 0.25,
#             max_det = 3000,
#             agnostic_nms = 0,
#             multi_label = 0,
#             resize_dim = 640,
#         ),
#     )

#     input_batch = [cv2.resize(cv2.imread(path), (1280, 720)) for path in glob(os.path.join(BASE_DIR, 'samples', 'inputs', 'face*'))]

#     preprocessed = client_pre.perform_inference(input_batch)
#     model_outputs = client_model.perform_inference(preprocessed)
#     batch_boxes, batch_labels = client_post.perform_inference(model_outputs)
#     print(batch_boxes.shape, batch_labels.shape)

#     for idx, (boxes, labels, cv2_image) in enumerate(zip(batch_boxes, batch_labels, input_batch)):
#         for (top_left, bottom_right) in boxes:
#             cv2.rectangle(cv2_image, top_left, bottom_right, [0, 0, 255], 2)

#         cv2.imwrite(os.path.join(BASE_DIR, 'samples', 'outputs', f'face_detection{idx}.jpg'), cv2_image)
    
#     import time
#     for _ in range(100):
#         s = time.time()    
#         preprocessed = client_pre.perform_inference(input_batch)
#         model_outputs = client_model.perform_inference(preprocessed)
#         postprocessed = client_post.perform_inference(model_outputs)

#         batch_boxes, batch_labels = client_post.perform_inference(model_outputs)
#         print(time.time() - s)  

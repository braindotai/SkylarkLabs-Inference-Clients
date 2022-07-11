import numpy as np
from clients.base_client import BaseGRPCClient

class PedestrianDetectionGRPCClient(BaseGRPCClient):
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
        super().__init__(encoding_quality = encoding_quality, triton_params = triton_params, model_name = 'pedestrian_detection', **kwargs)

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
    from clients.utils import draw_bounding_box

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    cap = cv2.VideoCapture(os.path.join(BASE_DIR, 'samples', 'videos', 'f0.mp4'))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    aspect = h / w
    w = 1280
    h = int(aspect * w)

    client = PedestrianDetectionGRPCClient(
        encoding_quality = 90,
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = h,
            original_width = w,
            
            resize_dim = w,

            iou_thres = 0.40,
            conf_thres = 0.15,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        )
    )

    video_writer = cv2.VideoWriter(
        os.path.join(BASE_DIR, 'samples', 'videos', 'f0_output.mp4'),
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'),
        fps = fps,
        frameSize = [w, h]
    )
    boundary_top_left = (800, 380)
    boundary_bottom_right = (1100, 695)

    def is_intruding(
        boundary_top_left,
        boundary_bottom_right,
        top_left,
        bottom_right,
    ):
        boundary_width = boundary_bottom_right[0] - boundary_top_left[0]
        boundary_height = boundary_bottom_right[1] - boundary_top_left[1]
        
        padding_x = int(0.2 * boundary_width)
        padding_y = int(0.2 * boundary_height)

        return top_left[0] > boundary_top_left[0] - padding_x and top_left[1] > boundary_top_left[1] - padding_x and bottom_right[0] < boundary_bottom_right[0] + padding_y and bottom_right[1] < boundary_bottom_right[1] + padding_y

    frame_idx = 0
    with client.monitor_performance():
        while True:
            has_frame, frame = cap.read()
            if has_frame:
                frame_idx += 1
                frame = cv2.resize(frame, (w, h))

                input_batch = [frame]

                batch_boxes, batch_labels = client.perform_inference(input_batch)

                draw_bounding_box(frame, top_left = boundary_top_left, bottom_right = boundary_bottom_right, label = 'Region to monitor', color = 'purple', transparency = 0.8)

                for boxes, labels, cv2_image in zip(batch_boxes, batch_labels, input_batch):
                    for (top_left, bottom_right), label in zip(boxes, labels):
                        if label == 0:
                            if is_intruding(boundary_top_left, boundary_bottom_right, top_left, bottom_right):
                                draw_bounding_box(frame, top_left = top_left, bottom_right = bottom_right, label = f'Intruder', color = 'red', transparency = 0.7)
                            else:
                                draw_bounding_box(frame, top_left = top_left, bottom_right = bottom_right, label = f'Person', color = 'blue', transparency = 0.7)

                video_writer.write(frame)
                
                if frame_idx == 10:
                    cv2.imwrite(os.path.join(BASE_DIR, 'samples', 'videos', 'frame.jpg'), frame)
                    video_writer.release()
                    break
            else:
                break
    
    video_writer.release()

    # input_batch = [cv2.resize(cv2.imread(path), (1280, 720)) for path in glob(os.path.join(BASE_DIR, 'samples', 'inputs', 'pedestrian*'))]
    # # input_batch += input_batch
    # print('Batch size:', len(input_batch))

    # with client.monitor_performance():
    #     for _ in range(100):
    #         batch_boxes, batch_labels = client.perform_inference(input_batch)

    # for idx, (boxes, labels, cv2_image) in enumerate(zip(batch_boxes, batch_labels, input_batch)):
    #     for (top_left, bottom_right) in boxes:
    #         cv2.rectangle(cv2_image, top_left, bottom_right, [0, 0, 255], 2)

    #     cv2.imwrite(os.path.join(BASE_DIR, 'samples', 'outputs', f'pedestrian_detection({idx}).jpg'), cv2_image)





# from clients.base_client import BaseGRPCClient

# class PedestrianDetectionGRPCClient(BaseGRPCClient):
#     def __init__(self, **kwargs):
#         super().__init__(model_name = 'pedestrian_detection_model', **kwargs)

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
#     client_model = PedestrianDetectionGRPCClient()
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

#     input_batch = [cv2.resize(cv2.imread(path), (1280, 720)) for path in glob(os.path.join(BASE_DIR, 'samples', 'inputs', 'pedestrian*'))]

#     preprocessed = client_pre.perform_inference(input_batch)
#     model_outputs = client_model.perform_inference(preprocessed)
#     batch_boxes, batch_labels = client_post.perform_inference(model_outputs)

#     for idx, (boxes, labels, cv2_image) in enumerate(zip(batch_boxes, batch_labels, input_batch)):
#         for (top_left, bottom_right) in boxes:
#             cv2.rectangle(cv2_image, top_left, bottom_right, [0, 0, 255], 2)

#         cv2.imwrite(os.path.join(BASE_DIR, 'samples', 'outputs', f'pedestrian_detection{idx}.jpg'), cv2_image)
    
#     import time
#     for _ in range(100):
#         s = time.time()    
#         preprocessed = client_pre.perform_inference(input_batch)
#         model_outputs = client_model.perform_inference(preprocessed)
#         postprocessed = client_post.perform_inference(model_outputs)

#         batch_boxes, batch_labels = client_post.perform_inference(model_outputs)
#         print(time.time() - s)  

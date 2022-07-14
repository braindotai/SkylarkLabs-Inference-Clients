from ..standard_clients.object_detection import ObjectDetectionGRPCClient


class PedestrianDetectionGRPCClient(ObjectDetectionGRPCClient):
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
        super().__init__(model_name = 'pedestrian_detection', encoding_quality = encoding_quality, triton_params = triton_params, **kwargs)


# if __name__ == '__main__':
#     import os
#     import cv2
#     from triton_inference_clients.utils import draw_bounding_box

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#     cap = cv2.VideoCapture(os.path.join(BASE_DIR, 'assets', 'videos', 'inputs', 'crowd.mp4'))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     aspect = h / w
#     w = 1280
#     h = int(aspect * w)

#     client = PedestrianDetectionGRPCClient(
#         encoding_quality = 90,
#         triton_params = dict(
#             joined_encodings = None,
#             split_indices = None,

#             original_height = h,
#             original_width = w,
            
#             resize_dim = w,

#             iou_thres = 0.40,
#             conf_thres = 0.15,
#             max_det = 1000,
#             agnostic_nms = 0,
#             multi_label = 0,
#         )
#     )

#     video_writer = cv2.VideoWriter(
#         os.path.join(BASE_DIR, 'assets', 'videos', 'outputs', 'pedestrian_detection_output.mp4'),
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v'),
#         fps = fps,
#         frameSize = [w, h]
#     )
#     boundary_top_left = (800, 380)
#     boundary_bottom_right = (1100, 695)

#     def is_intruding(
#         boundary_top_left,
#         boundary_bottom_right,
#         top_left,
#         bottom_right,
#     ):
#         boundary_width = boundary_bottom_right[0] - boundary_top_left[0]
#         boundary_height = boundary_bottom_right[1] - boundary_top_left[1]
        
#         padding_x = int(0.2 * boundary_width)
#         padding_y = int(0.2 * boundary_height)

#         return all((
#             top_left[0] > boundary_top_left[0] - padding_x,
#             top_left[1] > boundary_top_left[1] - padding_x,
#             bottom_right[0] < boundary_bottom_right[0] + padding_y,
#             bottom_right[1] < boundary_bottom_right[1] + padding_y
#         ))

#     frame_idx = 0
#     with client.monitor_performance():
#         while True:
#             has_frame, frame = cap.read()
#             if has_frame:
#                 frame_idx += 1
#                 frame = cv2.resize(frame, (w, h))

#                 input_batch = [frame]

#                 batch_boxes, batch_labels = client.perform_inference(input_batch)

#                 draw_bounding_box(
#                     frame,
#                     top_left = boundary_top_left,
#                     bottom_right = boundary_bottom_right,
#                     label = 'Region to monitor',
#                     color = 'purple',
#                     transparency = 0.8
#                 )

#                 for boxes, labels, cv2_image in zip(batch_boxes, batch_labels, input_batch):
#                     for (top_left, bottom_right), label in zip(boxes, labels):
#                         if label == 0:
#                             if is_intruding(boundary_top_left, boundary_bottom_right, top_left, bottom_right):
#                                 draw_bounding_box(
#                                     frame,
#                                     top_left = top_left,
#                                     bottom_right = bottom_right,
#                                     label = f'Intruder',
#                                     color = 'red',
#                                     transparency = 0.7
#                                 )
#                             else:
#                                 draw_bounding_box(
#                                     frame,
#                                     top_left = top_left,
#                                     bottom_right = bottom_right,
#                                     label = f'Person',
#                                     color = 'blue',
#                                     transparency = 0.7
#                                 )

#                 video_writer.write(frame)
#             else:
#                 break

#     video_writer.release()

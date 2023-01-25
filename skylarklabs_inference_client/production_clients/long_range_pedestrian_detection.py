from ..standard_clients.object_detection import ObjectDetectionGRPCClient


class LongRangePedestrianDetectionGRPCClient(ObjectDetectionGRPCClient):
    def __init__(
        self,
        encoding_quality = 50,
        inference_params = dict(
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

            spatial_split = 0,
        ),
        **kwargs
    ):
        super().__init__(model_name = 'long_range_pedestrian_detection', encoding_quality = encoding_quality, inference_params = inference_params, **kwargs)


# if __name__ == '__main__':
#     import os
#     import cv2
#     from skylarklabs_inference_client.utils import draw_bounding_box

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#     cap = cv2.VideoCapture(os.path.join(BASE_DIR, 'assets', 'videos', 'inputs', 'long_range_video (5).mp4'))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     aspect = h / w
#     w = 1280
#     h = int(aspect * w)

#     video_writer = cv2.VideoWriter(
#         os.path.join(BASE_DIR, 'assets', 'videos', 'outputs', 'long_range_video (5)_output.mp4'),
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v'),
#         fps = fps,
#         frameSize = [w, h]
#     )

#     client = LongRangePedestrianDetectionGRPCClient(
#         encoding_quality = 90,
#         inference_params = dict(
#             joined_encodings = None,
#             split_indices = None,

#             original_height = h,
#             original_width = w,
            
#             resize_dim = w,

#             iou_thres = 0.40,
#             conf_thres = 0.3,
#             max_det = 1000,
#             agnostic_nms = 0,
#             multi_label = 0,
#         )
#     )

#     # boundary_top_left = (7, 530)
#     # boundary_bottom_right = (1275, 750)
    
#     # boundary_top_left = (7, 490)
#     # boundary_bottom_right = (1275, 750)
    
#     boundary_top_left = (7, 520)
#     boundary_bottom_right = (1275, 790)

#     mark = False

#     def is_intruding(
#         boundary_top_left,
#         boundary_bottom_right,
#         top_left,
#         bottom_right,
#     ):
#         boundary_width = boundary_bottom_right[0] - boundary_top_left[0]
#         boundary_height = boundary_bottom_right[1] - boundary_top_left[1]
        
#         padding_x = int(0.01 * boundary_width)
#         padding_y = int(0.01 * boundary_height)

#         return all((
#             top_left[0] > boundary_top_left[0] - padding_x,
#             top_left[1] > boundary_top_left[1] - padding_x,
#             bottom_right[0] < boundary_bottom_right[0] + padding_y,
#             bottom_right[1] < boundary_bottom_right[1] + padding_y,
#         ))

#     frame_idx = 0
#     centroids = []
#     batch_boxes, batch_labels = [], []
#     # background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

#     with client.monitor_performance():
#         while True:
#             has_frame, frame = cap.read()
#             if has_frame:
#                 frame_idx += 1
#                 # frame = cv2.flip(frame, 1)
#                 frame = cv2.resize(frame, (w, h))
                
#                 # edges = cv2.Canny(frame, 10, 110, )

#                 input_batch = [frame]
                
#                 if frame_idx % 2 == 0:
#                     batch_boxes, batch_labels = client.perform_inference(input_batch)

#                 color = [100, 100, 255] if frame_idx >= 140 else [255, 100, 100]

#                 circled_frame = frame.copy()
#                 for centroid in centroids:
#                     circled_frame = cv2.circle(circled_frame, centroid, radius = 4, color = color, thickness = 3)
#                 frame = (0.7 * circled_frame.astype('float32') + 0.3 * frame.astype('float32')).astype('uint8')
                
#                 draw_bounding_box(frame, top_left = boundary_top_left, bottom_right = boundary_bottom_right, color = 'purple', transparency = 0.7)

#                 centroids = centroids[-200:]

#                 # b, r = np.zeros_like(edges), np.zeros_like(edges)
#                 # b[:int(b.shape[0] // 1.8), :] = edges[:int(b.shape[0] // 1.8), :]
#                 # r[int(r.shape[0] // 1.8):, :] = edges[int(r.shape[0] // 1.8):, :]
#                 # edges_map = np.transpose(np.stack([b, np.zeros_like(edges), r]), (1, 2, 0))
#                 # frame = ((0.35 * edges_map).astype('float32') + 0.65 * frame.astype('float32')).astype('uint8').copy()

#                 for boxes, labels, cv2_image in zip(batch_boxes, batch_labels, input_batch):
#                     for (top_left, bottom_right), label in zip(boxes, labels):
#                         if label == 0:
#                             centroids.append((int((top_left[0] + bottom_right[0]) * 0.5), int((top_left[1] + bottom_right[1]) * 0.5)))
#                             # if is_intruding(boundary_top_left, boundary_bottom_right, top_left, bottom_right):
#                             if frame_idx >= 140:
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

#                 cv2.imshow('output', cv2.resize(frame, (640, int(640 * aspect))))
#                 cv2.waitKey(5)
#             else:
#                 break

#     video_writer.release()

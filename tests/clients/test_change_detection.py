import cv2
from triton_inference_clients.production_clients import ChangeDetectionLocalClient, LongRangePedestrianDetectionGRPCClient
from triton_inference_clients import utils
import os

def test_change_detection():
    long_range_pedestrian_client = LongRangePedestrianDetectionGRPCClient(
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 1536,
            original_width = 1920,
            
            resize_dim = 1920,

            iou_thres = 0.2,
            conf_thres = 0.6,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        )
    )

    change_detection_client = ChangeDetectionLocalClient(
        threshold = 4,
        max_value = 2,
    )

    video_reader = utils.CV2ReadVideo(os.path.join('tests', 'assets', 'videos', 'inputs', 'long_range_video (0).mp4'))
    video_writer = utils.CV2WriteVideo(os.path.join('tests', 'assets', 'videos', 'outputs', 'change_detection.mp4'), 640, 580)

    for frame in video_reader.frames():
        input_batch = [frame]
        original_frame = frame.copy()
        
        if video_reader.frame_idx > 5:
            batch_boxes, batch_labels = long_range_pedestrian_client.perform_inference(input_batch)
            if len(batch_boxes):
                for (top_left, bottom_right), label in zip(batch_boxes[0], batch_labels[0]):
                    utils.draw_bounding_box(frame, top_left, bottom_right, label = 'Intruder', color = 'red')

        # original_frame = cv2.resize(original_frame, (880, 720))
        original_frame = change_detection_client.perform_inference(original_frame, frame)
        
        # video_reader.show(original_frame, resize = (640, 500), window_name = 'Change Detection')
        video_writer.write(original_frame)

        if video_reader.frame_idx > 10:
            break

    video_writer.release()
    # cv2.destroyAllWindows()
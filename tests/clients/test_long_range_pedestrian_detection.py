from triton_inference_clients.production_clients import LongRangePedestrianDetectionGRPCClient
from triton_inference_clients import utils

import os
import cv2


def test_long_range_pedestrian_detection():
    client = LongRangePedestrianDetectionGRPCClient(
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 1536,
            original_width = 1920,
            
            resize_dim = 1280,

            conf_thres = 0.2,
            iou_thres = 0.4,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        )
    )


    video_reader = utils.CV2ReadVideo(os.path.join('tests', 'assets', 'videos', 'inputs', 'long_range_video (0).mp4'))
    video_writer = utils.CV2WriteVideo(os.path.join('tests', 'assets', 'videos', 'outputs', 'long_range_pedestrian_detection.mp4'), 820, 760)

    for frame in video_reader.frames():
        input_batch = [frame]
        batch_boxes, batch_labels = client.perform_inference(input_batch)
        
        if len(batch_boxes):
            for (top_left, bottom_right), label in zip(batch_boxes[0], batch_labels[0]):
                if label == 0:
                    top_left, bottom_right = (int(top_left[0] * 1280), int(top_left[1] * 720)), (int(bottom_right[0] * 1280), int(bottom_right[1] * 720))
                    utils.draw_bounding_box(frame, top_left, bottom_right, label = 'Long Range []', color = 'red')

        # video_reader.show(frame, pause = 5, resize = False, window_name = 'Face Detection')

        video_writer.write(frame)

        if video_reader.frame_idx > 30:
            break
    
    # cv2.destroyAllWindows()

    video_writer.release()
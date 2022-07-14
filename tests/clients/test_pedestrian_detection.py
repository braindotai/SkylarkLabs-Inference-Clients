import cv2
from triton_inference_clients.production_clients import PedestrianDetectionGRPCClient
from triton_inference_clients import utils

import os
from glob import glob


def test_pedestrian_detection():
    client = PedestrianDetectionGRPCClient(
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 720,
            original_width = 1280,
            
            resize_dim = 640,

            conf_thres = 0.2,
            iou_thres = 0.4,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        )
    )


    video_reader = utils.CV2ReadVideo(os.path.join('tests', 'assets', 'videos', 'inputs', 'crowd.mp4'))
    video_writer = utils.CV2WriteVideo(os.path.join('tests', 'assets', 'videos', 'outputs', 'pedestrian_detection.mp4'), 1280, 720)

    for frame in video_reader.frames():
        input_batch = [frame]
        batch_boxes, batch_labels = client.perform_inference(input_batch)
        
        if len(batch_boxes):
            for (top_left, bottom_right), label in zip(batch_boxes[0], batch_labels[0]):
                if label == 0:
                    utils.draw_bounding_box(frame, top_left, bottom_right, label = 'Pedestrian', color = 'red')
                #     cv2.rectangle(frame, top_left, bottom_right, [0, 0, 255], 2)

        # video_reader.show(frame, pause = 5, resize = False, window_name = 'Pedestrian Detection')

        video_writer.write(frame)

        if video_reader.frame_idx > 10:
            break
    
    # cv2.destroyAllWindows()

    video_writer.release()

    input_batch = []
    for path in glob(os.path.join('tests', 'assets', 'images', 'inputs', 'pedestrian*')):
        input_batch.append(cv2.resize(cv2.imread(path), (1280, 720)))
    
    batch_boxes, batch_labels = client.perform_inference(input_batch)

    if len(batch_boxes):
        for idx, (image, boxes, labels) in enumerate(zip(input_batch, batch_boxes, batch_labels)):
            for (top_left, bottom_right), label in zip(boxes, labels):
                if label == 0:
                    utils.draw_bounding_box(image, top_left, bottom_right, label = 'Pedestrian', color = 'red')

            cv2.imwrite(os.path.join('tests', 'assets', 'images', 'outputs', f'pedestrian_detection{idx + 1}.jpg'), image)


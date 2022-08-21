from triton_inference_clients.production_clients import FaceDetectionGRPCClient
from triton_inference_clients import utils

import os
from glob import glob
import cv2


def test_face_detection():
    client = FaceDetectionGRPCClient(
        triton_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 720,
            original_width = 1280,
            
            resize_dim = 1280,

            conf_thres = 0.2,
            iou_thres = 0.4,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        )
    )

    video_reader = utils.CV2ReadVideo(os.path.join('tests', 'assets', 'videos', 'inputs', 'crowd.mp4'))
    # video_writer = utils.CV2WriteVideo(os.path.join('tests', 'assets', 'videos', 'outputs', 'face_detection.mp4'), 1280, 720)

    for frame in video_reader.frames():
        input_batch = [frame] * 10
        batch_boxes = client.perform_inference(input_batch)
        
        if len(batch_boxes):
            for (top_left, bottom_right) in batch_boxes[0]:
                top_left, bottom_right = utils.resize_box(top_left, bottom_right, (video_reader.width, video_reader.height))
                utils.draw_bounding_box(frame, top_left, bottom_right, label = 'Face', color = 'red')

        video_reader.show(frame, pause = 5, resize = False, window_name = 'Face Detection')

        # video_writer.write(frame)

        # if video_reader.frame_idx > 10:
        #     break
    
    # cv2.destroyAllWindows()

    # video_writer.release()

    # input_batch = []
    # for path in glob(os.path.join('tests', 'assets', 'images', 'inputs', 'face*')):
    #     input_batch.append(cv2.resize(cv2.imread(path), (1280, 720)))
    
    # batch_boxes, batch_labels = client.perform_inference(input_batch)

    # if len(batch_boxes):
    #     for idx, (image, boxes, labels) in enumerate(zip(input_batch, batch_boxes, batch_labels)):
    #         for (top_left, bottom_right), label in zip(boxes, labels):
    #             if label == 0:
    #                 top_left, bottom_right = (int(top_left[0] * 1280), int(top_left[1] * 720)), (int(bottom_right[0] * 1280), int(bottom_right[1] * 720))
    #                 utils.draw_bounding_box(image, top_left, bottom_right, label = 'Face', color = 'red')

    #         cv2.imwrite(os.path.join('tests', 'assets', 'images', 'outputs', f'face_detection{idx + 1}.jpg'), image)

if __name__ == '__main__':
    test_face_detection()
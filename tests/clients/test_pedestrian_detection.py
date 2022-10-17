import cv2
import os
os.environ['INFERENCE_TYPE'] = 'MONOLYTHIC_SERVER'

from triton_inference_clients.production_clients import PedestrianDetectionGRPCClient
from triton_inference_clients import utils


def test_pedestrian_detection():
    client = PedestrianDetectionGRPCClient(
        repository_root = os.path.join('tests', 'assets', 'models'),
        inference_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_height = 384,
            original_width = 640,
            
            resize_dim = 640,

            conf_thres = 0.2,
            iou_thres = 0.4,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0, # not needed

            spatial_split = 0,
        )
    )

    video_reader = utils.CV2ReadVideo(os.path.join('tests', 'assets', 'videos', 'inputs', 'crowd.mp4'))
    video_writer = utils.CV2WriteVideo(os.path.join('tests', 'assets', 'videos', 'outputs', 'pedestrian_detection.mp4'), 640, 384)

    for frame in video_reader.frames():
        frame = cv2.resize(frame, (640, 384))
        input_batch = [frame]
        batch_boxes = client.perform_inference(input_batch)
        
        if len(batch_boxes):
            for (top_left, bottom_right) in batch_boxes[0]:
                # top_left, bottom_right = (int(top_left[0] * 1280), int(top_left[1] * 720)), (int(bottom_right[0] * 1280), int(bottom_right[1] * 720))
                top_left, bottom_right = utils.resize_box(top_left, bottom_right, (640, 384))
                utils.draw_bounding_box(frame, top_left, bottom_right, label = 'Pedestrian', color = 'red')
                # cv2.rectangle(frame, top_left, bottom_right, [0, 0, 255], 2)

        video_reader.show(frame, pause = 5, resize = False, window_name = 'Pedestrian Detection')

        video_writer.write(frame)

        # if video_reader.frame_idx > 10:
        #     break
    
    video_writer.release()

    # input_batch = []
    # for path in glob(os.path.join('tests', 'assets', 'images', 'inputs', 'pedestrian*')):
    #     input_batch.append(cv2.resize(cv2.imread(path), (1280, 720)))
    
    # batch_boxes = client.perform_inference(input_batch)

    # if len(batch_boxes):
    #     for idx, (image, boxes) in enumerate(zip(input_batch, batch_boxes)):
    #         for (top_left, bottom_right) in boxes:
    #             # top_left, bottom_right = (int(top_left[0] * 1280), int(top_left[1] * 720)), (int(bottom_right[0] * 1280), int(bottom_right[1] * 720))
    #             top_left, bottom_right = utils.resize_box(top_left, bottom_right, (video_reader.width, video_reader.height))
    #             utils.draw_bounding_box(image, top_left, bottom_right, label = 'Pedestrian', color = 'red')

    #         cv2.imwrite(os.path.join('tests', 'assets', 'images', 'outputs', f'pedestrian_detection{idx + 1}.jpg'), image)


if __name__ == '__main__':
    test_pedestrian_detection()

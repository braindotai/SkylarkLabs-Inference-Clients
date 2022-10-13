import cv2
from triton_inference_clients.production_clients import HeadRecognitionGRPCClient
# from triton_inference_clients.standard_clients import ObjectDetectionGRPCClient
from triton_inference_clients import utils
import os


def test_head_features():
    client = HeadRecognitionGRPCClient(
        inference_params = dict(
            joined_encodings = None,
            split_indices = None,

            original_width = 1280,
            original_height = 720,
            
            resize_dim = 1280,

            iou_thres = 0.40,
            conf_thres = 0.2,
            max_det = 1000,
            agnostic_nms = 0,
            multi_label = 0,
        )
    )

    video_reader = utils.CV2ReadVideo(os.path.join('tests', 'assets', 'videos', 'inputs', 'crowd.mp4'), sampling_fps = 5)
    
    performance_monitor = utils.PerformanceMonitor(client)
    performance_monitor.start_monitoring()

    for frame in video_reader.frames():
        input_batch = [frame]
        batch_features, batch_boxes = client.perform_inference(input_batch)

        frame = cv2.putText(
            frame,
            f'Features shape: {batch_features.shape}',
            (20, 110),
            0,
            0.8,
            [0, 0, 255],
            2
        )

        frame = cv2.putText(
            frame,
            f'Boxes shape: {batch_boxes[0].shape}',
            (20, 150),
            0,
            0.8,
            [0, 0, 255],
            2
        )

        for top_left, bottom_right in batch_boxes[0]:
            top_left, bottom_right = utils.resize_box(top_left, bottom_right, (video_reader.width, video_reader.height))
            utils.draw_bounding_box(frame, top_left, bottom_right, color = 'blue')

        video_reader.show(frame, pause = 5, resize = False, window_name = 'Head Features')
    
    performance_monitor.end_monitoring()

if __name__ == '__main__':
    test_head_features()
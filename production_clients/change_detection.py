import os
import cv2
import numpy as np
from vidstab import VidStab

class ChangeDetectionLocalClient:
    def __init__(self, threshold = 2, max_value = 2, output_video_path = True):
        self.background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.frame_idx = 0
        self.threshold = threshold
        self.max_value = max_value
        self.output_video_path = output_video_path

        self.object_tracker = cv2.TrackerCSRT_create()
        self.stabilizer = VidStab()

    def perform_inference(self, frame):
        self.frame_idx += 1
        stabilized_frame = self.stabilizer.stabilize_frame(input_frame = frame, smoothing_window = 30)
        h, w = stabilized_frame.shape[:2]
        # stabilized_frame = stabilized_frame[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]
        
        if self.frame_idx == 1:
            height, width = stabilized_frame.shape[:2]
            self.accum_image = np.zeros((height, width), np.uint8)

        filters = self.background_subtractor.apply(stabilized_frame)

        ret, th1 = cv2.threshold(filters, self.threshold, self.max_value, cv2.THRESH_BINARY)
        self.accum_image = cv2.add(self.accum_image, th1)
        color_image_video = cv2.applyColorMap(self.accum_image, cv2.COLORMAP_JET)
        output_frame = cv2.addWeighted(stabilized_frame, 0.5, color_image_video, 0.5, 0)

        if self.output_video_path and self.frame_idx == 1:
            self.video_writer = cv2.VideoWriter(
                self.output_video_path,
                fourcc = cv2.VideoWriter_fourcc(*'mp4v'),
                fps = 24,
                frameSize = [output_frame.shape[1], output_frame.shape[0]]
            )

        if self.output_video_path:
            self.video_writer.write(output_frame)

        return output_frame

    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cap = cv2.VideoCapture(os.path.join(BASE_DIR, 'samples', 'videos', 'long_range_video (0).mp4'))
fps = int(cap.get(cv2.CAP_PROP_FPS))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

client = ChangeDetectionLocalClient(
    threshold = 4,
    max_value = 2,
    output_video_path = os.path.join(BASE_DIR, 'samples', 'videos', 'change_detection0.mp4'),
)

while True:
    has_frame, frame = cap.read()

    if has_frame:
        client.perform_inference(frame)
    else:
        client.video_writer.release()
        break


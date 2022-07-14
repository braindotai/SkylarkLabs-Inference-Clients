import cv2
import numpy as np


class ChangeDetectionLocalClient:
    def __init__(self, threshold = 2, max_value = 2):
        self.background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.frame_idx = 0
        self.threshold = threshold
        self.max_value = max_value

        self.object_tracker = cv2.TrackerCSRT_create()

    def perform_inference(self, frame, to_plot_on):
        self.frame_idx += 1
        h, w = frame.shape[:2]
        frame = cv2.blur(frame, (11, 11))
        # cv2.imwrite('del.jpg', frame)
        to_h, to_w = to_plot_on.shape[:2]
        
        if self.frame_idx == 1:
            height, width = frame.shape[:2]
            self.accum_image = np.zeros((height, width), np.uint8)

        filters = self.background_subtractor.apply(frame)

        ret, th1 = cv2.threshold(filters, self.threshold, self.max_value, cv2.THRESH_BINARY)

        self.accum_image = cv2.add(self.accum_image, th1)
        color_image_video = cv2.applyColorMap(self.accum_image, cv2.COLORMAP_JET)
        output_frame = cv2.addWeighted(to_plot_on, 0.5, cv2.resize(color_image_video, (to_plot_on.shape[1], to_plot_on.shape[0])), 0.5, 0)

        return output_frame

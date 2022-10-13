import cv2
import numpy as np



class ChangeDetectionLocalClient:
    def __init__(self, threshold = 2, max_value = 2, min_contour_area = 400):
        self.background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.frame_idx = 0
        self.threshold = threshold
        self.max_value = max_value
        self.min_contour_area = min_contour_area

    def perform_inference(self, frame, to_plot_on):
        self.frame_idx += 1
        frame = cv2.blur(frame, (21, 21))
        
        if self.frame_idx == 1:
            height, width = frame.shape[:2]
            self.accum_image = np.zeros((height, width), np.uint8)

        vectors = self.background_subtractor.apply(frame)

        ret, current_changes = cv2.threshold(vectors, self.threshold, self.max_value, cv2.THRESH_BINARY)

        # self.accum_image = cv2.add(self.accum_image, current_changes)
        color_image_video = cv2.applyColorMap(current_changes, 1)
        contours, hierarchy = cv2.findContours(current_changes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w * h >= self.min_contour_area:
                color_image_video = cv2.rectangle(color_image_video, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # output_frame = cv2.addWeighted(to_plot_on, 0.7, cv2.resize(color_image_video, (to_plot_on.shape[1], to_plot_on.shape[0])), 0.3, 0)

        return cv2.resize(color_image_video, (to_plot_on.shape[1], to_plot_on.shape[0]))



# class ChangeDetectionLocalClient:
#     def __init__(self, threshold = 2, max_value = 2):
#         self.frame_idx = 0
#         self.last_frame = None
#         self.threshold = threshold
#         self.max_value = max_value

#     def perform_inference(self, frame, to_plot_on):
#         self.frame_idx += 1
#         frame = cv2.blur(frame, (27, 27))
        
#         if self.frame_idx == 1:
#             height, width = frame.shape[:2]
#             self.last_frame = frame
#             self.accum_image = np.zeros((height, width, 3), np.uint8)

#         # filters = self.background_subtractor.apply(frame)
#         vector = frame - self.last_frame
#         self.last_frame = frame
#         vector[vector < int(0.99 * vector.max())] = 0
#         # vector[vector > int(0.99 * vector.max())] = 255
        
#         ret, th1 = cv2.threshold(vector, self.threshold, self.max_value, cv2.THRESH_BINARY)
        
#         self.accum_image = cv2.add(self.accum_image, th1)
#         color_image_video = cv2.applyColorMap(self.accum_image, cv2.COLORMAP_JET)
#         output_frame = cv2.addWeighted(to_plot_on, 0.5, cv2.resize(color_image_video, (to_plot_on.shape[1], to_plot_on.shape[0])), 0.5, 0)

#         return output_frame

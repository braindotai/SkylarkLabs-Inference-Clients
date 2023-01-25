import cv2
import numpy as np
from ..utils import resize_box
import torch
from torchvision.ops import box_iou


class GatheringDetectionGRPCClient:
	def __init__(
		self,
		base_client,
		group_threshold = 1,
		eps = 0.9,
		iou_threshold = 0.7,
		long_range = False,
	):
		self.base_client = base_client
		self.group_threshold = group_threshold
		self.eps = eps
		self.iou_threshold = iou_threshold
		self.long_range = long_range

	def perform_inference(self, input_batch):
		batch_boxes, batch_labels = self.base_client.perform_inference(input_batch)
		batch_grouped_boxes = []
		
		for boxes in batch_boxes:
			if self.long_range:
				rectangles = []
			else:
				heatmap = np.zeros((input_batch[0].shape[0], input_batch[0].shape[1], 3), dtype = 'uint8')

			scaled_boxes = []
			for top_left, bottom_right in boxes:
				(top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = resize_box(
					top_left,
					bottom_right,
					(self.base_client.original_width, self.base_client.original_height)
				)
				scaled_boxes.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
				
				if self.long_range:
					rectangles.append([top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y])
				else:
					heatmap[top_left_y: bottom_right_y, top_left_x: bottom_right_x] = 255
				
			if self.long_range:
				group_rectangles, weights = cv2.groupRectangles(
					rectangles,
					groupThreshold = self.group_threshold,
					eps = self.eps,
				)
			else:
				group_rectangles = []
				contours, hierarchy = cv2.findContours(heatmap[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

				for contour in contours:
					x, y, w, h = cv2.boundingRect(contour)
					group_box = torch.tensor([[x, y, (x + w), (y + h)]])
					torch_boxes = torch.from_numpy(np.array(scaled_boxes))
					max_iou = box_iou(group_box, torch_boxes).max()
					
					if max_iou < self.iou_threshold:
						group_rectangles.append((x, y, w, h))

			batch_grouped_boxes.append(group_rectangles)

		return batch_grouped_boxes, batch_boxes, batch_labels

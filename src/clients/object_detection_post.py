import numpy as np

from clients.base_client import BaseGRPCClient

class ObjectDetectionPostGRPCClient(BaseGRPCClient):
    def __init__(
        self,
        triton_params = dict(
            original_height = 720,
            original_width = 1280,
            iou_thres = 0.40,
            conf_thres = 0.25,
            max_det = 3000,
            agnostic_nms = 0,
            multi_label = 0,
            resized_dim = 640,
        ),
        # classes = {
        #     0: 41,
        #     1: 42,
        # },
        classes = {
            26: 1,
            27: 2,
        },
        **kwargs
    ) -> None:
        super().__init__(
            triton_params = triton_params,
            classes = classes,
            model_name = 'object_detection_post',
            **kwargs
        )

    def postprocess(self, batch_boxes, batch_labels):
        split_indices = np.where(batch_labels == -1)[0]
        
        return np.split(batch_boxes, split_indices)[:-1], np.split(batch_labels, split_indices)[:-1]

if __name__ == '__main__':
    import cv2
    from face_detection import FaceDetectionGRPCClient
    from pedestrian_detection import PedestrianDetectionGRPCClient

    face_detection_client = PedestrianDetectionGRPCClient(resized_dim = 640)
    
    cv2_image1 = cv2.resize(cv2.imread('production_clients/samples/inputs/pedestrian_detection1.jpg'), (1280, 720))
    # preprocessed_inputs1 = face_detection_client.preprocess(cv2_image1)
    
    # cv2_image2 = cv2.resize(cv2.imread('production_clients/samples/inputs/face_detection2.jpg'), (1280, 720))
    # preprocessed_inputs2 = face_detection_client.preprocess(cv2_image2)
    
    face_detection_batch = [cv2_image1]
    face_detection_outputs = face_detection_client.perform_inference(face_detection_batch)
    
    client = ObjectDetectionPostGRPCClient(
        triton_params = dict(
            original_height = 720,
            original_width = 1280,
            iou_thres = 0.40,
            conf_thres = 0.25,
            max_det = 3000,
            agnostic_nms = 0,
            multi_label = 0,
            resized_dim = 640,
        ),
        classes = {
            26: 1,
            0: 2,
            24: 3,
        }
    )
    # preprocessed_inputs = client.preprocess(face_detection_outputs)
    print(face_detection_outputs.shape)
    
    with client.monitor_performance():
        for _ in range(100):
            client.perform_inference(face_detection_outputs)

    for idx, ((boxes, labels), cv2_image) in enumerate(zip(client.perform_inference(face_detection_outputs), [cv2_image1])):
        for (top_left, bottom_right), label in zip(boxes, labels):
            cv2.rectangle(cv2_image, top_left, bottom_right, [0, 0, 255], 2)
            cv2.putText(cv2_image, f'[{label}]', org = (top_left[0], top_left[1] - 5), fontFace = 0, fontScale = 0.45, color = [255, 255, 0], thickness = 1)
        cv2.imwrite(f'production_clients/samples/outputs/face_detection_post{idx}.jpg', cv2_image)
    
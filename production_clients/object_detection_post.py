import numpy as np
import tritonclient.grpc as grpcclient

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
    
    def generate_request(self, *input_batches):
        super().generate_request(*input_batches)

        for key, value in self.triton_params.items():
            input_batch = np.full((len(input_batches[0]), 1), fill_value = value, dtype = np.float32)
            
            infer_inputs = grpcclient.InferInput(key, input_batch.shape, 'FP32')
            infer_inputs.set_data_from_numpy(input_batch)
            self.inputs.append(infer_inputs)
    
    def postprocess(self, batch_boxes, batch_labels):
        split_indices = np.where(batch_labels == -1)[0]
        for idx in range(len(split_indices)):
            if idx != 0:
                yield batch_boxes[split_indices[idx - 1] + 1: split_indices[idx]], batch_labels[split_indices[idx - 1] + 1: split_indices[idx]]
                # yield batch_boxes[split_indices[idx - 1] + 1: split_indices[idx]], [self.classes[class_idx] for class_idx in batch_labels[split_indices[idx - 1] + 1: split_indices[idx]]]
            else:
                yield batch_boxes[: split_indices[idx]], batch_labels[: split_indices[idx]]
                # yield batch_boxes[: split_indices[idx]], [self.classes[class_idx] for class_idx in batch_labels[: split_indices[idx]]]

if __name__ == '__main__':
    import cv2
    from face_detection import FaceDetectionGRPCClient
    from pedestrian_detection import PedestrianDetectionGRPCClient

    face_detection_client = PedestrianDetectionGRPCClient(resized_dim = 640)
    
    cv2_image1 = cv2.resize(cv2.imread('production_clients/samples/inputs/pedestrian_detection1.jpg'), (1280, 720))
    preprocessed_inputs1 = face_detection_client.preprocess(cv2_image1)
    
    # cv2_image2 = cv2.resize(cv2.imread('production_clients/samples/inputs/face_detection2.jpg'), (1280, 720))
    # preprocessed_inputs2 = face_detection_client.preprocess(cv2_image2)
    
    face_detection_batch = [preprocessed_inputs1]
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
    preprocessed_inputs = client.preprocess(face_detection_outputs)
    
    with client.monitor_performance():
        for _ in range(100):
            client.perform_inference(preprocessed_inputs)

    for idx, ((boxes, labels), cv2_image) in enumerate(zip(client.perform_inference(preprocessed_inputs), [cv2_image1])):
        for (top_left, bottom_right), label in zip(boxes, labels):
            cv2.rectangle(cv2_image, top_left, bottom_right, [0, 0, 255], 2)
            cv2.putText(cv2_image, f'[{label}]', org = (top_left[0], top_left[1] - 5), fontFace = 0, fontScale = 0.45, color = [255, 255, 0], thickness = 1)
        cv2.imwrite(f'production_clients/samples/outputs/face_detection_post{idx}.jpg', cv2_image)
    
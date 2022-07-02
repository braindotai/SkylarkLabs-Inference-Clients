from clients.object_detection import ObjectDetectionGRPCClient

class FaceDetectionGRPCClient(ObjectDetectionGRPCClient):
    def __init__(
        self,
        bgr2rgb = True,
        **kwargs
    ):
        super().__init__(
            bgr2rgb = bgr2rgb,
            model_name = 'face_detection',
            **kwargs
        )
    

if __name__ == '__main__':
    import cv2
    import numpy as np

    client = FaceDetectionGRPCClient()
    cv2_image = cv2.imread('production_clients/samples/inputs/face_detection1.jpg')
    preprocessed_inputs = client.preprocess(cv2_image)

    print(len(list(client.perform_inference([preprocessed_inputs] * 4))))
    
    with client.monitor_performance():
        for _ in range(100):
            for output in client.perform_inference([preprocessed_inputs] * 4):
                break

        for top_left, bottom_right in output:
            cv2.rectangle(cv2_image, top_left, bottom_right, [0, 0, 255], 2)

        cv2.imwrite('production_clients/samples/outputs/face_detection1.jpg', cv2_image)
    
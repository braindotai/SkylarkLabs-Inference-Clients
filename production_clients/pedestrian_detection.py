from clients.object_detection import ObjectDetectionGRPCClient

class PedestrianDetectionGRPCClient(ObjectDetectionGRPCClient):
    def __init__(
        self,
        bgr2rgb = True,
        **kwargs
    ):
        super().__init__(
            bgr2rgb = bgr2rgb,
            model_name = 'pedestrian_detection',
            **kwargs
        )
    

if __name__ == '__main__':
    import cv2
    import numpy as np

    client = PedestrianDetectionGRPCClient()
    cv2_image = cv2.imread('production_clients/samples/inputs/pedestrian_detection1.jpg')
    preprocessed_inputs = client.preprocess(cv2_image)

    with client.monitor_performance():
        for _ in range(100):
            for output in client.perform_inference([preprocessed_inputs] * 2):
                pass

        for top_left, bottom_right in output:
            cv2.rectangle(cv2_image, top_left, bottom_right, [0, 0, 255], 2)

        cv2.imwrite('production_clients/samples/outputs/pedestrian_detection1.jpg', cv2_image)
            
    
import os
import cv2
from triton_inference_clients.production_clients import DaliDecoderGRPCClient
from triton_inference_clients import utils

def test_dali_decoder():
    client = DaliDecoderGRPCClient()

    input = utils.cv2_imencode(cv2.imread(os.path.join('tests', 'assets', 'images', 'inputs', 'pedestrian_detection1.jpg')))

    output = client.perform_inference([input])

    cv2.imwrite(os.path.join('tests', 'assets', 'images', 'outputs', 'dali_decoder.jpg'), output[0])

    # cv2.imshow('Dali Decoder', output[0])
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
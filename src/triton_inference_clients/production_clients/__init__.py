from .face_detection import FaceDetectionGRPCClient
from .pedestrian_detection import PedestrianDetectionGRPCClient
from .long_range_pedestrian_detection import LongRangePedestrianDetectionGRPCClient

from .batch_similarity import BatchSimilarityGRPCClient
from .face_recognition import FaceRecognitionGRPCClient

from .change_detection import ChangeDetectionLocalClient

from .head_detection import HeadDetectionGRPCClient
from .head_recognition import HeadRecognitionGRPCClient

__all__ = [
    FaceDetectionGRPCClient,
    PedestrianDetectionGRPCClient,
    LongRangePedestrianDetectionGRPCClient,
    BatchSimilarityGRPCClient,
    FaceRecognitionGRPCClient,
    ChangeDetectionLocalClient,
    HeadDetectionGRPCClient,
    HeadRecognitionGRPCClient,
]

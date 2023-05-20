from .face_detection import FaceDetectionGRPCClient
from .pedestrian_detection import PedestrianDetectionGRPCClient
from .long_range_pedestrian_detection import LongRangePedestrianDetectionGRPCClient
from .crowd_detection import CrowdDetectionGRPCClient
from .vehicle_detection import VehicleDetectionGRPCClient
from .gathering_detection import GatheringDetectionGRPCClient

from .batch_similarity import BatchSimilarityGRPCClient
from .face_recognition import FaceRecognitionGRPCClient
from .facial_landmarks import FacialLandmarksGRPCClient
from .face_orientation_detection import FaceOrientationDetection

from .change_detection import ChangeDetectionLocalClient

from .head_detection import HeadDetectionGRPCClient
from .head_recognition import HeadRecognitionGRPCClient

from .feature_based_tracking import FeatureBasedTrackingGRPCClient
from .violence_detection import ViolenceDetectionGRPCClient


__all__ = [
    FaceDetectionGRPCClient,
    VehicleDetectionGRPCClient,
    LongRangePedestrianDetectionGRPCClient,
    CrowdDetectionGRPCClient,
    PedestrianDetectionGRPCClient,
    GatheringDetectionGRPCClient,
    BatchSimilarityGRPCClient,
    FaceRecognitionGRPCClient,
    FacialLandmarksGRPCClient,
    FaceOrientationDetection,
    ChangeDetectionLocalClient,
    HeadDetectionGRPCClient,
    HeadRecognitionGRPCClient,
    FeatureBasedTrackingGRPCClient,
    ViolenceDetectionGRPCClient,
]

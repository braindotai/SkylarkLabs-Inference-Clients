import torch
from torch import nn
from torchvision import models
import numpy as np
import time
import cv2

# from memory_profiler import profile
from .batch_similarity import BatchSimilarityGRPCClient
from ..utils import resize_box


class FeatureBasedTrackingGRPCClient:
    def __init__(
        self,
        cosine_similarity_threshold: float = 0.8,
    ):
        self.cosine_similarity_threshold = cosine_similarity_threshold

        # self.feature_extractor = models.mnasnet0_5(weights = models.MNASNet0_5_Weights).layers.cuda()
        # self.feature_extractor = models.mnasnet0_5(weights = models.MNASNet0_5_Weights).cuda()
        # self.feature_extractor.classifier = nn.Identity()
        # self.feature_extractor.eval()

        # self.feature_extractor = models.mnasnet0_5(weights = models.MNASNet0_5_Weights).layers.cuda()
        self.feature_extractor = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).cuda()
        self.feature_extractor.classifier = nn.Sequential(
            self.feature_extractor.classifier[0],
            self.feature_extractor.classifier[1],
        )
        self.feature_extractor.eval()

        self.track_history = []
        self.batch_similarity = BatchSimilarityGRPCClient()


    # @torch.inference_mode()
    # def _compute_gram_matrix_features(self, query_features, reference_features):
        # query_features -> (-1, 1280, 7, 7)
        # reference_features -> (-1, 1280, 7, 7)

        # qb, c, w, h = query_features.shape
        # query_features = query_features.view(qb, c, w * h)
        
        # rb, c, w, h = reference_features.shape
        # reference_features = reference_features.view(rb, c, w * h)

        # return (
        #     torch.bmm(query_features, query_features.permute(0, 2, 1)).view(-1, c * c), # (qb, c, w * h) @ (qb, w * h, c) -> (qb, c, c)
        #     torch.bmm(reference_features, reference_features.permute(0, 2, 1)).view(-1, c * c), # (rb, c, w * h) @ (rb, w * h, c) -> (rb, c, c)
        # )

        # return query_features, reference_features

    
    def aspect_ratio_resize(self, image, resize_dim):  
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if h > w:
            aspect_ratio = w / h
            h, w = resize_dim, int(aspect_ratio * resize_dim)
            image = cv2.resize(image, (w, h))
            padding = resize_dim - w
            image = cv2.copyMakeBorder(image, 0, 0, int(padding / 2), padding - int(padding / 2), cv2.BORDER_CONSTANT, value = (0, 0, 0))
        else:
            aspect_ratio = h / w
            h, w = int(aspect_ratio * resize_dim), resize_dim
            image = cv2.resize(image, (w, h))
            padding = resize_dim - h
            image = cv2.copyMakeBorder(image, int(padding / 2), padding - int(padding / 2), 0, 0, cv2.BORDER_CONSTANT, value = (0, 0, 0))
        
        return image


    def perform_inference(self, boxes, frame, frame_idx):
        crops = []

        for (top_left, bottom_right) in boxes:
            top_left, bottom_right = resize_box(top_left, bottom_right, (frame.shape[1], frame.shape[0]))
            resized_image = self.aspect_ratio_resize(frame[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :], 224)
            # cv2.imwrite(f'samples/{frame_idx}-{top_left}.jpg', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
            resized_image = resized_image.astype('float32')
            resized_image = (resized_image - resized_image.min()) / resized_image.ptp()
            crops.append(resized_image)

        crops_tensor = torch.from_numpy(np.array(crops))
        print(crops_tensor.shape)
        crops_tensor[:, :, :, 0] = (crops_tensor[:, :, :, 0] - 0.485) / 0.229
        crops_tensor[:, :, :, 1] = (crops_tensor[:, :, :, 1] - 0.456) / 0.224
        crops_tensor[:, :, :, 2] = (crops_tensor[:, :, :, 2] - 0.406) / 0.225

        crops_tensor = crops_tensor.permute(0, 3, 1, 2).cuda()
        
        with torch.inference_mode():
            query_features = self.feature_extractor(crops_tensor)

        tracking_results = []
        
        if len(self.track_history) == 0:
            for idx, features in enumerate(query_features):
                random_color = list(np.random.random(size = 3) * 256)
                self.track_history.append({
                    'features': features,
                    'id': [idx],
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'color': random_color,
                    'matches': [],
                    'matched': False,
                })
                tracking_results.append((idx, False, random_color))
        else:
            reference_features = torch.stack([track['features'] for track in self.track_history])
            # query_features, reference_features = self._compute_gram_matrix_features(query_features, reference_features)

            matched_indices, max_similarities, min_similarities = self.batch_similarity.perform_inference(
                query_features.cpu().numpy(),
                reference_features.cpu().numpy(),
            )
            print('=' * 60, '\n', max_similarities, '\n', min_similarities)

            for matched_idx, max_sim, min_sim, features in zip(matched_indices, max_similarities, min_similarities, query_features):
                if max_sim > (self.cosine_similarity_threshold * 0.8):
                    selected_track = self.track_history[matched_idx]
                    selected_track['features'] = features
                    selected_track['last_seen'] = time.time()
                    selected_track['matches'].append(max_sim)
                    selected_track['matches'] = selected_track['matches'][-600:]
                    selected_track['matched'] = sum(selected_track['matches']) / len(selected_track['matches']) >= self.cosine_similarity_threshold
                    tracking_results.append((selected_track['id'], selected_track['matched'], selected_track['color']))
                else:
                    random_color = list(np.random.random(size = 3) * 256)
                    new_track = {
                        'features': features,
                        'id': len(self.track_history),
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                        'color': random_color,
                        'matches': [],
                        'matched': False
                    }
                    self.track_history.append(new_track)
                    tracking_results.append((new_track['id'], False, new_track['color']))
        
        del self.track_history[:-1000]

        return tracking_results
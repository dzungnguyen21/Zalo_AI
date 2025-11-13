"""
Similarity Matching Module for Drone Object Detection
Matches detected objects with reference features using cosine similarity
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass

from feature_extract import FeatureExtractor
from object_detector import Detection, compute_iou


@dataclass
class ScoredDetection:
    """Detection with similarity score"""
    detection: Detection
    similarity_score: float
    feature_vector: Optional[torch.Tensor] = None


class SimilarityMatcher:
    """
    Match detected objects with reference features based on visual similarity
    """
    
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 similarity_threshold: float = 0.5,
                 top_k: Optional[int] = None):
        """
        Initialize similarity matcher
        
        Args:
            feature_extractor: FeatureExtractor instance
            similarity_threshold: Minimum similarity score to keep detection
            top_k: Keep only top-k most similar detections (None = keep all above threshold)
        """
        self.feature_extractor = feature_extractor
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
    
    def match_detections(self,
                        frame: np.ndarray,
                        detections: List[Detection],
                        reference_features: torch.Tensor) -> List[ScoredDetection]:
        """
        Match detections with reference features
        
        Args:
            frame: Image as numpy array (H, W, C) in BGR format
            detections: List of Detection objects
            reference_features: Reference feature vector
            
        Returns:
            List of ScoredDetection objects sorted by similarity (highest first)
        """
        if len(detections) == 0:
            return []
        
        # Extract features from detected regions
        bboxes = [det.bbox for det in detections]
        detection_features = self.feature_extractor.extract_patch_features(frame, bboxes)
        
        # Compute similarities
        similarities = self.feature_extractor.compute_similarity(
            reference_features, detection_features
        )
        
        # Create scored detections
        scored_detections = []
        for det, sim, feat in zip(detections, similarities.cpu().numpy(), detection_features):
            if sim >= self.similarity_threshold:
                scored_detections.append(ScoredDetection(
                    detection=det,
                    similarity_score=float(sim),
                    feature_vector=feat
                ))
        
        # Sort by similarity score (highest first)
        scored_detections.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Keep top-k if specified
        if self.top_k is not None and len(scored_detections) > self.top_k:
            scored_detections = scored_detections[:self.top_k]
        
        return scored_detections
    
    def match_with_weights(self,
                          frame: np.ndarray,
                          detections: List[Detection],
                          reference_features: torch.Tensor,
                          detection_weight: float = 0.5,
                          similarity_weight: float = 0.5) -> List[ScoredDetection]:
        """
        Match detections using weighted combination of detection confidence and similarity
        
        Args:
            frame: Image as numpy array
            detections: List of Detection objects
            reference_features: Reference feature vector
            detection_weight: Weight for detection confidence
            similarity_weight: Weight for similarity score
            
        Returns:
            List of ScoredDetection objects sorted by combined score
        """
        if len(detections) == 0:
            return []
        
        # Extract features and compute similarities
        bboxes = [det.bbox for det in detections]
        detection_features = self.feature_extractor.extract_patch_features(frame, bboxes)
        similarities = self.feature_extractor.compute_similarity(
            reference_features, detection_features
        )
        
        # Compute weighted scores
        scored_detections = []
        for det, sim, feat in zip(detections, similarities.cpu().numpy(), detection_features):
            combined_score = (
                detection_weight * det.confidence +
                similarity_weight * sim
            )
            
            if sim >= self.similarity_threshold:
                scored_detections.append(ScoredDetection(
                    detection=det,
                    similarity_score=float(combined_score),
                    feature_vector=feat
                ))
        
        # Sort by combined score
        scored_detections.sort(key=lambda x: x.similarity_score, reverse=True)
        
        if self.top_k is not None and len(scored_detections) > self.top_k:
            scored_detections = scored_detections[:self.top_k]
        
        return scored_detections


class TemporalSimilarityMatcher:
    """
    Enhanced matcher that uses temporal information to improve matching
    """
    
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 similarity_threshold: float = 0.5,
                 temporal_window: int = 5,
                 temporal_weight: float = 0.3):
        """
        Initialize temporal similarity matcher
        
        Args:
            feature_extractor: FeatureExtractor instance
            similarity_threshold: Minimum similarity score
            temporal_window: Number of previous frames to consider
            temporal_weight: Weight for temporal consistency
        """
        self.feature_extractor = feature_extractor
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window
        self.temporal_weight = temporal_weight
        
        # History of detections
        self.detection_history: List[List[ScoredDetection]] = []
    
    def match_with_temporal_context(self,
                                   frame: np.ndarray,
                                   detections: List[Detection],
                                   reference_features: torch.Tensor) -> List[ScoredDetection]:
        """
        Match detections considering temporal context from previous frames
        
        Args:
            frame: Current frame
            detections: Current frame detections
            reference_features: Reference feature vector
            
        Returns:
            List of ScoredDetection objects
        """
        if len(detections) == 0:
            self.detection_history.append([])
            return []
        
        # Extract features and compute base similarities
        bboxes = [det.bbox for det in detections]
        detection_features = self.feature_extractor.extract_patch_features(frame, bboxes)
        similarities = self.feature_extractor.compute_similarity(
            reference_features, detection_features
        )
        
        # Boost scores based on temporal consistency
        scored_detections = []
        for det, sim, feat in zip(detections, similarities.cpu().numpy(), detection_features):
            temporal_boost = self._compute_temporal_boost(det.bbox)
            final_score = (1 - self.temporal_weight) * sim + self.temporal_weight * temporal_boost
            
            if final_score >= self.similarity_threshold:
                scored_detections.append(ScoredDetection(
                    detection=det,
                    similarity_score=float(final_score),
                    feature_vector=feat
                ))
        
        # Sort by score
        scored_detections.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update history
        self.detection_history.append(scored_detections)
        if len(self.detection_history) > self.temporal_window:
            self.detection_history.pop(0)
        
        return scored_detections
    
    def _compute_temporal_boost(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Compute temporal consistency boost for a bounding box
        
        Args:
            bbox: Current bounding box
            
        Returns:
            Boost score based on overlap with previous detections
        """
        if len(self.detection_history) == 0:
            return 0.0
        
        max_iou = 0.0
        for past_detections in reversed(self.detection_history[-self.temporal_window:]):
            for past_det in past_detections:
                iou = compute_iou(bbox, past_det.detection.bbox)
                if iou > max_iou:
                    max_iou = iou
        
        return max_iou
    
    def reset(self):
        """Reset temporal history"""
        self.detection_history = []


class MultiScaleMatcher:
    """
    Match objects at multiple scales for better robustness
    """
    
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 similarity_threshold: float = 0.5,
                 scales: List[float] = [0.8, 1.0, 1.2]):
        """
        Initialize multi-scale matcher
        
        Args:
            feature_extractor: FeatureExtractor instance
            similarity_threshold: Minimum similarity score
            scales: Scaling factors to apply to bounding boxes
        """
        self.feature_extractor = feature_extractor
        self.similarity_threshold = similarity_threshold
        self.scales = scales
    
    def match_multiscale(self,
                        frame: np.ndarray,
                        detections: List[Detection],
                        reference_features: torch.Tensor) -> List[ScoredDetection]:
        """
        Match detections at multiple scales
        
        Args:
            frame: Image as numpy array
            detections: List of Detection objects
            reference_features: Reference feature vector
            
        Returns:
            List of ScoredDetection objects with best scale
        """
        if len(detections) == 0:
            return []
        
        all_scored = []
        
        for det in detections:
            best_score = 0.0
            best_features = None
            
            # Try different scales
            for scale in self.scales:
                scaled_bbox = self._scale_bbox(det.bbox, scale, frame.shape)
                features = self.feature_extractor.extract_patch_features(
                    frame, [scaled_bbox]
                )
                
                if features.shape[0] > 0:
                    similarity = self.feature_extractor.compute_similarity(
                        reference_features, features
                    )[0].item()
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_features = features[0]
            
            if best_score >= self.similarity_threshold:
                all_scored.append(ScoredDetection(
                    detection=det,
                    similarity_score=best_score,
                    feature_vector=best_features
                ))
        
        # Sort by similarity
        all_scored.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return all_scored
    
    def _scale_bbox(self, 
                   bbox: Tuple[int, int, int, int], 
                   scale: float,
                   frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Scale a bounding box around its center
        
        Args:
            bbox: Original bounding box
            scale: Scaling factor
            frame_shape: Frame dimensions (H, W, C)
            
        Returns:
            Scaled bounding box
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        
        # Compute center and size
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Scale
        new_width = width * scale
        new_height = height * scale
        
        # Compute new coordinates
        new_x1 = int(cx - new_width / 2)
        new_y1 = int(cy - new_height / 2)
        new_x2 = int(cx + new_width / 2)
        new_y2 = int(cy + new_height / 2)
        
        # Clip to frame boundaries
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(w, new_x2)
        new_y2 = min(h, new_y2)
        
        return (new_x1, new_y1, new_x2, new_y2)


if __name__ == "__main__":
    # Example usage
    from feature_extract import FeatureExtractor
    from object_detector import ObjectDetector
    
    # Initialize components
    extractor = FeatureExtractor(model_name="clip")
    detector = ObjectDetector()
    matcher = SimilarityMatcher(extractor, similarity_threshold=0.5)
    
    # Extract reference features
    ref_images = ["path/to/ref1.jpg", "path/to/ref2.jpg", "path/to/ref3.jpg"]
    ref_features = extractor.extract_reference_features(ref_images)
    
    # Process frame
    frame = cv2.imread("path/to/frame.jpg")
    detections = detector.detect(frame)
    
    # Match detections
    matched = matcher.match_detections(frame, detections, ref_features)
    
    print(f"Found {len(matched)} matching objects")
    for scored_det in matched:
        det = scored_det.detection
        print(f"  {det.class_name}: similarity={scored_det.similarity_score:.3f} at {det.bbox}")

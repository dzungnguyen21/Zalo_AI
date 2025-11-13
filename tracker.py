"""
Tracking Module for Drone Object Detection
Implements SORT and ByteTrack-inspired tracking algorithms
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from object_detector import Detection, compute_iou
from similarity_matcher import ScoredDetection


@dataclass
class Track:
    """Represents a tracked object"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    feature_vector: Optional[np.ndarray] = None
    age: int = 0  # Number of frames since last update
    hits: int = 0  # Number of successful matches
    time_since_update: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes
    State: [x_center, y_center, area, aspect_ratio, dx, dy, da, dr]
    """
    
    count = 0
    
    def __init__(self, bbox: Tuple[int, int, int, int]):
        """
        Initialize Kalman filter with a bounding box
        
        Args:
            bbox: Initial bounding box (x1, y1, x2, y2)
        """
        # Initialize Kalman filter (7D state, 4D measurement)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_state(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def _bbox_to_state(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert bounding box to state representation"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h  # area
        r = w / float(h) if h > 0 else 1.0  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _state_to_bbox(self, state: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert state representation to bounding box"""
        x, y, s, r = state[:4, 0]
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 1.0
        x1 = int(x - w / 2.0)
        y1 = int(y - h / 2.0)
        x2 = int(x + w / 2.0)
        y2 = int(y + h / 2.0)
        return (x1, y1, x2, y2)
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """
        Update the state with a new detection
        
        Args:
            bbox: Detected bounding box (x1, y1, x2, y2)
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_state(bbox))
    
    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict the next state
        
        Returns:
            Predicted bounding box (x1, y1, x2, y2)
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self._state_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current bounding box"""
        return self._state_to_bbox(self.kf.x)


class SORTTracker:
    """
    Simple Online and Realtime Tracking (SORT) algorithm
    """
    
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize SORT tracker
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
    
    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Track]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of bounding boxes (x1, y1, x2, y2)
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)
        
        # Generate output tracks
        tracks = []
        for trk in self.trackers:
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = trk.get_state()
                tracks.append(Track(
                    track_id=trk.id,
                    bbox=bbox,
                    confidence=1.0,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update,
                    state="confirmed"
                ))
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        return tracks
    
    def _associate_detections_to_trackers(self,
                                         detections: List[Tuple[int, int, int, int]],
                                         trackers: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Assign detections to tracked objects using IoU
        
        Returns:
            matched_indices, unmatched_detections, unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = compute_iou(det, tuple(trk.astype(int)))
        
        # Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.column_stack((row_ind, col_ind))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Find unmatched detections
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        # Find unmatched trackers
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, unmatched_detections, unmatched_trackers
    
    def reset(self):
        """Reset tracker"""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0


class FeatureBasedTracker(SORTTracker):
    """
    Enhanced tracker that uses both IoU and feature similarity
    """
    
    def __init__(self,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 feature_threshold: float = 0.5,
                 feature_weight: float = 0.5):
        """
        Initialize feature-based tracker
        
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before a track is confirmed
            iou_threshold: Minimum IoU for matching
            feature_threshold: Minimum feature similarity for matching
            feature_weight: Weight for feature similarity (0-1)
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.feature_threshold = feature_threshold
        self.feature_weight = feature_weight
        self.tracker_features: Dict[int, np.ndarray] = {}
    
    def update_with_features(self,
                            scored_detections: List[ScoredDetection]) -> List[Track]:
        """
        Update tracks with detections that have feature vectors
        
        Args:
            scored_detections: List of ScoredDetection objects
            
        Returns:
            List of active tracks
        """
        if len(scored_detections) == 0:
            return self.update([])
        
        # Extract bboxes and features
        detections = [sd.detection.bbox for sd in scored_detections]
        features = [sd.feature_vector.cpu().numpy() if sd.feature_vector is not None else None 
                   for sd in scored_detections]
        
        self.frame_count += 1
        
        # Get predictions from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        for t in reversed(to_del):
            tracker = self.trackers.pop(t)
            if tracker.id in self.tracker_features:
                del self.tracker_features[tracker.id]
        
        # Associate using both IoU and features
        matched, unmatched_dets, unmatched_trks = self._associate_with_features(
            detections, features, trks
        )
        
        # Update matched trackers
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(detections[det_idx])
            # Update feature
            if features[det_idx] is not None:
                trk_id = self.trackers[trk_idx].id
                if trk_id in self.tracker_features:
                    # Exponential moving average
                    self.tracker_features[trk_id] = (
                        0.7 * self.tracker_features[trk_id] + 
                        0.3 * features[det_idx]
                    )
                else:
                    self.tracker_features[trk_id] = features[det_idx]
        
        # Create new trackers
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)
            if features[i] is not None:
                self.tracker_features[trk.id] = features[i]
        
        # Generate output
        tracks = []
        for trk in self.trackers:
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = trk.get_state()
                feature = self.tracker_features.get(trk.id, None)
                tracks.append(Track(
                    track_id=trk.id,
                    bbox=bbox,
                    confidence=1.0,
                    feature_vector=feature,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update,
                    state="confirmed"
                ))
        
        # Remove dead trackers
        dead_trackers = [t for t in self.trackers if t.time_since_update >= self.max_age]
        for t in dead_trackers:
            if t.id in self.tracker_features:
                del self.tracker_features[t.id]
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        return tracks
    
    def _associate_with_features(self,
                                 detections: List[Tuple[int, int, int, int]],
                                 features: List[Optional[np.ndarray]],
                                 trackers: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Associate detections to trackers using both IoU and feature similarity
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = compute_iou(det, tuple(trk.astype(int)))
        
        # Compute feature similarity matrix
        feature_matrix = np.zeros((len(detections), len(trackers)))
        for d, feat in enumerate(features):
            if feat is None:
                continue
            for t, trk in enumerate(self.trackers):
                if trk.id in self.tracker_features:
                    trk_feat = self.tracker_features[trk.id]
                    # Cosine similarity
                    similarity = np.dot(feat.flatten(), trk_feat.flatten()) / (
                        np.linalg.norm(feat) * np.linalg.norm(trk_feat) + 1e-8
                    )
                    feature_matrix[d, t] = similarity
        
        # Combine IoU and feature similarity
        combined_matrix = (
            (1 - self.feature_weight) * iou_matrix +
            self.feature_weight * feature_matrix
        )
        
        # Hungarian algorithm
        if min(combined_matrix.shape) > 0:
            row_ind, col_ind = linear_sum_assignment(-combined_matrix)
            matched_indices = np.column_stack((row_ind, col_ind))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Find unmatched
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter matches
        matches = []
        for m in matched_indices:
            # Require either good IoU or good feature match
            iou_ok = iou_matrix[m[0], m[1]] >= self.iou_threshold
            feat_ok = feature_matrix[m[0], m[1]] >= self.feature_threshold
            
            if iou_ok or feat_ok:
                matches.append(m.reshape(1, 2))
            else:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, unmatched_detections, unmatched_trackers


if __name__ == "__main__":
    # Example usage
    tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Simulate detections over frames
    for frame_idx in range(100):
        # Example detections
        detections = [
            (100 + frame_idx, 100, 200 + frame_idx, 200),
            (300, 300 + frame_idx, 400, 400 + frame_idx)
        ]
        
        tracks = tracker.update(detections)
        
        print(f"Frame {frame_idx}: {len(tracks)} active tracks")
        for track in tracks:
            print(f"  Track {track.track_id}: {track.bbox}")

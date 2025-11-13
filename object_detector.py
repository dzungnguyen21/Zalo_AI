"""
Object Detection Module for Drone Object Detection
Uses YOLOv8 and SAM for generating bounding box candidates
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional
from ultralytics import YOLO
from dataclasses import dataclass


@dataclass
class Detection:
    """Data class for detection results"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class ObjectDetector:
    """
    Object detector using YOLOv8 for generating bounding box candidates
    """
    
    def __init__(self, 
                 model_name: str = "yolov8x.pt",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = None):
        """
        Initialize object detector
        
        Args:
            model_name: YOLOv8 model name (yolov8n/s/m/l/x)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Loading YOLO model {model_name} on {self.device}...")
        self.model = YOLO(model_name)
        
    def detect(self, frame: np.ndarray, 
               conf_threshold: Optional[float] = None,
               classes: Optional[List[int]] = None) -> List[Detection]:
        """
        Detect objects in a frame
        
        Args:
            frame: Image as numpy array (H, W, C) in BGR format
            conf_threshold: Override default confidence threshold
            classes: Filter by specific class IDs (None = all classes)
            
        Returns:
            List of Detection objects
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # Run inference
        results = self.model(
            frame,
            conf=conf,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = result.names[class_id]
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                ))
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray],
                    conf_threshold: Optional[float] = None,
                    classes: Optional[List[int]] = None) -> List[List[Detection]]:
        """
        Detect objects in multiple frames
        
        Args:
            frames: List of images as numpy arrays
            conf_threshold: Override default confidence threshold
            classes: Filter by specific class IDs
            
        Returns:
            List of detection lists, one per frame
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # Run batch inference
        results = self.model(
            frames,
            conf=conf,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )
        
        all_detections = []
        for result in results:
            detections = []
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = result.names[class_id]
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                ))
            all_detections.append(detections)
        
        return all_detections
    
    def get_bboxes(self, detections: List[Detection]) -> List[Tuple[int, int, int, int]]:
        """
        Extract bounding boxes from detections
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2)
        """
        return [det.bbox for det in detections]
    
    def filter_by_class(self, detections: List[Detection], 
                       class_names: List[str]) -> List[Detection]:
        """
        Filter detections by class names
        
        Args:
            detections: List of Detection objects
            class_names: List of class names to keep
            
        Returns:
            Filtered list of detections
        """
        return [det for det in detections if det.class_name in class_names]
    
    def filter_by_size(self, detections: List[Detection],
                      min_area: int = 100,
                      max_area: Optional[int] = None) -> List[Detection]:
        """
        Filter detections by bounding box area
        
        Args:
            detections: List of Detection objects
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area (None = no limit)
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                if max_area is None or area <= max_area:
                    filtered.append(det)
        return filtered


class SlidingWindowDetector:
    """
    Generate candidate bounding boxes using sliding window approach
    Useful when combined with feature matching
    """
    
    def __init__(self, 
                 window_sizes: List[Tuple[int, int]] = [(64, 64), (128, 128), (256, 256)],
                 stride_ratio: float = 0.5):
        """
        Initialize sliding window detector
        
        Args:
            window_sizes: List of (width, height) for sliding windows
            stride_ratio: Stride as ratio of window size
        """
        self.window_sizes = window_sizes
        self.stride_ratio = stride_ratio
    
    def generate_windows(self, frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Generate sliding window bounding boxes
        
        Args:
            frame_shape: (height, width) of the frame
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2)
        """
        h, w = frame_shape
        windows = []
        
        for win_w, win_h in self.window_sizes:
            stride_w = int(win_w * self.stride_ratio)
            stride_h = int(win_h * self.stride_ratio)
            
            for y in range(0, h - win_h + 1, stride_h):
                for x in range(0, w - win_w + 1, stride_w):
                    windows.append((x, y, x + win_w, y + win_h))
        
        return windows
    
    def generate_windows_adaptive(self, 
                                  frame_shape: Tuple[int, int],
                                  reference_size: Tuple[int, int],
                                  scale_factors: List[float] = [0.5, 1.0, 1.5, 2.0]) -> List[Tuple[int, int, int, int]]:
        """
        Generate sliding windows based on reference object size
        
        Args:
            frame_shape: (height, width) of the frame
            reference_size: (width, height) of the reference object
            scale_factors: Scaling factors to apply to reference size
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2)
        """
        h, w = frame_shape
        ref_w, ref_h = reference_size
        windows = []
        
        for scale in scale_factors:
            win_w = int(ref_w * scale)
            win_h = int(ref_h * scale)
            
            if win_w > w or win_h > h:
                continue
            
            stride_w = int(win_w * self.stride_ratio)
            stride_h = int(win_h * self.stride_ratio)
            
            for y in range(0, h - win_h + 1, stride_h):
                for x in range(0, w - win_w + 1, stride_w):
                    windows.append((x, y, x + win_w, y + win_h))
        
        return windows


def compute_iou(bbox1: Tuple[int, int, int, int], 
                bbox2: Tuple[int, int, int, int]) -> float:
    """
    Compute IoU between two bounding boxes
    
    Args:
        bbox1, bbox2: Bounding boxes as (x1, y1, x2, y2)
        
    Returns:
        IoU score
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(detections: List[Detection], 
                           iou_threshold: float = 0.5) -> List[Detection]:
    """
    Apply NMS to remove overlapping detections
    
    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    keep = []
    while len(detections) > 0:
        # Keep the detection with highest confidence
        best = detections[0]
        keep.append(best)
        detections = detections[1:]
        
        # Remove detections with high IoU
        filtered = []
        for det in detections:
            if compute_iou(best.bbox, det.bbox) < iou_threshold:
                filtered.append(det)
        detections = filtered
    
    return keep


if __name__ == "__main__":
    # Example usage
    detector = ObjectDetector(model_name="yolov8x.pt")
    
    # Detect objects in a frame
    frame = cv2.imread("path/to/frame.jpg")
    detections = detector.detect(frame, conf_threshold=0.3)
    
    print(f"Found {len(detections)} objects")
    for det in detections:
        print(f"  {det.class_name}: {det.confidence:.3f} at {det.bbox}")

"""
Main Inference Pipeline for Drone Object Detection Challenge
Combines detection, feature matching, and tracking
"""

import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from feature_extract import FeatureExtractor, load_reference_images
from object_detector import ObjectDetector
from similarity_matcher import SimilarityMatcher, TemporalSimilarityMatcher
from tracker import FeatureBasedTracker, Track


class DroneObjectDetectionPipeline:
    """
    End-to-end pipeline for detecting and tracking objects in drone videos
    """
    
    def __init__(self,
                 feature_model: str = "clip",
                 detector_model: str = "yolov8x.pt",
                 similarity_threshold: float = 0.5,
                 detection_conf: float = 0.25,
                 use_temporal: bool = True,
                 use_tracking: bool = True,
                 device: str = None):
        """
        Initialize the pipeline
        
        Args:
            feature_model: Feature extraction model ('clip', 'dinov2', or 'both')
            detector_model: YOLO model name
            similarity_threshold: Minimum similarity score for matching
            detection_conf: Detection confidence threshold
            use_temporal: Use temporal similarity matching
            use_tracking: Use tracking to smooth detections
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Initializing pipeline components...")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            model_name=feature_model,
            device=self.device
        )
        
        # Initialize object detector
        self.detector = ObjectDetector(
            model_name=detector_model,
            conf_threshold=detection_conf,
            device=self.device
        )
        
        # Initialize similarity matcher
        if use_temporal:
            self.matcher = TemporalSimilarityMatcher(
                feature_extractor=self.feature_extractor,
                similarity_threshold=similarity_threshold
            )
        else:
            self.matcher = SimilarityMatcher(
                feature_extractor=self.feature_extractor,
                similarity_threshold=similarity_threshold
            )
        
        # Initialize tracker
        self.use_tracking = use_tracking
        if use_tracking:
            self.tracker = FeatureBasedTracker(
                max_age=30,
                min_hits=3,
                iou_threshold=0.3,
                feature_threshold=0.4
            )
        else:
            self.tracker = None
        
        print("Pipeline initialized successfully!")
    
    def process_video(self,
                     video_path: str,
                     reference_images: List[str],
                     frame_skip: int = 1,
                     max_frames: Optional[int] = None) -> Dict:
        """
        Process a single video
        
        Args:
            video_path: Path to drone video
            reference_images: List of paths to reference images
            frame_skip: Process every N-th frame (1 = all frames)
            max_frames: Maximum number of frames to process (None = all)
            
        Returns:
            Dictionary with detection results
        """
        print(f"\nProcessing video: {video_path}")
        
        # Extract reference features
        print("Extracting reference features...")
        ref_features = self.feature_extractor.extract_reference_features(reference_images)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames @ {fps:.2f} FPS")
        
        # Reset matcher and tracker
        if hasattr(self.matcher, 'reset'):
            self.matcher.reset()
        if self.tracker is not None:
            self.tracker.reset()
        
        # Process frames
        detections_by_frame = {}
        frame_idx = 0
        processed_count = 0
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue
                
                # Detect objects
                detections = self.detector.detect(frame)
                
                # Match with reference
                if isinstance(self.matcher, TemporalSimilarityMatcher):
                    scored_detections = self.matcher.match_with_temporal_context(
                        frame, detections, ref_features
                    )
                else:
                    scored_detections = self.matcher.match_detections(
                        frame, detections, ref_features
                    )
                
                # Track objects
                if self.use_tracking and len(scored_detections) > 0:
                    tracks = self.tracker.update_with_features(scored_detections)
                    # Store tracks as detections
                    if len(tracks) > 0:
                        # Use the best track (highest confidence)
                        best_track = max(tracks, key=lambda t: t.confidence)
                        detections_by_frame[frame_idx] = {
                            'bbox': best_track.bbox,
                            'confidence': best_track.confidence,
                            'track_id': best_track.track_id
                        }
                elif len(scored_detections) > 0:
                    # No tracking, use best detection
                    best_det = scored_detections[0]
                    detections_by_frame[frame_idx] = {
                        'bbox': best_det.detection.bbox,
                        'confidence': best_det.similarity_score,
                        'track_id': -1
                    }
                
                processed_count += 1
                frame_idx += 1
                pbar.update(1)
                
                if max_frames is not None and processed_count >= max_frames:
                    break
        
        cap.release()
        
        # Group consecutive detections
        detection_sequences = self._group_detections(detections_by_frame)
        
        print(f"Found {len(detection_sequences)} detection sequences")
        
        return {
            'detections': detection_sequences,
            'total_frames': total_frames,
            'processed_frames': processed_count
        }
    
    def _group_detections(self, 
                         detections_by_frame: Dict[int, Dict]) -> List[Dict]:
        """
        Group consecutive detections into sequences
        
        Args:
            detections_by_frame: Dict mapping frame_idx to detection info
            
        Returns:
            List of detection sequences
        """
        if not detections_by_frame:
            return []
        
        # Sort frames
        sorted_frames = sorted(detections_by_frame.keys())
        
        sequences = []
        current_sequence = []
        prev_frame = -100
        
        for frame_idx in sorted_frames:
            det_info = detections_by_frame[frame_idx]
            
            # Start new sequence if gap is too large
            if frame_idx - prev_frame > 10:  # Max gap of 10 frames
                if current_sequence:
                    sequences.append({'bboxes': current_sequence})
                current_sequence = []
            
            # Add to current sequence
            x1, y1, x2, y2 = det_info['bbox']
            current_sequence.append({
                'frame': int(frame_idx),
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            })
            
            prev_frame = frame_idx
        
        # Add last sequence
        if current_sequence:
            sequences.append({'bboxes': current_sequence})
        
        return sequences
    
    def process_dataset(self,
                       dataset_dir: str,
                       output_path: str,
                       frame_skip: int = 1) -> Dict:
        """
        Process entire dataset
        
        Args:
            dataset_dir: Path to dataset directory (containing samples/)
            output_path: Path to save predictions JSON
            frame_skip: Process every N-th frame
            
        Returns:
            Dictionary with all predictions
        """
        samples_dir = Path(dataset_dir) / "samples"
        
        if not samples_dir.exists():
            raise ValueError(f"Samples directory not found: {samples_dir}")
        
        # Get all video directories
        video_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
        
        print(f"Found {len(video_dirs)} videos to process")
        
        all_predictions = []
        
        for video_dir in video_dirs:
            video_id = video_dir.name
            
            # Find video file
            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
            if not video_files:
                print(f"Warning: No video file found in {video_dir}")
                continue
            
            video_path = str(video_files[0])
            
            # Find reference images
            ref_dir = video_dir / "object_images"
            if not ref_dir.exists():
                print(f"Warning: No object_images directory in {video_dir}")
                continue
            
            ref_images = load_reference_images(str(ref_dir))
            if not ref_images:
                print(f"Warning: No reference images found in {ref_dir}")
                continue
            
            # Process video
            try:
                result = self.process_video(
                    video_path=video_path,
                    reference_images=ref_images,
                    frame_skip=frame_skip
                )
                
                all_predictions.append({
                    'video_id': video_id,
                    'detections': result['detections']
                })
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                all_predictions.append({
                    'video_id': video_id,
                    'detections': []
                })
        
        # Save predictions
        with open(output_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"\nPredictions saved to: {output_path}")
        
        return all_predictions


def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Drone Object Detection Pipeline")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="predictions.json",
                       help="Output JSON file path")
    parser.add_argument("--feature_model", type=str, default="clip",
                       choices=["clip", "dinov2", "both"],
                       help="Feature extraction model")
    parser.add_argument("--detector_model", type=str, default="yolov8x.pt",
                       help="YOLO model name")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                       help="Similarity threshold for matching")
    parser.add_argument("--detection_conf", type=float, default=0.25,
                       help="Detection confidence threshold")
    parser.add_argument("--frame_skip", type=int, default=1,
                       help="Process every N-th frame")
    parser.add_argument("--no_temporal", action="store_true",
                       help="Disable temporal matching")
    parser.add_argument("--no_tracking", action="store_true",
                       help="Disable tracking")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DroneObjectDetectionPipeline(
        feature_model=args.feature_model,
        detector_model=args.detector_model,
        similarity_threshold=args.similarity_threshold,
        detection_conf=args.detection_conf,
        use_temporal=not args.no_temporal,
        use_tracking=not args.no_tracking,
        device=args.device
    )
    
    # Process dataset
    pipeline.process_dataset(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        frame_skip=args.frame_skip
    )


if __name__ == "__main__":
    main()

"""
Interactive Demo Notebook
Use this script to explore the pipeline interactively
"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import our modules
from feature_extract import FeatureExtractor, load_reference_images
from object_detector import ObjectDetector
from similarity_matcher import SimilarityMatcher
from tracker import FeatureBasedTracker
from evaluate import compute_stiou, evaluate_predictions


class InteractiveDemo:
    """Interactive demo for exploring the pipeline"""
    
    def __init__(self):
        """Initialize demo"""
        print("Initializing Drone Object Detection Demo...")
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(model_name="clip")
        self.detector = ObjectDetector(model_name="yolov8m.pt")
        self.matcher = SimilarityMatcher(
            self.feature_extractor,
            similarity_threshold=0.5
        )
        self.tracker = FeatureBasedTracker()
        
        print("✓ Demo initialized successfully!")
    
    def load_video_sample(self, sample_dir: str):
        """
        Load a video sample
        
        Args:
            sample_dir: Path to sample directory (e.g., train/samples/Backpack_0)
        """
        sample_path = Path(sample_dir)
        
        # Find video file
        video_files = list(sample_path.glob("*.mp4")) + list(sample_path.glob("*.avi"))
        if not video_files:
            raise ValueError(f"No video found in {sample_dir}")
        
        self.video_path = str(video_files[0])
        
        # Load reference images
        ref_dir = sample_path / "object_images"
        self.ref_images = load_reference_images(str(ref_dir))
        
        print(f"✓ Loaded video: {self.video_path}")
        print(f"✓ Loaded {len(self.ref_images)} reference images")
        
        # Extract reference features
        self.ref_features = self.feature_extractor.extract_reference_features(
            self.ref_images
        )
        print(f"✓ Extracted reference features: shape {self.ref_features.shape}")
    
    def show_reference_images(self):
        """Display reference images"""
        fig, axes = plt.subplots(1, len(self.ref_images), figsize=(12, 4))
        
        if len(self.ref_images) == 1:
            axes = [axes]
        
        for i, img_path in enumerate(self.ref_images):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"Reference {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_single_frame(self, frame_idx: int, visualize: bool = True):
        """
        Process a single frame and show results
        
        Args:
            frame_idx: Frame index to process
            visualize: Whether to show visualization
        """
        # Read frame
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Could not read frame {frame_idx}")
            return
        
        # Detect objects
        detections = self.detector.detect(frame)
        print(f"Found {len(detections)} objects")
        
        # Match with reference
        scored_detections = self.matcher.match_detections(
            frame, detections, self.ref_features
        )
        print(f"Matched {len(scored_detections)} objects")
        
        if visualize:
            self._visualize_frame_results(
                frame, detections, scored_detections, frame_idx
            )
        
        return frame, detections, scored_detections
    
    def _visualize_frame_results(self, frame, detections, scored_detections, frame_idx):
        """Visualize detection results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show all detections
        frame_all = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame_all, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame_all,
                f"{det.class_name}: {det.confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
        
        axes[0].imshow(cv2.cvtColor(frame_all, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"All Detections ({len(detections)})")
        axes[0].axis('off')
        
        # Show matched detections
        frame_matched = frame.copy()
        for scored_det in scored_detections:
            det = scored_det.detection
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame_matched, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_matched,
                f"Sim: {scored_det.similarity_score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        axes[1].imshow(cv2.cvtColor(frame_matched, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Matched Objects ({len(scored_detections)})")
        axes[1].axis('off')
        
        plt.suptitle(f"Frame {frame_idx}")
        plt.tight_layout()
        plt.show()
    
    def process_video_segment(self, start_frame: int, end_frame: int, step: int = 1):
        """
        Process a segment of the video
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index
            step: Frame step
        """
        cap = cv2.VideoCapture(self.video_path)
        
        results = []
        for frame_idx in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            detections = self.detector.detect(frame)
            scored_detections = self.matcher.match_detections(
                frame, detections, self.ref_features
            )
            
            results.append({
                'frame': frame_idx,
                'num_detections': len(detections),
                'num_matches': len(scored_detections),
                'matches': scored_detections
            })
            
            print(f"Frame {frame_idx}: {len(detections)} detections, {len(scored_detections)} matches")
        
        cap.release()
        
        return results
    
    def compare_with_ground_truth(self, annotations_path: str, video_id: str):
        """
        Compare predictions with ground truth
        
        Args:
            annotations_path: Path to annotations JSON
            video_id: Video ID to compare
        """
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Find ground truth for this video
        gt = next((item for item in annotations if item['video_id'] == video_id), None)
        
        if not gt:
            print(f"No ground truth found for {video_id}")
            return
        
        # Get GT frames
        gt_frames = set()
        for seq in gt['annotations']:
            for bbox in seq['bboxes']:
                gt_frames.add(bbox['frame'])
        
        print(f"Ground truth has detections in {len(gt_frames)} frames")
        print(f"Frame range: {min(gt_frames)} - {max(gt_frames)}")
        
        # Show sample GT frames
        sample_frames = sorted(list(gt_frames))[:5]
        print(f"\nSample GT frames: {sample_frames}")


def main():
    """Main demo function"""
    print("="*60)
    print("DRONE OBJECT DETECTION - INTERACTIVE DEMO")
    print("="*60)
    
    # Create demo instance
    demo = InteractiveDemo()
    
    # Example usage
    print("\n" + "="*60)
    print("EXAMPLE: Loading a sample video")
    print("="*60)
    
    # Load a sample (update path as needed)
    sample_dir = "./train/samples/Backpack_0"
    
    if Path(sample_dir).exists():
        demo.load_video_sample(sample_dir)
        
        print("\n" + "="*60)
        print("Showing reference images...")
        print("="*60)
        demo.show_reference_images()
        
        print("\n" + "="*60)
        print("Processing a sample frame...")
        print("="*60)
        demo.process_single_frame(frame_idx=370)
        
        print("\n" + "="*60)
        print("Processing a video segment...")
        print("="*60)
        results = demo.process_video_segment(
            start_frame=365,
            end_frame=380,
            step=1
        )
    else:
        print(f"\nSample directory not found: {sample_dir}")
        print("Please update the path in the script")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    main()

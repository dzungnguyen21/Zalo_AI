"""
Visualization Utilities for Drone Object Detection
Visualize predictions, ground truth, and evaluation results
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bbox(frame: np.ndarray,
             bbox: Dict,
             color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2,
             label: Optional[str] = None) -> np.ndarray:
    """
    Draw bounding box on frame
    
    Args:
        frame: Image as numpy array
        bbox: Dict with 'x1', 'y1', 'x2', 'y2'
        color: BGR color tuple
        thickness: Line thickness
        label: Optional label text
        
    Returns:
        Frame with drawn bbox
    """
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return frame


def visualize_video_predictions(video_path: str,
                                predictions: Dict,
                                ground_truth: Optional[Dict] = None,
                                output_path: Optional[str] = None,
                                show_frame_number: bool = True):
    """
    Visualize predictions on video
    
    Args:
        video_path: Path to video file
        predictions: Predictions dict for this video
        ground_truth: Optional ground truth dict
        output_path: Optional path to save output video
        show_frame_number: Show frame numbers on video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Build frame-to-bbox mappings
    pred_by_frame = {}
    for seq in predictions.get('detections', []):
        for bbox in seq['bboxes']:
            pred_by_frame[bbox['frame']] = bbox
    
    gt_by_frame = {}
    if ground_truth:
        for seq in ground_truth.get('annotations', []):
            for bbox in seq['bboxes']:
                gt_by_frame[bbox['frame']] = bbox
    
    frame_idx = 0
    
    print("Press 'q' to quit, 'space' to pause")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw ground truth (green)
        if frame_idx in gt_by_frame:
            frame = draw_bbox(
                frame,
                gt_by_frame[frame_idx],
                color=(0, 255, 0),
                label="GT"
            )
        
        # Draw prediction (blue)
        if frame_idx in pred_by_frame:
            frame = draw_bbox(
                frame,
                pred_by_frame[frame_idx],
                color=(255, 0, 0),
                label="Pred"
            )
        
        # Draw frame number
        if show_frame_number:
            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
        
        # Write frame if output video is specified
        if writer:
            writer.write(frame)
        
        # Display frame
        cv2.imshow('Video Predictions', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)  # Pause
        
        frame_idx += 1
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_idx} frames")


def plot_stiou_distribution(results: Dict, save_path: Optional[str] = None):
    """
    Plot STIoU score distribution across videos
    
    Args:
        results: Results dictionary from evaluate.py
        save_path: Optional path to save plot
    """
    scores = list(results['per_video_scores'].values())
    video_ids = list(results['per_video_scores'].keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(results['final_score'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {results['final_score']:.3f}")
    ax1.set_xlabel('STIoU Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('STIoU Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar plot
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_videos = [video_ids[i] for i in sorted_indices]
    
    colors = ['green' if s > 0.5 else 'orange' if s > 0.3 else 'red' for s in sorted_scores]
    
    ax2.barh(range(len(sorted_scores)), sorted_scores, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(sorted_videos)))
    ax2.set_yticklabels(sorted_videos, fontsize=8)
    ax2.set_xlabel('STIoU Score')
    ax2.set_title('Per-Video STIoU Scores')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def create_comparison_grid(video_path: str,
                          predictions: Dict,
                          ground_truth: Dict,
                          output_path: str,
                          num_samples: int = 6):
    """
    Create a grid showing sample frames with GT and predictions
    
    Args:
        video_path: Path to video file
        predictions: Predictions dict
        ground_truth: Ground truth dict
        output_path: Path to save grid image
        num_samples: Number of sample frames to show
    """
    # Get frames with both GT and predictions
    pred_by_frame = {}
    for seq in predictions.get('detections', []):
        for bbox in seq['bboxes']:
            pred_by_frame[bbox['frame']] = bbox
    
    gt_by_frame = {}
    for seq in ground_truth.get('annotations', []):
        for bbox in seq['bboxes']:
            gt_by_frame[bbox['frame']] = bbox
    
    # Find common frames
    common_frames = sorted(set(pred_by_frame.keys()) & set(gt_by_frame.keys()))
    
    if not common_frames:
        print("No common frames found")
        return
    
    # Sample frames
    if len(common_frames) > num_samples:
        step = len(common_frames) // num_samples
        sampled_frames = [common_frames[i * step] for i in range(num_samples)]
    else:
        sampled_frames = common_frames
    
    # Create grid
    cap = cv2.VideoCapture(video_path)
    
    fig, axes = plt.subplots(2, len(sampled_frames), figsize=(4*len(sampled_frames), 8))
    if len(sampled_frames) == 1:
        axes = axes.reshape(2, 1)
    
    for idx, frame_num in enumerate(sampled_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Show GT
        ax = axes[0, idx]
        ax.imshow(frame_rgb)
        gt_bbox = gt_by_frame[frame_num]
        rect = patches.Rectangle(
            (gt_bbox['x1'], gt_bbox['y1']),
            gt_bbox['x2'] - gt_bbox['x1'],
            gt_bbox['y2'] - gt_bbox['y1'],
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title(f'GT - Frame {frame_num}')
        ax.axis('off')
        
        # Show Prediction
        ax = axes[1, idx]
        ax.imshow(frame_rgb)
        pred_bbox = pred_by_frame[frame_num]
        rect = patches.Rectangle(
            (pred_bbox['x1'], pred_bbox['y1']),
            pred_bbox['x2'] - pred_bbox['x1'],
            pred_bbox['y2'] - pred_bbox['y1'],
            linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax.add_patch(rect)
        ax.set_title(f'Pred - Frame {frame_num}')
        ax.axis('off')
    
    cap.release()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison grid saved to: {output_path}")
    plt.close()


def main():
    """Main function for visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Predictions")
    parser.add_argument("--mode", type=str, required=True,
                       choices=['video', 'plot', 'grid'],
                       help="Visualization mode")
    parser.add_argument("--video_path", type=str,
                       help="Path to video file")
    parser.add_argument("--predictions", type=str,
                       help="Path to predictions JSON")
    parser.add_argument("--ground_truth", type=str,
                       help="Path to ground truth JSON")
    parser.add_argument("--video_id", type=str,
                       help="Video ID to visualize")
    parser.add_argument("--output", type=str,
                       help="Output path")
    parser.add_argument("--eval_results", type=str,
                       help="Path to evaluation results JSON")
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        # Load predictions
        with open(args.predictions, 'r') as f:
            all_preds = json.load(f)
        
        pred = next((p for p in all_preds if p['video_id'] == args.video_id), None)
        if not pred:
            print(f"Video ID {args.video_id} not found in predictions")
            return
        
        # Load ground truth if provided
        gt = None
        if args.ground_truth:
            with open(args.ground_truth, 'r') as f:
                all_gt = json.load(f)
            gt = next((g for g in all_gt if g['video_id'] == args.video_id), None)
        
        visualize_video_predictions(args.video_path, pred, gt, args.output)
    
    elif args.mode == 'plot':
        with open(args.eval_results, 'r') as f:
            results = json.load(f)
        
        if 'evaluation' in results:
            results = results['evaluation']
        
        plot_stiou_distribution(results, args.output)
    
    elif args.mode == 'grid':
        with open(args.predictions, 'r') as f:
            all_preds = json.load(f)
        with open(args.ground_truth, 'r') as f:
            all_gt = json.load(f)
        
        pred = next((p for p in all_preds if p['video_id'] == args.video_id), None)
        gt = next((g for g in all_gt if g['video_id'] == args.video_id), None)
        
        if not pred or not gt:
            print("Video ID not found")
            return
        
        create_comparison_grid(args.video_path, pred, gt, args.output)


if __name__ == "__main__":
    main()

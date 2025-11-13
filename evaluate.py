"""
Evaluation Module for Drone Object Detection Challenge
Implements Spatio-Temporal IoU (STIoU) metric
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict


def compute_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Compute IoU between two bounding boxes
    
    Args:
        bbox1, bbox2: Dicts with keys 'x1', 'y1', 'x2', 'y2'
        
    Returns:
        IoU score
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2']
    x1_2, y1_2, x2_2, y2_2 = bbox2['x1'], bbox2['y1'], bbox2['x2'], bbox2['y2']
    
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


def compute_stiou(ground_truth_bboxes: List[Dict], 
                  predicted_bboxes: List[Dict]) -> float:
    """
    Compute Spatio-Temporal IoU (STIoU) for a single video
    
    STIoU = sum(IoU at overlapping frames) / total frames in union
    
    Args:
        ground_truth_bboxes: List of ground truth bboxes with 'frame', 'x1', 'y1', 'x2', 'y2'
        predicted_bboxes: List of predicted bboxes with 'frame', 'x1', 'y1', 'x2', 'y2'
        
    Returns:
        STIoU score (0.0 to 1.0)
    """
    # Build frame-to-bbox mappings
    gt_by_frame = {bbox['frame']: bbox for bbox in ground_truth_bboxes}
    pred_by_frame = {bbox['frame']: bbox for bbox in predicted_bboxes}
    
    # Get all frames that have either GT or prediction
    all_frames = set(gt_by_frame.keys()) | set(pred_by_frame.keys())
    
    if len(all_frames) == 0:
        return 0.0
    
    # Compute IoU sum over intersection frames
    iou_sum = 0.0
    for frame in all_frames:
        if frame in gt_by_frame and frame in pred_by_frame:
            # Frame exists in both GT and prediction
            iou = compute_iou(gt_by_frame[frame], pred_by_frame[frame])
            iou_sum += iou
        # else: frame only in GT or only in prediction -> IoU = 0
    
    # STIoU = average IoU over all union frames
    stiou = iou_sum / len(all_frames)
    
    return stiou


def compute_stiou_for_sequences(ground_truth_sequences: List[Dict],
                                predicted_sequences: List[Dict]) -> float:
    """
    Compute STIoU when there are multiple detection sequences
    Uses best matching between GT and predicted sequences
    
    Args:
        ground_truth_sequences: List of sequences, each with 'bboxes' list
        predicted_sequences: List of sequences, each with 'bboxes' list
        
    Returns:
        Best STIoU score
    """
    if not ground_truth_sequences and not predicted_sequences:
        return 1.0  # Both empty = perfect match
    
    if not ground_truth_sequences or not predicted_sequences:
        return 0.0  # One is empty, other is not
    
    # Flatten all GT bboxes
    all_gt_bboxes = []
    for seq in ground_truth_sequences:
        all_gt_bboxes.extend(seq['bboxes'])
    
    # Try each predicted sequence and take the best
    best_stiou = 0.0
    for pred_seq in predicted_sequences:
        pred_bboxes = pred_seq['bboxes']
        stiou = compute_stiou(all_gt_bboxes, pred_bboxes)
        best_stiou = max(best_stiou, stiou)
    
    return best_stiou


def evaluate_predictions(ground_truth_path: str, 
                        predictions_path: str) -> Dict:
    """
    Evaluate predictions against ground truth
    
    Args:
        ground_truth_path: Path to ground truth JSON file
        predictions_path: Path to predictions JSON file
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Build lookup dictionaries
    gt_by_video = {item['video_id']: item for item in ground_truth}
    pred_by_video = {item['video_id']: item for item in predictions}
    
    # Get all video IDs
    all_video_ids = set(gt_by_video.keys()) | set(pred_by_video.keys())
    
    # Compute STIoU for each video
    stiou_scores = {}
    for video_id in all_video_ids:
        gt_item = gt_by_video.get(video_id, {'annotations': []})
        pred_item = pred_by_video.get(video_id, {'detections': []})
        
        gt_sequences = gt_item.get('annotations', [])
        pred_sequences = pred_item.get('detections', [])
        
        stiou = compute_stiou_for_sequences(gt_sequences, pred_sequences)
        stiou_scores[video_id] = stiou
    
    # Compute final score (average over all videos)
    final_score = np.mean(list(stiou_scores.values())) if stiou_scores else 0.0
    
    return {
        'final_score': final_score,
        'per_video_scores': stiou_scores,
        'num_videos': len(all_video_ids),
        'metrics': {
            'mean_stiou': final_score,
            'std_stiou': np.std(list(stiou_scores.values())),
            'min_stiou': np.min(list(stiou_scores.values())),
            'max_stiou': np.max(list(stiou_scores.values()))
        }
    }


def print_evaluation_report(results: Dict):
    """
    Print a formatted evaluation report
    
    Args:
        results: Results dictionary from evaluate_predictions
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"\nNumber of videos: {results['num_videos']}")
    print(f"\nFinal Score (Mean STIoU): {results['final_score']:.4f}")
    print(f"Standard Deviation: {results['metrics']['std_stiou']:.4f}")
    print(f"Min STIoU: {results['metrics']['min_stiou']:.4f}")
    print(f"Max STIoU: {results['metrics']['max_stiou']:.4f}")
    
    print("\nPer-Video Scores:")
    print("-" * 60)
    
    # Sort by score
    sorted_scores = sorted(results['per_video_scores'].items(), 
                          key=lambda x: x[1], 
                          reverse=True)
    
    for video_id, score in sorted_scores:
        print(f"{video_id:30s}: {score:.4f}")
    
    print("="*60)


def analyze_predictions(predictions_path: str) -> Dict:
    """
    Analyze prediction statistics
    
    Args:
        predictions_path: Path to predictions JSON file
        
    Returns:
        Dictionary with statistics
    """
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    stats = {
        'total_videos': len(predictions),
        'videos_with_detections': 0,
        'videos_without_detections': 0,
        'total_sequences': 0,
        'total_frames_detected': 0,
        'avg_frames_per_video': 0.0,
        'avg_sequences_per_video': 0.0
    }
    
    frame_counts = []
    sequence_counts = []
    
    for pred in predictions:
        detections = pred.get('detections', [])
        num_sequences = len(detections)
        
        sequence_counts.append(num_sequences)
        stats['total_sequences'] += num_sequences
        
        if num_sequences > 0:
            stats['videos_with_detections'] += 1
        else:
            stats['videos_without_detections'] += 1
        
        # Count frames
        num_frames = sum(len(seq['bboxes']) for seq in detections)
        frame_counts.append(num_frames)
        stats['total_frames_detected'] += num_frames
    
    if stats['total_videos'] > 0:
        stats['avg_frames_per_video'] = np.mean(frame_counts)
        stats['avg_sequences_per_video'] = np.mean(sequence_counts)
    
    return stats


def print_prediction_analysis(stats: Dict):
    """
    Print prediction statistics
    
    Args:
        stats: Statistics dictionary from analyze_predictions
    """
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    print(f"\nTotal videos: {stats['total_videos']}")
    print(f"Videos with detections: {stats['videos_with_detections']}")
    print(f"Videos without detections: {stats['videos_without_detections']}")
    print(f"\nTotal detection sequences: {stats['total_sequences']}")
    print(f"Total frames with detections: {stats['total_frames_detected']}")
    print(f"\nAverage sequences per video: {stats['avg_sequences_per_video']:.2f}")
    print(f"Average frames per video: {stats['avg_frames_per_video']:.2f}")
    print("="*60)


def main():
    """Main function for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Drone Object Detection Predictions")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Path to ground truth annotations JSON")
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save detailed results JSON")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze predictions without evaluation")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Just analyze predictions
        stats = analyze_predictions(args.predictions)
        print_prediction_analysis(stats)
    else:
        # Evaluate predictions
        results = evaluate_predictions(args.ground_truth, args.predictions)
        print_evaluation_report(results)
        
        # Also print prediction stats
        stats = analyze_predictions(args.predictions)
        print_prediction_analysis(stats)
        
        # Save detailed results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'evaluation': results,
                    'statistics': stats
                }, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()

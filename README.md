# Drone Object Detection Challenge

AI solution for detecting and tracking target objects in drone-captured videos for search-and-rescue missions.

## 🎯 Overview

This project implements an end-to-end pipeline for spatio-temporal object localization in drone videos. Given 3 reference images of a target object and a drone video, the system predicts when and where the object appears across video frames.

## 🏗️ Architecture

The solution combines multiple state-of-the-art techniques:

1. **Feature Extraction**: CLIP and DINOv2 for robust visual representations
2. **Object Detection**: YOLOv8 for generating bounding box candidates
3. **Similarity Matching**: Cosine similarity between reference and detected objects
4. **Temporal Tracking**: SORT-based tracking with feature fusion for temporal consistency
5. **Post-processing**: Temporal smoothing and interpolation

## 📁 Project Structure

```
Zalo_AI/
├── feature_extract.py       # Feature extraction using CLIP/DINOv2
├── object_detector.py        # YOLOv8-based object detection
├── similarity_matcher.py     # Cosine similarity matching
├── tracker.py                # SORT and feature-based tracking
├── inference_pipeline.py     # Main inference pipeline
├── evaluate.py               # STIoU evaluation metrics
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── train/                    # Training dataset
│   ├── samples/              # Video samples with reference images
│   └── annotations/          # Ground truth annotations
└── public_test/              # Public test dataset
    └── samples/
```

## 🚀 Installation

1. **Clone the repository** (if applicable):
```bash
git clone <repository_url>
cd Zalo_AI
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Note: CLIP will be installed from GitHub. If you encounter issues, you may need to install it manually:
```bash
pip install git+https://github.com/openai/CLIP.git
```

## 💻 Usage

### Basic Inference

Run the pipeline on a dataset:

```bash
python inference_pipeline.py --dataset_dir ./train --output predictions.json
```

### Advanced Options

```bash
python inference_pipeline.py \
    --dataset_dir ./train \
    --output predictions.json \
    --feature_model clip \
    --detector_model yolov8x.pt \
    --similarity_threshold 0.5 \
    --detection_conf 0.25 \
    --frame_skip 1 \
    --device cuda
```

**Parameters:**
- `--dataset_dir`: Path to dataset directory containing samples/
- `--output`: Output JSON file path for predictions
- `--feature_model`: Feature extraction model (clip, dinov2, or both)
- `--detector_model`: YOLO model (yolov8n/s/m/l/x.pt)
- `--similarity_threshold`: Minimum similarity score (0.0-1.0)
- `--detection_conf`: Detection confidence threshold (0.0-1.0)
- `--frame_skip`: Process every N-th frame (1=all frames)
- `--no_temporal`: Disable temporal similarity matching
- `--no_tracking`: Disable object tracking
- `--device`: Device to use (cuda or cpu)

### Evaluation

Evaluate predictions against ground truth:

```bash
python evaluate.py \
    --ground_truth ./train/annotations/annotations.json \
    --predictions predictions.json \
    --output evaluation_results.json
```

Analyze predictions only (without ground truth):

```bash
python evaluate.py \
    --predictions predictions.json \
    --analyze_only
```

## 🔧 Module Details

### 1. Feature Extraction (`feature_extract.py`)

Extracts visual features using:
- **CLIP**: Vision Transformer pre-trained with language
- **DINOv2**: Self-supervised vision transformer
- Supports single or combined feature extraction

```python
from feature_extract import FeatureExtractor

extractor = FeatureExtractor(model_name="clip")
ref_features = extractor.extract_reference_features(["ref1.jpg", "ref2.jpg", "ref3.jpg"])
```

### 2. Object Detection (`object_detector.py`)

YOLOv8-based detection with:
- Configurable confidence and IoU thresholds
- Batch processing support
- Filtering by class and size

```python
from object_detector import ObjectDetector

detector = ObjectDetector(model_name="yolov8x.pt")
detections = detector.detect(frame, conf_threshold=0.3)
```

### 3. Similarity Matching (`similarity_matcher.py`)

Matches detections with reference features:
- Cosine similarity scoring
- Temporal consistency boosting
- Multi-scale matching

```python
from similarity_matcher import SimilarityMatcher

matcher = SimilarityMatcher(extractor, similarity_threshold=0.5)
matched = matcher.match_detections(frame, detections, ref_features)
```

### 4. Tracking (`tracker.py`)

SORT-based tracking enhanced with visual features:
- Kalman filter for motion prediction
- Hungarian algorithm for data association
- Feature similarity for re-identification

```python
from tracker import FeatureBasedTracker

tracker = FeatureBasedTracker(max_age=30, min_hits=3)
tracks = tracker.update_with_features(scored_detections)
```

### 5. Inference Pipeline (`inference_pipeline.py`)

End-to-end pipeline combining all modules:
- Video processing with progress tracking
- Automatic detection grouping
- JSON output in competition format

## 📊 Evaluation Metric

The solution is evaluated using **Spatio-Temporal IoU (STIoU)**:

$$
\text{STIoU} = \frac{\sum_{f \in \text{intersection}} \text{IoU}(B_f, B'_f)}{\sum_{f \in \text{union}} 1}
$$

Final score is the average STIoU across all videos:

$$
\text{Final Score} = \frac{1}{N} \sum_{i=1}^{N} \text{STIoU}_{\text{video}_i}
$$

## 🎛️ Hyperparameter Tuning

Key hyperparameters to tune:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `similarity_threshold` | 0.3-0.7 | Minimum similarity to accept detection |
| `detection_conf` | 0.15-0.4 | YOLO confidence threshold |
| `frame_skip` | 1-5 | Process every N-th frame |
| `max_age` | 20-50 | Max frames to keep track alive |
| `min_hits` | 2-5 | Min detections to confirm track |
| `iou_threshold` | 0.2-0.5 | IoU threshold for tracking |

## 🔍 Tips for Better Performance

1. **Feature Model Selection**:
   - Use `clip` for fast processing
   - Use `dinov2` for better accuracy
   - Use `both` for best results (slower)

2. **Detection Optimization**:
   - Lower `detection_conf` to catch more candidates
   - Increase `similarity_threshold` to reduce false positives

3. **Speed vs Accuracy**:
   - Use `yolov8n` or `yolov8s` for faster inference
   - Use `yolov8x` for best detection quality
   - Increase `frame_skip` to process fewer frames

4. **Tracking**:
   - Enable tracking for smoother temporal consistency
   - Adjust `max_age` based on video characteristics

## 📝 Output Format

Predictions are saved in JSON format:

```json
[
  {
    "video_id": "drone_video_001",
    "detections": [
      {
        "bboxes": [
          {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
          {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
        ]
      }
    ]
  },
  {
    "video_id": "drone_video_002",
    "detections": []
  }
]
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Use smaller YOLO model (`yolov8n` or `yolov8s`)
- Increase `frame_skip` to process fewer frames
- Use `--device cpu` if GPU memory is limited

### CLIP Installation Issues
```bash
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

### Slow Processing
- Use `frame_skip > 1` to process every N-th frame
- Use lighter models (`yolov8n`, `clip` only)
- Disable temporal matching with `--no_temporal`

## 📚 References

- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193)
- [YOLOv8: Ultralytics](https://github.com/ultralytics/ultralytics)
- [SORT: Simple Online Realtime Tracking](https://arxiv.org/abs/1602.00763)

## 📄 License

This project is for educational and competition purposes.

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

---

**Good luck with the challenge! 🚁🔍**
#   Z a l o _ A I  
 
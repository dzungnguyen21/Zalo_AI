# Quick Start Guide - Drone Object Detection Challenge

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check if everything is installed correctly
python run.py check
```

### 3. Run Quick Test

```bash
# Test the pipeline with fast settings
python run.py test --dataset_dir ./train
```

This will process the training dataset with optimized settings for speed.

---

## 📊 Common Use Cases

### Use Case 1: Generate Predictions for Submission

```bash
# Run inference with best settings
python run.py infer \
    --dataset_dir ./public_test \
    --output submission.json \
    --model clip \
    --detector yolov8x.pt \
    --threshold 0.5
```

### Use Case 2: Evaluate on Training Data

```bash
# Run inference on training data
python run.py infer --dataset_dir ./train --output train_predictions.json

# Evaluate predictions
python run.py eval \
    --ground_truth ./train/annotations/annotations.json \
    --predictions train_predictions.json \
    --output eval_results.json
```

### Use Case 3: Experiment with Different Settings

```bash
# Try different similarity thresholds
python batch_process.py --mode similarity --dataset_dir ./train

# Compare different models
python batch_process.py --mode models --dataset_dir ./train

# Run all presets (fast, balanced, accurate)
python batch_process.py --mode presets --dataset_dir ./train
```

### Use Case 4: Visualize Results

```bash
# Visualize predictions on a video
python visualize.py --mode video \
    --video_path ./train/samples/Backpack_0/drone_video.mp4 \
    --predictions predictions.json \
    --video_id Backpack_0 \
    --output visualization.mp4

# Plot STIoU distribution
python visualize.py --mode plot \
    --eval_results eval_results.json \
    --output stiou_plot.png

# Create comparison grid
python visualize.py --mode grid \
    --video_path ./train/samples/Backpack_0/drone_video.mp4 \
    --predictions predictions.json \
    --ground_truth ./train/annotations/annotations.json \
    --video_id Backpack_0 \
    --output comparison_grid.png
```

---

## ⚙️ Configuration Presets

### Fast Mode (for testing)
```bash
python run.py infer \
    --dataset_dir ./train \
    --model clip \
    --detector yolov8n.pt \
    --skip 5 \
    --no_temporal
```
- **Speed**: ⚡⚡⚡⚡⚡
- **Accuracy**: ⭐⭐⭐
- **Use when**: Quick testing, limited resources

### Balanced Mode (recommended)
```bash
python run.py infer \
    --dataset_dir ./train \
    --model clip \
    --detector yolov8l.pt \
    --skip 2 \
    --threshold 0.5
```
- **Speed**: ⚡⚡⚡
- **Accuracy**: ⭐⭐⭐⭐
- **Use when**: General purpose, good balance

### Accurate Mode (for final submission)
```bash
python run.py infer \
    --dataset_dir ./public_test \
    --model both \
    --detector yolov8x.pt \
    --skip 1 \
    --threshold 0.45 \
    --conf 0.15
```
- **Speed**: ⚡
- **Accuracy**: ⭐⭐⭐⭐⭐
- **Use when**: Final submission, best results needed

---

## 🎯 Parameter Tuning Guide

### Key Parameters and Their Effects

| Parameter | Range | Effect on Speed | Effect on Accuracy | Recommendation |
|-----------|-------|----------------|-------------------|----------------|
| `--model` | clip/dinov2/both | both=slowest | both=best | Start with 'clip' |
| `--detector` | n/s/m/l/x | x=slowest | x=best | Use 'x' for final |
| `--threshold` | 0.3-0.7 | minimal | critical | Try 0.4-0.6 |
| `--conf` | 0.1-0.4 | minimal | important | Start with 0.25 |
| `--skip` | 1-10 | linear | inverse | 1-2 for accuracy |

### Tuning Strategy

1. **Start with fast settings** to iterate quickly
2. **Adjust similarity threshold** (0.4-0.6) based on false positive rate
3. **Lower detection confidence** (0.15-0.25) if missing detections
4. **Switch to better models** when ready for final run
5. **Reduce frame skip** to 1 for maximum accuracy

---

## 🔧 Troubleshooting

### Out of Memory Error
```bash
# Use smaller model
python run.py infer --dataset_dir ./train --detector yolov8n.pt --skip 3

# Or use CPU
python run.py infer --dataset_dir ./train --cpu
```

### Too Many False Positives
```bash
# Increase similarity threshold
python run.py infer --dataset_dir ./train --threshold 0.6

# Increase detection confidence
python run.py infer --dataset_dir ./train --conf 0.3
```

### Missing Detections
```bash
# Lower similarity threshold
python run.py infer --dataset_dir ./train --threshold 0.4

# Lower detection confidence
python run.py infer --dataset_dir ./train --conf 0.15

# Process more frames
python run.py infer --dataset_dir ./train --skip 1
```

### Slow Processing
```bash
# Use lighter models
python run.py infer --dataset_dir ./train --model clip --detector yolov8n.pt

# Skip more frames
python run.py infer --dataset_dir ./train --skip 5

# Disable temporal matching
python run.py infer --dataset_dir ./train --no_temporal
```

---

## 📁 Expected Directory Structure

```
Zalo_AI/
├── train/
│   ├── samples/
│   │   ├── Backpack_0/
│   │   │   ├── object_images/
│   │   │   │   ├── img_1.jpg
│   │   │   │   ├── img_2.jpg
│   │   │   │   └── img_3.jpg
│   │   │   └── drone_video.mp4
│   │   └── ...
│   └── annotations/
│       └── annotations.json
└── public_test/
    └── samples/
        └── ...
```

---

## 💡 Pro Tips

1. **Start with a subset**: Test on 1-2 videos first
2. **Use visualization**: Always visualize results to understand failures
3. **Monitor GPU usage**: Use `nvidia-smi` to check GPU utilization
4. **Save intermediate results**: Don't reprocess from scratch
5. **Try ensemble**: Combine predictions from different models
6. **Temporal consistency**: Enable tracking for smoother results

---

## 🎓 Learning Resources

- **CLIP Paper**: https://arxiv.org/abs/2103.00020
- **DINOv2 Paper**: https://arxiv.org/abs/2304.07193
- **YOLOv8 Docs**: https://docs.ultralytics.com
- **SORT Paper**: https://arxiv.org/abs/1602.00763

---

## 📞 Getting Help

If you encounter issues:

1. Check the logs in `experiments/*/stderr.log`
2. Verify data format matches expected structure
3. Try with minimal settings first
4. Check GPU memory with `nvidia-smi`

---

**Good luck! 🎯**

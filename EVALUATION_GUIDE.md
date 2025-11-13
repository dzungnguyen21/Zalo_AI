# Evaluation and Improvement Guide

## 📊 Your Current Results Summary

**Quick Test Results:**
- Mean STIoU: **0.003** (Very Low - needs improvement)
- Videos with detections: 7/14 (50%)
- Best performing: Person1_0 (0.0275)

### Why Low Scores?

The quick test used **fast settings** optimized for speed, not accuracy:
- ❌ YOLOv8n (smallest, least accurate model)
- ❌ frame_skip=5 (only 20% of frames processed)
- ❌ Low detection confidence (misses many objects)

---

## 🎯 How to Improve Results

### **Method 1: Use Better Configuration Presets**

#### A. Balanced Mode (Recommended)
Good balance between speed and accuracy:

```bash
python run.py infer \
    --dataset_dir ./train \
    --output predictions_balanced.json \
    --model clip \
    --detector yolov8l.pt \
    --threshold 0.45 \
    --conf 0.2 \
    --skip 2
```

**Expected improvement:** 0.003 → **0.3-0.5** STIoU

#### B. Accurate Mode (Best Quality)
For final submission:

```bash
python run.py infer \
    --dataset_dir ./train \
    --output predictions_accurate.json \
    --model both \
    --detector yolov8x.pt \
    --threshold 0.4 \
    --conf 0.15 \
    --skip 1
```

**Expected improvement:** 0.003 → **0.5-0.7** STIoU

---

### **Method 2: Tune Hyperparameters**

Run experiments to find best settings:

```bash
# Test different similarity thresholds
python batch_process.py \
    --mode similarity \
    --dataset_dir ./train

# Test different model combinations
python batch_process.py \
    --mode models \
    --dataset_dir ./train

# Run all presets
python batch_process.py \
    --mode presets \
    --dataset_dir ./train
```

After experiments, check results:
```bash
cat experiments/summary.json
```

---

### **Method 3: Analyze Per-Video Performance**

Some videos are harder than others. Let's see which ones need attention:

```bash
# Analyze predictions
python run.py analyze --predictions quick_test_predictions.json
```

#### Videos with 0.0 score need improvement:
- Backpack_0, Backpack_1
- Jacket_0, Jacket_1
- MobilePhone_0, MobilePhone_1
- WaterBottle_0
- Laptop_0, Laptop_1
- Lifering_0

**Possible issues:**
1. Object too small → Lower detection confidence
2. Object appearance changes → Use 'both' feature model
3. Object moves fast → Reduce frame_skip
4. Poor similarity match → Lower similarity threshold

---

## 🔍 Step-by-Step Improvement Process

### Step 1: Run Better Inference (Currently Running)

```bash
# This is running now
python run.py infer \
    --dataset_dir ./train \
    --output predictions_better.json \
    --model clip \
    --detector yolov8m.pt \
    --threshold 0.45 \
    --conf 0.2 \
    --skip 2
```

### Step 2: Evaluate New Predictions

```bash
python run.py eval \
    --ground_truth ./train/annotations/annotations.json \
    --predictions predictions_better.json \
    --output eval_better.json
```

### Step 3: Visualize Results

Pick a video to debug:

```bash
# Visualize Person1_0 (your best performing video)
python visualize.py \
    --mode video \
    --video_path ./train/samples/Person1_0/drone_video.mp4 \
    --predictions predictions_better.json \
    --ground_truth ./train/annotations/annotations.json \
    --video_id Person1_0
```

### Step 4: Compare Results

```bash
# Plot STIoU distribution
python visualize.py \
    --mode plot \
    --eval_results eval_better.json \
    --output stiou_plot.png
```

---

## 📈 Parameter Tuning Guide

### Key Parameters and Their Effects:

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `--detector` | Detection quality | yolov8m.pt to yolov8x.pt |
| `--threshold` | Similarity strictness | 0.3 to 0.6 |
| `--conf` | Detection confidence | 0.15 to 0.3 |
| `--skip` | Processing speed | 1 to 3 |
| `--model` | Feature quality | clip or both |

### Tuning Strategy:

**If many videos have 0.0 score:**
- ✅ Lower `--threshold` (try 0.3 or 0.35)
- ✅ Lower `--conf` (try 0.15)
- ✅ Reduce `--skip` (try 1)

**If scores are low but not 0:**
- ✅ Better detector (yolov8l or yolov8x)
- ✅ Better features (--model both)
- ✅ Fine-tune thresholds

**If false positives are high:**
- ✅ Increase `--threshold` (try 0.6 or 0.7)
- ✅ Increase `--conf` (try 0.3 or 0.4)

---

## 🎯 Expected STIoU Scores

| Configuration | Mean STIoU | Quality |
|---------------|-----------|---------|
| Quick Test (current) | 0.001-0.01 | ⭐ Poor |
| Fast Mode | 0.1-0.3 | ⭐⭐ Fair |
| Balanced Mode | 0.3-0.5 | ⭐⭐⭐ Good |
| Accurate Mode | 0.5-0.7 | ⭐⭐⭐⭐ Excellent |
| Tuned Configuration | 0.6-0.8 | ⭐⭐⭐⭐⭐ Best |

---

## 🚀 Quick Commands Reference

### Evaluate Existing Predictions
```bash
python run.py eval \
    --ground_truth ./train/annotations/annotations.json \
    --predictions <your_predictions>.json
```

### Analyze Predictions Only
```bash
python run.py analyze --predictions <your_predictions>.json
```

### Visualize Video
```bash
python visualize.py \
    --mode video \
    --video_path ./train/samples/<VIDEO_NAME>/drone_video.mp4 \
    --predictions <your_predictions>.json \
    --video_id <VIDEO_NAME>
```

### Interactive Exploration
```bash
python demo.py
# Then modify the script to load specific videos
```

---

## 💡 Tips for Best Results

1. **Start with balanced settings** - don't jump to the slowest immediately
2. **Analyze failures** - understand which videos fail and why
3. **Visualize results** - seeing is understanding
4. **Iterate** - test different parameter combinations
5. **Use batch processing** - test multiple configs automatically

---

## 🐛 Troubleshooting Low Scores

### Score = 0.0 for all videos?
- Detection confidence too high → Lower `--conf`
- Similarity threshold too high → Lower `--threshold`
- Wrong model → Try different detector

### Some detections but low IoU?
- Bounding boxes misaligned → Better detector (yolov8x)
- Object size mismatch → Check reference images
- Temporal issues → Enable tracking (default)

### Processing too slow?
- Increase `--skip` to 3 or 5
- Use smaller model (yolov8s or yolov8m)
- Use only CLIP (not 'both')

---

## ✅ Next Steps

1. **Wait for better inference to complete** (currently running)
2. **Evaluate the new predictions**
3. **Compare with quick test results**
4. **Visualize a few videos to understand performance**
5. **Run batch experiments to find optimal parameters**
6. **Generate final predictions for submission**

---

**Remember:** The quick test was just for speed. Real accuracy requires better settings!

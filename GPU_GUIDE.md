# Hướng dẫn Sử dụng GPU

## ✅ GPU đã được cài đặt thành công!

**Thông tin GPU của bạn:**
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- VRAM: 4GB
- CUDA Version: 12.3 (Driver)
- PyTorch CUDA: 11.8 (Compatible)

---

## 🚀 Chạy Inference với GPU

### Cách 1: Tự động (Mặc định sử dụng GPU)

Pipeline sẽ **tự động** sử dụng GPU nếu có:

```powershell
# GPU sẽ được dùng tự động (PowerShell - một dòng)
python run.py infer --dataset_dir ./train --output predictions_gpu.json --detector yolov8m.pt --threshold 0.45 --conf 0.2 --skip 2
```

### Cách 2: Chỉ định rõ ràng

```powershell
# Chỉ định dùng GPU
python run.py infer --dataset_dir ./train --output predictions_gpu.json --detector yolov8m.pt --device cuda
```

### Cách 3: Bắt buộc dùng CPU (nếu cần)

```powershell
# Bắt buộc dùng CPU
python run.py infer --dataset_dir ./train --output predictions_cpu.json --detector yolov8m.pt --device cpu
```

---

## ⚡ Tốc độ Cải thiện với GPU

**So sánh CPU vs GPU:**

| Cấu hình | CPU (Core i5/i7) | GPU (RTX 3050) | Tăng tốc |
|----------|------------------|----------------|----------|
| YOLOv8n + CLIP | ~5 fps | ~15 fps | **3x** |
| YOLOv8m + CLIP | ~2 fps | ~10 fps | **5x** |
| YOLOv8x + CLIP | ~0.5 fps | ~5 fps | **10x** |
| YOLOv8x + Both | ~0.3 fps | ~3 fps | **10x** |

---

## 🎯 Cấu hình Đề xuất cho GPU 4GB

### Cấu hình Tối ưu (Cân bằng tốc độ & chất lượng)

```powershell
python run.py infer --dataset_dir ./train --output predictions_optimal.json --model clip --detector yolov8l.pt --threshold 0.45 --conf 0.2 --skip 1
```

**Ưu điểm:**
- ✅ Tận dụng GPU hiệu quả
- ✅ Không bị Out of Memory
- ✅ Chất lượng tốt (~0.4-0.6 STIoU)
- ✅ Tốc độ: ~8-10 fps

### Cấu hình Chất lượng Cao (Chậm hơn nhưng chính xác)

```powershell
python run.py infer --dataset_dir ./train --output predictions_high_quality.json --model both --detector yolov8x.pt --threshold 0.4 --conf 0.15 --skip 1
```

**Lưu ý:**
- ⚠️ Có thể hết VRAM nếu video resolution cao
- ⚠️ Tốc độ: ~3-5 fps
- ✅ Chất lượng tốt nhất (~0.5-0.7 STIoU)

### Cấu hình Nhanh (Để test)

```powershell
python run.py infer --dataset_dir ./train --output predictions_fast.json --model clip --detector yolov8m.pt --threshold 0.5 --conf 0.25 --skip 2
```

**Ưu điểm:**
- ✅ Rất nhanh: ~10-15 fps
- ✅ Tiết kiệm VRAM
- ⚠️ Chất lượng trung bình (~0.3-0.5 STIoU)

---

## 💡 Tips Tối ưu GPU 4GB

### 1. Giảm Batch Size (nếu cần)
Pipeline mặc định xử lý từng frame, đã tối ưu cho 4GB VRAM.

### 2. Tránh Out of Memory

**Nếu gặp lỗi CUDA Out of Memory:**

```powershell
# Giảm xuống model nhỏ hơn
python run.py infer --dataset_dir ./train --output predictions.json --detector yolov8m.pt
```

Hoặc:

```powershell
# Dùng chỉ CLIP thay vì 'both'
python run.py infer --dataset_dir ./train --output predictions.json --model clip
```

### 3. Theo dõi VRAM Usage

Mở terminal mới và chạy:
```bash
# Xem GPU usage real-time
nvidia-smi -l 1
```

### 4. Giải phóng VRAM sau mỗi run

```bash
# Thoát Python sau mỗi lần chạy
# VRAM sẽ tự động giải phóng
```

---

## 🎮 Kiểm tra GPU đang hoạt động

Trong khi chạy inference, mở terminal mới:

```bash
# Xem GPU usage
nvidia-smi

# Hoặc xem liên tục
nvidia-smi -l 1
```

Bạn sẽ thấy:
- **GPU-Util**: ~80-100% khi đang xử lý
- **Memory-Usage**: Tăng lên ~2-3GB

---

## 📊 Ví dụ Chạy Thực tế

### Bước 1: Chạy inference với GPU

```bash
python run.py infer \
    --dataset_dir ./train \
    --output predictions_gpu.json \
    --model clip \
    --detector yolov8l.pt \
    --threshold 0.45 \
    --conf 0.2 \
    --skip 1
```

### Bước 2: Đánh giá kết quả

```bash
python run.py eval \
    --ground_truth ./train/annotations/annotations.json \
    --predictions predictions_gpu.json
```

### Bước 3: Visualize

```bash
python visualize.py \
    --mode video \
    --video_path ./train/samples/Person1_0/drone_video.mp4 \
    --predictions predictions_gpu.json \
    --video_id Person1_0
```

---

## 🔥 Lệnh Khuyên dùng cho RTX 3050 4GB

```bash
# Chạy ngay bây giờ với cấu hình tối ưu
python run.py infer \
    --dataset_dir ./train \
    --output predictions_gpu_optimal.json \
    --model clip \
    --detector yolov8l.pt \
    --threshold 0.45 \
    --conf 0.2 \
    --skip 1

# Sau đó evaluate
python run.py eval \
    --ground_truth ./train/annotations/annotations.json \
    --predictions predictions_gpu_optimal.json \
    --output eval_gpu.json
```

**Thời gian ước tính:** ~5-10 phút cho toàn bộ dataset (14 videos)

---

## ⚠️ Xử lý Lỗi

### Lỗi: CUDA Out of Memory
```bash
RuntimeError: CUDA out of memory
```

**Giải pháp:**
1. Dùng model nhỏ hơn: `yolov8m.pt` thay vì `yolov8x.pt`
2. Dùng chỉ CLIP: `--model clip` thay vì `--model both`
3. Tăng frame skip: `--skip 2` hoặc `--skip 3`
4. Đóng các ứng dụng khác đang dùng GPU

### Lỗi: GPU không được dùng
```bash
# Kiểm tra
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu False, cài lại PyTorch với CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📈 Kỳ vọng Kết quả

Với GPU và cấu hình tối ưu:

| Metric | Giá trị |
|--------|---------|
| Mean STIoU | 0.4 - 0.6 |
| Processing Time | 5-10 phút |
| GPU Utilization | 80-100% |
| VRAM Usage | 2-3 GB |

---

**Chúc bạn thành công! 🚀**

# 🤖 AI-Powered ID Card Detection

This folder contains scripts for automated ID card detection and segmentation using deep learning models.

## 📁 Contents

### `demo.py`

AI-powered ID card detection using YOLOv5 segmentation model with ONNX Runtime.

**Features:**
- 🎯 YOLOv5 instance segmentation for ID card detection
- 🔲 Automatic corner detection using morphological operations
- 📐 Perspective transformation and rectification
- 💻 ONNX Runtime backend (CPU/GPU support)
- 🎨 Non-maximum suppression (NMS) using torchvision

**Usage:**

```bash
# Run detection with YOLOv5 model
python demo.py -d yolov5x-seg-id-dr-pp-best.onnx -c yolov5

# Command-line arguments:
# -d, --detector: Path to ONNX model file
# -c, --config: Detector configuration (yolov5)
```

**How it Works:**

1. **Model Loading**: Loads YOLOv5-Seg ONNX model
2. **Inference**: Detects ID cards in video frames
3. **Segmentation**: Extracts precise ID card masks
4. **Corner Detection**: Finds quadrilateral corners from mask
5. **Perspective Correction**: Warps ID card to standard view
6. **Output**: Saves rectified ID card images

**Classes:**

| Class | Description |
|-------|-------------|
| `OrtBase` | Base class for ONNX Runtime inference |
| `IDCardDetectionBase` | Base detector with NMS and IoU utilities |
| `YoloV5` | YOLOv5-specific detection implementation |

**Dependencies:**
- `onnxruntime` - Model inference
- `torch` / `torchvision` - NMS operations
- `opencv-python` - Image processing
- `numpy` - Numerical operations

**Model Requirements:**

You'll need a YOLOv5-Seg model trained on ID cards (e.g., `yolov5x-seg-id-dr-pp-best.onnx`). The model should output:
- Bounding boxes (XYXY format)
- Confidence scores
- Segmentation masks

## 🔗 Related

Return to [main documentation](../README.md) for the complete processing pipeline.

# ID Card Video → Aligned Images & Quick Metrics

A small toolkit for turning handheld video of an ID card into aligned image stacks, generating max/mean/min composites, and computing simple color/edge/specular metrics.

> Scripts in this repo:
> - `main.py` — “batteries‑included” pipeline: extract high‑quality frames → optional edges → center crop → SIFT‑based global alignment → write max/mean/min and difference images.
> - `demo.py` — ONNXRuntime + YOLOv5‑Seg demo for ID‑card detection, segmentation‑mask corner extraction, perspective alignment, plus utilities for alignment and composites.
> - `check.py` — analysis helpers that compute color histogram stats, Sobel edge stats, a crude specularity index, and a simple contrast ratio over the specular region.

---

## Requirements

- Python 3.8+
- Packages
  - Core: `opencv-python`, **`opencv-contrib-python`** (SIFT), `numpy`
  - Demo (optional): `onnxruntime`, `torch`, `torchvision`
  - Metrics (optional): `scikit-image`
  - Some scripts import `dlib` (only for build compatibility); not required by the main pipeline steps shown below.

> **Note:** SIFT lives in `opencv-contrib-python`. If you only install `opencv-python`, SIFT will be missing and alignment will fail.

### Install

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install opencv-python opencv-contrib-python onnxruntime numpy torch torchvision scikit-image dlib
```

---

## Quick Start (End‑to‑End)

`main.py` uses simple, hardcoded paths. Put your input video next to the script (e.g., `print_video.mp4`), then run:

```bash
python main.py
```

### What it does

1. **Extract sharp frames** from the video into `./frames/` using variance‑of‑Laplacian with a default threshold of **100**.
2. (Optional) **Edge previews** into `./edge/` using Canny, just for visual inspection.
3. **Center crop** a horizontal band (default **height=400** px around the middle) into `./cropped_images/`.
4. **Align frames** with **SIFT + FLANN + Homography (RANSAC)** to the first frame; results in `./aligned_images/`.
5. **Composites**: writes `max_image.jpg`, `mean_image.jpg`, `min_image.jpg`, and pairwise differences (`max_min_image.jpg`, `max_mean_image.jpg`, `mean_min_image.jpg`) into `./output/`.

> If you want to tweak thresholds/paths, open `main.py` and adjust the arguments passed to the functions at the bottom of the file.

---

## Script Reference

### `main.py`
Functions:
- `save_high_quality_frames(video_path, quality_threshold=100, save_folder='frames')`  
  Extracts frames with variance‑of‑Laplacian above `quality_threshold`.
- `find_and_save_edges(input_folder, output_folder)`  
  Saves Canny edges (debug/inspection).
- `crop_center_and_save(input_folder, output_folder, height=400)`  
  Crops a central horizontal band from each frame.
- `align_images(input_folder, output_folder)`  
  SIFT feature matching + FLANN + `cv2.findHomography(..., RANSAC, 5.0)`; warps every image to the first.
- `generate_max_mean_min_images(input_folder, output_folder)`  
  Stacks aligned images (H×W×3×N) and saves max/mean/min plus differences.

Default pipeline (at bottom of the file):
```python
save_high_quality_frames('print_video.mp4', quality_threshold=100, save_folder='frames')
find_and_save_edges('./frames', './edge')
crop_center_and_save('frames', 'cropped_images')
align_images('cropped_images', 'aligned_images')
generate_max_mean_min_images('aligned_images', 'output')
```

---

### `demo.py` (Detection‑assisted workflow)
- **Backend:** ONNXRuntime session (CPU by default) for a YOLOv5‑Seg model (`yolov5x-seg-id-dr-pp-best.onnx` by default).
- **NMS:** uses `torchvision.ops.nms`.
- **Mask → quadrilateral:** post‑process best mask (morph → contours → `approxPolyDP`) and order corners by region centroid.
- **Perspective rectification:** warp to an ID card aspect (e.g., 86×54 for a class; 125×88 for others—see code).
- **Utilities:** reuses SIFT alignment & composite generation similar to `main.py`.

CLI:
```bash
python demo.py -d yolov5x-seg-id-dr-pp-best.onnx -c yolov5
```
Example `__main__` flow:
1) Extract sharp frames from `second.mp4` (hardcoded in the sample).  
2) Run detection & alignment in memory; replace frames with perspective‑rectified crops.  
3) (Optionally) run folder alignment + composites to `aligned_images/` and `output/`.

> **Saving detections:** The sample keeps aligned crops in memory. If you want files, extend the call to pass output paths when drawing/saving predictions, or write the returned crops to disk before running the SIFT alignment utilities.

---

### `check.py` (Image metrics)
For each input image it computes:
- **Color histogram metrics** per RGB channel: `mean`, `std`, `max hist count`, plus grayscale mean.
- **Sobel edge stats:** mean and std of the edge response.
- **Specularity index:** max grayscale intensity (crude highlight proxy).
- **Contrast ratio:** mean intensity in a high‑threshold (“specular”) mask divided by mean of the rest.

Usage:
```bash
python check.py
```
Edit the image paths at the bottom of the script to point to your own images (e.g., the composites you produced in `output/`).

---

## Tips & Troubleshooting

- **SIFT not found?** Install `opencv-contrib-python` (in addition to or instead of `opencv-python`).  
- **Alignment poor or failing?** Increase texture/contrast (or skip heavy crops), raise the feature match quality, or try increasing the RANSAC reprojection threshold slightly.  
- **Too few sharp frames?** Lower `quality_threshold` from **100** (e.g., 60–80) to keep more frames.  
- **Memory/size issues when stacking images?** Limit the input set or downscale frames before alignment/compositing.

---

## Roadmap

- ONNXRuntime CUDA / TensorRT providers for detection.
- Save aligned crops directly in the detection loop.
- CLI flags for all I/O paths and thresholds in `main.py`.
- Batch inference over folders.

---

## License

Add your preferred license text here.

## Contact

Questions or suggestions? Open an issue or message the maintainer.

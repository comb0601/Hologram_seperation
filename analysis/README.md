# 📊 Image Analysis & Metrics

This folder contains tools for analyzing security features in ID card images through quantitative metrics.

## 📁 Contents

### `check.py`

Comprehensive image analysis toolkit for measuring security features like holograms, microprinting, and reflective elements.

**Features:**
- 🎨 **Color Histogram Analysis** - RGB distribution metrics
- 🔍 **Edge Detection Metrics** - Sobel operator for microprinting
- ✨ **Specularity Index** - Reflective surface measurement
- 📈 **Contrast Ratio** - Highlight vs. base material analysis

**Usage:**

```python
from check import color_metrics, edge_metrics, specularity_index

# Load your composite image
image = cv2.imread('output/max_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Analyze color distribution
color_stats = color_metrics(image)
print(f"Red channel mean: {color_stats['red_mean']:.2f}")
print(f"Green channel std: {color_stats['green_std']:.2f}")

# Analyze edge features (microprinting)
edge_stats = edge_metrics(image)
print(f"Edge intensity: {edge_stats['edge_mean_intensity']:.4f}")

# Calculate specularity (hologram reflectivity)
spec_index = specularity_index(image)
print(f"Specularity index: {spec_index:.4f}")
```

**Functions:**

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `color_metrics(image)` | RGB image array | Dict of statistics | RGB channel distribution analysis |
| `edge_metrics(image)` | RGB image array | Dict of edge stats | Sobel edge detection metrics |
| `specularity_index(image)` | RGB image array | Float (0-1) | Reflective surface measurement |

**Metrics Explained:**

### Color Metrics
- `{channel}_mean`: Average intensity per color channel
- `{channel}_std`: Standard deviation (color variance)
- `{channel}_max`: Peak histogram frequency
- `gray_mean_intensity`: Overall brightness

### Edge Metrics
- `edge_mean_intensity`: Average edge strength
- `edge_std_intensity`: Edge variation
- `edge_max_intensity`: Strongest edge response

### Specularity Index
Higher values indicate more reflective/specular surfaces (typical of holograms and OVI inks)

**Example Analysis:**

```python
# Compare max vs. mean composite images
max_img = io.imread('output/max_image.jpg')
mean_img = io.imread('output/mean_image.jpg')

max_spec = specularity_index(max_img)
mean_spec = specularity_index(mean_img)

print(f"Max image specularity: {max_spec:.3f}")
print(f"Mean image specularity: {mean_spec:.3f}")
print(f"Hologram enhancement: {(max_spec/mean_spec - 1)*100:.1f}%")
```

**Dependencies:**
- `opencv-python` - Image I/O
- `scikit-image` - Color conversion, Sobel filter
- `numpy` - Numerical operations

## 🔬 Use Cases

- **Hologram Detection**: High specularity index indicates holographic features
- **Microprint Analysis**: Strong edge metrics suggest fine text/patterns
- **Quality Control**: Color variance shows OVI (Optically Variable Ink)
- **Counterfeit Detection**: Compare metrics against known authentic samples

## 🔗 Related

Return to [main documentation](../README.md) for the complete processing pipeline.

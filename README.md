# Ground Control Point (GCP) Inference Pipeline

## Project Description
This repository contains a computer vision and geospatial inference pipeline designed to detect Ground Control Points (GCPs) from large aerial orthomosaic rasters. To process high-resolution GeoTIFF images under memory constraints, the pipeline uses a bounded-memory sliding window approach to extract tile windows, feeds them into an ONNX pose estimation model, filters candidates by confidence score, merges overlapping keypoints through spatial deduplication, and reprojects local pixel coordinates to global WGS84 geographic coordinates.

## How It Was Built
- **Language & Libraries**: Built with Python using `rasterio` for windowed raster I/O, `onnxruntime` for model inference, `pyproj` for spatial coordinate reprojection, and `numpy`/`opencv-python` for array processing.
- **Tiling & Deduplication**: Features sliding window extraction with a 128-pixel stride, background tile filtering, and distance-based spatial weighted averaging (20-pixel radius) to eliminate duplicate detections.
- **Containerization**: Includes a multi-stage `Dockerfile` configured to run in an offline, read-only, non-root evaluation environment.

## Prerequisites
Before running the pipeline, ensure you have:
1. **Python 3.10+** installed (or **Docker** for containerized execution).
2. **ONNX Model File**: `gcp_pose.onnx` placed inside the `model/` directory.
3. **Target Rasters & Manifest**: GeoTIFF (`.tif`) files and `manifest.json` placed inside `data/test/`.

### Directory Structure
```text
.
├── Dockerfile
├── README.md
├── predictions.json
├── requirements.txt
├── validate_predictions.py
├── solution/
│   ├── dataset.py
│   ├── infer.py
│   ├── postprocess.py
│   └── spatial.py
├── model/                  <-- Place gcp_pose.onnx here
│   └── gcp_pose.onnx
└── data/                   <-- Place manifest.json and rasters here
    └── test/
        ├── manifest.json
        └── rasters/
            └── <scene_id>.tif

```

## How to Run

### Method 1: Python Virtual Environment

1. Install dependencies:
```bash
pip install -r requirements.txt

```


2. Run inference:
```bash
python -m solution.infer \
  --manifest data/test/manifest.json \
  --model model/gcp_pose.onnx \
  --output predictions.json

```


3. Validate the output schema:
```bash
python validate_predictions.py --predictions predictions.json

```



### Method 2: Docker Container

1. Build the image:
```bash
docker build -t gcp-submission .

```


2. Run the container:
```bash
docker run --rm \
  --network none \
  -v "$(pwd)/data/test:/input:ro" \
  -v "$(pwd)/model:/model:ro" \
  -v "$(pwd):/output" \
  gcp-submission \
  --manifest /input/manifest.json \
  --model /model/gcp_pose.onnx \
  --output /output/predictions.json

```

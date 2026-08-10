# Skylark GCP Inference Pipeline Solution

## Pipeline Overview
This repository contains a production-oriented inference pipeline designed to detect Ground Control Points (GCPs) from large aerial orthomosaics using a pre-trained ONNX pose estimation model.

## Core Features & Design Decisions
- **Bounded-Memory Windowing:** Processes large rasters in $640 \times 640$ tile windows with a 128-pixel stride overlap to avoid loading whole images into RAM.
- **Background Filtering:** Automatically skips pure zero/black background padding tiles to accelerate processing speed.
- **ONNX Inference & Post-Processing:** Parses dense candidate outputs (`[1, 9, 8400]`), filtering by keypoint confidence score ($>= 0.35$).
- **Multi-Window Deduplication:** Groups overlapping candidate keypoints across adjacent window tiles using weighted spatial averaging within a 20-pixel radius.
- **Geospatial Coordinate Reprojection:** Maps local window pixel coordinates to full-raster continuous column/row pixels, then translates native spatial map coordinates to WGS84 (`EPSG:4326`) `[longitude, latitude]` format.

## Container Execution Commands
```bash
# Build Docker Image
docker build --platform linux/amd64 -t gcp-submission .

# Run Offline Test Container
docker run --rm \
  --platform linux/amd64 \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,size=2g \
  -v "/path/to/data/test:/input:ro" \
  -v "/path/to/model:/model:ro" \
  -v "/path/to/output:/output" \
  gcp-submission \
  --manifest /input/manifest.json \
  --model /model/gcp_pose.onnx \
  --output /output/predictions.json
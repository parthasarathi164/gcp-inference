import argparse
import json
import os
import numpy as np
import onnxruntime as ort
import rasterio

from solution.dataset import generate_windows, preprocess_tile
from solution.postprocess import parse_onnx_output, local_to_global_pixels
from solution.spatial import deduplicate_detections, convert_pixels_to_wgs84

def process_scene(scene_id, raster_path, session, input_name, score_thresh=0.35, dist_thresh=20.0):
    if not os.path.exists(raster_path):
        print(f"Warning: Raster file {raster_path} not found. Returning empty detections.")
        return {"scene_id": scene_id, "detections": []}

    detections = []
    
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs

        for window, col_off, row_off in generate_windows(src.width, src.height):
            tensor, c_off, r_off = preprocess_tile(src, window)
            if tensor is None:
                continue

            input_batch = np.expand_dims(tensor, axis=0)
            raw_out = session.run(None, {input_name: input_batch})[0]
            kpts_local, scores = parse_onnx_output(raw_out, score_threshold=score_thresh)

            if len(kpts_local) > 0:
                kpts_global = local_to_global_pixels(kpts_local, c_off, r_off)
                for (px, py), score in zip(kpts_global, scores):
                    detections.append({
                        'pixel_x': float(px),
                        'pixel_y': float(py),
                        'confidence': float(score)
                    })

        # Deduplicate predictions across overlapping windows
        merged = deduplicate_detections(detections, distance_threshold=dist_thresh)

        # Convert to WGS84
        final_detections = []
        for det in merged:
            lon, lat = convert_pixels_to_wgs84(det['pixel_x'], det['pixel_y'], transform, crs)
            final_detections.append({
                "pixel_x": round(det['pixel_x'], 2),
                "pixel_y": round(det['pixel_y'], 2),
                "longitude": round(lon, 7),
                "latitude": round(lat, 7),
                "confidence": round(det['confidence'], 4)
            })

    return {"scene_id": scene_id, "detections": final_detections}

def main():
    parser = argparse.ArgumentParser(description="Skylark GCP Inference Pipeline")
    parser.add_argument("--manifest", required=True, help="Path to input manifest JSON")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--output", required=True, help="Path to write output predictions JSON")
    args = parser.parse_args()

    # Load manifest
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    with open(args.manifest, "r") as f:
        manifest_data = json.load(f)

    # Initialize ONNX Session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(args.model, opts)
    input_name = session.get_inputs()[0].name

    results = {
        "schema_version": "1.0",
        "scenes": []
    }

    for scene in manifest_data.get("scenes", []):
        scene_id = scene["scene_id"]
        rel_raster_path = scene["raster_path"]
        abs_raster_path = os.path.normpath(os.path.join(manifest_dir, rel_raster_path))

        print(f"Processing scene: {scene_id} -> {abs_raster_path}")
        scene_result = process_scene(scene_id, abs_raster_path, session, input_name)
        results["scenes"].append(scene_result)

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save output JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Successfully generated predictions at {args.output}")

if __name__ == "__main__":
    main()
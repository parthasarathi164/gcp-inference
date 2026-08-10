import numpy as np
from pyproj import Transformer
import rasterio.transform

def deduplicate_detections(detections, distance_threshold=20.0):
    """
    Groups nearby raw detections within a pixel radius (distance_threshold)
    and merges them into a single representative GCP detection.
    """
    if not detections:
        return []

    # Convert list of dicts to numpy array of coordinates and scores
    coords = np.array([[d['pixel_x'], d['pixel_y']] for d in detections], dtype=np.float32)
    scores = np.array([d['confidence'] for d in detections], dtype=np.float32)

    # Sort candidates by confidence descending
    order = np.argsort(-scores)
    coords = coords[order]
    scores = scores[order]

    keep = []
    used = np.zeros(len(coords), dtype=bool)

    for i in range(len(coords)):
        if used[i]:
            continue

        # Find all unvisited neighbors within the pixel distance threshold
        dist = np.linalg.norm(coords[i] - coords, axis=1)
        neighbors = np.where((dist < distance_threshold) & (~used))[0]

        # Mark neighbors as used
        used[neighbors] = True

        # Calculate weighted average position based on confidence scores
        neighbor_scores = scores[neighbors]
        weights = neighbor_scores / neighbor_scores.sum()
        
        avg_x = float(np.sum(coords[neighbors, 0] * weights))
        avg_y = float(np.sum(coords[neighbors, 1] * weights))
        max_conf = float(neighbor_scores.max())

        keep.append({
            'pixel_x': avg_x,
            'pixel_y': avg_y,
            'confidence': max_conf
        })

    return keep

def convert_pixels_to_wgs84(pixel_x, pixel_y, transform, src_crs):
    """
    Translates full-raster pixel coordinates (x, y) to spatial native coordinates,
    then reprojects native coordinates to WGS84 EPSG:4326 (longitude, latitude).
    """
    # Map pixel column/row to native CRS spatial coordinates (X, Y in meters)
    native_x, native_y = rasterio.transform.xy(transform, pixel_y, pixel_x)

    # Transform native spatial coordinates to WGS84 EPSG:4326
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(native_x, native_y)

    return float(lon), float(lat)
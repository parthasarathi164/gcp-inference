import numpy as np

def parse_onnx_output(output_tensor, score_threshold=0.3):
    """
    Parses [1, 9, 8400] ONNX model output.
    Returns array of candidate dicts with keypoints in local [0, 640] window coordinates.
    """
    # Transpose to [8400, 9]
    predictions = output_tensor[0].T
    
    boxes_xywh = predictions[:, 0:4]
    class_probs = predictions[:, 4:7]
    keypoints_xy = predictions[:, 7:9]
    
    # Calculate max proposal score across the 3 GCP marker classes
    scores = class_probs.max(axis=1)
    
    # Filter candidates above confidence threshold
    valid_mask = scores >= score_threshold
    
    filtered_keypoints = keypoints_xy[valid_mask]
    filtered_scores = scores[valid_mask]
    
    return filtered_keypoints, filtered_scores

def local_to_global_pixels(keypoints_local, col_off, row_off):
    """
    Converts local window keypoint coordinates (0..640) to full-raster pixel coordinates.
    """
    if len(keypoints_local) == 0:
        return np.empty((0, 2), dtype=np.float32)
    
    global_pixels = np.zeros_like(keypoints_local, dtype=np.float32)
    global_pixels[:, 0] = keypoints_local[:, 0] + col_off  # pixel_x
    global_pixels[:, 1] = keypoints_local[:, 1] + row_off  # pixel_y
    
    return global_pixels
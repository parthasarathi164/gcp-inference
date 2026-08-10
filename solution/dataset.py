import numpy as np
import rasterio
from rasterio.windows import Window

def generate_windows(width, height, window_size=640, stride=512):
    for col_off in range(0, width, stride):
        for row_off in range(0, height, stride):
            w = min(window_size, width - col_off)
            h = min(window_size, height - row_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h), col_off, row_off

def preprocess_tile(raster_src, window, target_size=640):
    tile_data = raster_src.read([1, 2, 3], window=window)
    
    # FAST SKIP: If tile is completely black/empty background, return None
    if not np.any(tile_data):
        return None, window.col_off, window.row_off

    _, h, w = tile_data.shape
    padded_tile = np.zeros((3, target_size, target_size), dtype=np.float32)
    float_data = tile_data.astype(np.float32)

    if raster_src.dtypes[0] == 'uint16':
        max_val = float_data.max()
        if max_val > 4095.0:
            float_data /= 65535.0
        elif max_val > 0:
            float_data /= 4095.0
    elif float_data.max() > 1.0:
        float_data /= 255.0

    float_data = np.clip(float_data, 0.0, 1.0)
    padded_tile[:, :h, :w] = float_data

    return padded_tile, window.col_off, window.row_off
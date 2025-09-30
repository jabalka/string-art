from __future__ import annotations
import numpy as np
import cv2
from functools import lru_cache
from typing import Tuple

def _aa_line_indices_weights(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    thickness_px: int,
    blur_sigma: float,
    canvas_size: int,
    endpoint_taper: float,
):
    """Rasterize an anti-aliased chord segment.

    Applies optional Gaussian blur and a Tukey window based endpoint taper so
    thread contribution fades near nails. Returns a tuple of:
        (flat_pixel_indices[int64], per_pixel_weights[float32])
    """
    x0, y0 = start_point
    x1, y1 = end_point
    raster = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.line(
        raster,
        (int(round(x0)), int(round(y0))),
        (int(round(x1)), int(round(y1))),
        1.0,
        thickness=max(1, int(thickness_px)),
        lineType=cv2.LINE_AA,
    )
    if blur_sigma > 0:
        raster = cv2.GaussianBlur(raster, (0, 0), float(blur_sigma))

    pixel_y, pixel_x = np.nonzero(raster > 0)
    if pixel_y.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Parametric t position along segment for taper weighting.
    X = pixel_x.astype(np.float32) + 0.5
    Y = pixel_y.astype(np.float32) + 0.5
    vx = float(x1 - x0)
    vy = float(y1 - y0)
    length_sq = max(1e-6, vx * vx + vy * vy)
    t = ((X - x0) * vx + (Y - y0) * vy) / length_sq
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    taper = float(max(0.0, min(1.0, endpoint_taper)))
    if taper > 0:
        endpoint_weights = np.ones_like(t, dtype=np.float32)
        near_start = t < taper / 2
        endpoint_weights[near_start] = 0.5 * (1 - np.cos(2 * np.pi * t[near_start] / taper))
        near_end = t > 1 - taper / 2
        endpoint_weights[near_end] = 0.5 * (1 - np.cos(2 * np.pi * (1 - t[near_end]) / taper))
    else:
        endpoint_weights = np.ones_like(t, dtype=np.float32)

    weights = raster[pixel_y, pixel_x].astype(np.float32) * endpoint_weights
    keep_mask = weights > 1e-7
    if not np.any(keep_mask):
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    pixel_y = pixel_y[keep_mask]
    pixel_x = pixel_x[keep_mask]
    weights = weights[keep_mask]
    flat_indices = (pixel_y.astype(np.int64) * raster.shape[1]) + pixel_x.astype(np.int64)
    return flat_indices, weights

@lru_cache(maxsize=500_000)
def get_chord_mask(
    nail_index_a: int,
    nail_index_b: int,
    nail_positions,
    canvas_size: int,
    thickness_px: int,
    blur_sigma: float,
    endpoint_taper: float = 0.2,
):
    """Return (indices, weights) for drawing a chord between two nails.

    Cached aggressively because many chords repeat during the search.
    """
    start_point = nail_positions[nail_index_a]
    end_point = nail_positions[nail_index_b]
    return _aa_line_indices_weights(start_point, end_point, thickness_px, float(blur_sigma), canvas_size, float(endpoint_taper))

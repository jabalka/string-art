from __future__ import annotations
import numpy as np
import cv2
from functools import lru_cache
from typing import Tuple

def _aa_line_indices_weights(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    thickness: int,
    blur_sigma: float,
    canvas: int,
    endpoint_taper: float,
):
    """
    Rasterize an anti-aliased line with optional Gaussian blur and a Tukey
    endpoint taper so weights fade near the endpoints. Returns:
        (flat_indices[int64], weights[float32])
    """
    x0, y0 = p0; x1, y1 = p1
    img = np.zeros((canvas, canvas), dtype=np.float32)

    cv2.line(
        img,
        (int(round(x0)), int(round(y0))),
        (int(round(x1)), int(round(y1))),
        1.0,
        thickness=max(1, int(thickness)),
        lineType=cv2.LINE_AA,
    )
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), float(blur_sigma))

    yy, xx = np.nonzero(img > 0)
    if yy.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Parameter t \in [0,1] for each pixel center projected onto the segment
    X = xx.astype(np.float32) + 0.5
    Y = yy.astype(np.float32) + 0.5
    vx = float(x1 - x0); vy = float(y1 - y0)
    L2 = max(1e-6, vx * vx + vy * vy)
    t = ((X - x0) * vx + (Y - y0) * vy) / L2
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    # Tukey window taper on the ends
    a = float(max(0.0, min(1.0, endpoint_taper)))  # 0..1
    if a > 0:
        w_end = np.ones_like(t, dtype=np.float32)
        # near start
        m1 = t < a / 2
        w_end[m1] = 0.5 * (1 - np.cos(2 * np.pi * t[m1] / a))
        # near end
        m2 = t > 1 - a / 2
        w_end[m2] = 0.5 * (1 - np.cos(2 * np.pi * (1 - t[m2]) / a))
    else:
        w_end = np.ones_like(t, dtype=np.float32)

    w = img[yy, xx].astype(np.float32) * w_end
    keep = w > 1e-7
    if not np.any(keep):
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    yy = yy[keep]; xx = xx[keep]; w = w[keep]
    idx = (yy.astype(np.int64) * img.shape[1]) + xx.astype(np.int64)
    return idx, w

@lru_cache(maxsize=500_000)
def get_chord_mask(
    i: int,
    j: int,
    nails_xy,
    canvas: int,
    thickness: int,
    blur_sigma: float,
    endpoint_taper: float = 0.2,
):
    p0 = nails_xy[i]
    p1 = nails_xy[j]
    return _aa_line_indices_weights(p0, p1, thickness, float(blur_sigma), canvas, float(endpoint_taper))

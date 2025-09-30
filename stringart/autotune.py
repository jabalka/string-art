from __future__ import annotations
import os
import numpy as np
import cv2
from typing import Dict, Any

from .detect import detect_board_circle, detect_nail_count


def auto_config(
    image_path: str,
    canvas_size: int = 1200,
    invert: bool = True,
    apply_mask: bool = True,
    auto_nails: bool = True,
    verbose: bool = True,
    min_nails: int = 20,
    max_nails: int = 400,
    debug_dir: str | None = None,
) -> Dict[str, Any]:
    """Infer board geometry & nail count; return a SolverParams-compatible dict.

    Steps:
      1. Read image as grayscale in [0,1].
      2. Detect board circle (center & radius) – Hough + fallback.
      3. Optionally detect nail count (FFT angular analysis). Otherwise default=240.
      4. Produce heuristic solver parameter defaults.
    """
    raw_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise FileNotFoundError(image_path)
    gray_normalized = cv2.normalize(raw_gray.astype(np.float32), raw_gray.astype(np.float32), 0, 1, cv2.NORM_MINMAX)

    center_x, center_y, board_radius = detect_board_circle(gray_normalized)

    if auto_nails:
        nail_count = detect_nail_count(
            gray_normalized,
            center_x,
            center_y,
            board_radius,
            min_nails=min_nails,
            max_nails=max_nails,
            debug_dir=debug_dir,
        )
    else:
        nail_count = 240

    if verbose:
        print(f"[autotune] Detected {nail_count} nails at radius={board_radius}px (center=({center_x},{center_y}))")

    scaled_radius = int(board_radius * (canvas_size / raw_gray.shape[0]))
    params: Dict[str, Any] = dict(
        canvas=canvas_size,
        radius=scaled_radius,
        nails=nail_count,
        invert=invert,
        apply_mask=apply_mask,
        alpha=0.22,
        line_thickness_px=1,
        blur_sigma=0.8,
        min_chord_px=30,
        per_nail_max_hits=40,
        auto_steps=True,
        coverage=0.85,
        rel_improve=1e-4,
        abs_improve=1e-6,
        patience=50,
        window=20,
        endpoint_taper=0.25,
        angle_smooth=0.35,
    )
    return params

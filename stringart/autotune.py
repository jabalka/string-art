from __future__ import annotations
import os
import numpy as np
import cv2
from typing import Dict, Any

from .detect import detect_board_circle, detect_nail_count


def auto_config(
    image_path: str,
    canvas: int = 1200,
    invert: bool = True,
    apply_mask: bool = True,
    auto_nails: bool = True,
    verbose: bool = True,
    n_min: int = 20,
    n_max: int = 400,
    debug_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Automatically determine circle center/radius and nail count from an input image.
    Returns a dict of parameters for SolverParams.
    """

    # load + grayscale [0..1]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    gray01 = cv2.normalize(img.astype(np.float32), img.astype(np.float32), 0, 1, cv2.NORM_MINMAX)

    # detect board circle
    cx, cy, R = detect_board_circle(gray01)

    # nail count detection
    if auto_nails:
        nails = detect_nail_count(gray01, cx, cy, R, n_min=n_min, n_max=n_max, debug_dir=debug_dir)
    else:
        nails = 240  # fallback default

    if verbose:
        print(f"[autotune] Detected {nails} nails at radius={R}px (center=({cx},{cy}))")

    # heuristic defaults
    params: Dict[str, Any] = dict(
        canvas=canvas,
        radius=int(R * (canvas / img.shape[0])),  # scale to resized canvas
        nails=nails,
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

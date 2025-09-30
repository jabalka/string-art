from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageOps
import cv2

from .geometry import generate_nails

def _to_gray01(im: Image.Image) -> np.ndarray:
    g = im.convert("L")
    return np.asarray(g, dtype=np.float32) / 255.0

def _square_letterbox(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    m = max(h, w)
    out = np.zeros((m, m), dtype=arr.dtype)
    y0 = (m - h) // 2; x0 = (m - w) // 2
    out[y0:y0+h, x0:x0+w] = arr
    return out

def _feathered_circle_mask(canvas: int, radius: int, edge_margin_px: int, feather_px: int) -> np.ndarray:
    r_inner = max(0, radius - max(0, edge_margin_px))
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    cx = cy = canvas // 2
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (rr <= r_inner).astype(np.float32)
    if feather_px > 0:
        ring = (rr > r_inner) & (rr <= r_inner + feather_px)
        mask[ring] = np.clip(1.0 - (rr[ring] - r_inner) / float(feather_px), 0.0, 1.0)
    return mask

def _suppress_nail_heads(target: np.ndarray, canvas: int, radius: int, nail_radius_px: int, n: int) -> None:
    if nail_radius_px <= 0:
        return
    cx = cy = canvas // 2
    nails = generate_nails(n=n, center=(cx, cy), radius=radius)
    for (x, y) in nails:
        cv2.circle(target, (int(round(x)), int(round(y))), nail_radius_px, 0.0, thickness=-1)

def preprocess_image(
    image_path: str,
    canvas: int,
    radius: int,
    invert: bool = False,
    apply_mask: bool = True,
    outdir: str | None = None,
    rim: str = "keep",               # "keep" | "feather" | "erode"
    edge_margin_px: int = 12,
    feather_px: int = 8,
    suppress_nail_heads_px: int = 0,
    n: int = 240,
) -> np.ndarray:
    """Letterbox (no crop), resize, optional rim policy, optional nail suppression."""
    pil = Image.open(image_path).convert("RGB")
    pil = ImageOps.autocontrast(pil)

    gray = _to_gray01(pil)
    gray = _square_letterbox(gray)
    gray = cv2.resize(gray, (canvas, canvas), interpolation=cv2.INTER_AREA)
    gray = np.clip(gray, 0.0, 1.0)

    target = (1.0 - gray) if invert else gray
    target = target.astype(np.float32)

    if apply_mask:
        if rim == "keep":
            mask = _feathered_circle_mask(canvas, radius, edge_margin_px=0, feather_px=0)
        elif rim == "feather":
            mask = _feathered_circle_mask(canvas, radius, edge_margin_px=edge_margin_px, feather_px=feather_px)
        else:  # erode
            mask = _feathered_circle_mask(canvas, radius, edge_margin_px=edge_margin_px, feather_px=0)
        target *= mask

    if suppress_nail_heads_px > 0:
        _suppress_nail_heads(target, canvas, radius, nail_radius_px=suppress_nail_heads_px, n=n)

    target = np.clip(target, 0.0, 1.0)

    if outdir:
        orig = Image.open(image_path).convert("RGB")
        orig_vis = ImageOps.contain(orig, (canvas, canvas))
        left = Image.new("RGB", (canvas, canvas), (0, 0, 0))
        left.paste(orig_vis, ((canvas - orig_vis.width)//2, (canvas - orig_vis.height)//2))
        right_vis = Image.fromarray(np.uint8(target * 255)).convert("L").convert("RGB")
        combo = Image.new("RGB", (canvas*2, canvas), (0,0,0))
        combo.paste(left, (0,0)); combo.paste(right_vis, (canvas,0))
        combo.save(os.path.join(outdir, "preprocess_preview.png"), "PNG")

    return target

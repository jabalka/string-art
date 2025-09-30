from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageOps
import cv2

from .geometry import generate_nails

def _to_gray01(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to float32 grayscale in [0,1]."""
    grayscale = image.convert("L")
    return np.asarray(grayscale, dtype=np.float32) / 255.0

def _square_letterbox(image_gray: np.ndarray) -> np.ndarray:
    """Pad the 2D array to a square canvas with zeros (centered)."""
    height, width = image_gray.shape[:2]
    largest_dim = max(height, width)
    output = np.zeros((largest_dim, largest_dim), dtype=image_gray.dtype)
    top = (largest_dim - height) // 2
    left = (largest_dim - width) // 2
    output[top:top+height, left:left+width] = image_gray
    return output

def _feathered_circle_mask(canvas_size: int, radius: int, edge_margin_px: int, feather_px: int) -> np.ndarray:
    """Create a circular mask with optional feathered edge.

    edge_margin_px shrinks the hard radius, feather_px adds a linear fade-out region.
    """
    inner_radius = max(0, radius - max(0, edge_margin_px))
    y_grid, x_grid = np.mgrid[0:canvas_size, 0:canvas_size]
    center = canvas_size // 2
    radial_dist = np.sqrt((x_grid - center) ** 2 + (y_grid - center) ** 2)
    mask = (radial_dist <= inner_radius).astype(np.float32)
    if feather_px > 0:
        feather_zone = (radial_dist > inner_radius) & (radial_dist <= inner_radius + feather_px)
        mask[feather_zone] = np.clip(
            1.0 - (radial_dist[feather_zone] - inner_radius) / float(feather_px), 0.0, 1.0
        )
    return mask

def _suppress_nail_heads(image_gray: np.ndarray, canvas_size: int, radius: int, nail_radius_px: int, nail_count: int) -> None:
    """Optionally darken (zero) tiny circular regions where nail heads would appear.

    This prevents the solver from over-attributing importance to bright/dark spots at the nail perimeter.
    """
    if nail_radius_px <= 0:
        return
    center = canvas_size // 2
    nail_positions = generate_nails(n=nail_count, center=(center, center), radius=radius)
    for (x_pos, y_pos) in nail_positions:
        cv2.circle(image_gray, (int(round(x_pos)), int(round(y_pos))), nail_radius_px, 0.0, thickness=-1)

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
    mask_inflate_frac: float = 0.0,
    defer_mask: bool = False,
) -> np.ndarray:
    """Letterbox (no crop), resize.

    If defer_mask=True the circular masking is skipped (caller can mask later).
    mask_inflate_frac inflates the mask radius before application.
    """
    original_rgb = Image.open(image_path).convert("RGB")
    original_rgb = ImageOps.autocontrast(original_rgb)

    gray01 = _to_gray01(original_rgb)
    gray01 = _square_letterbox(gray01)
    gray01 = cv2.resize(gray01, (canvas, canvas), interpolation=cv2.INTER_AREA)
    gray01 = np.clip(gray01, 0.0, 1.0)

    processed = (1.0 - gray01) if invert else gray01
    processed = processed.astype(np.float32)

    if apply_mask and not defer_mask:
        inflated_radius = int(min(canvas//2 - 1, radius * (1.0 + max(0.0, mask_inflate_frac))))
        if rim == "keep":
            mask = _feathered_circle_mask(canvas, inflated_radius, edge_margin_px=0, feather_px=0)
        elif rim == "feather":
            mask = _feathered_circle_mask(canvas, inflated_radius, edge_margin_px=edge_margin_px, feather_px=feather_px)
        else:  # erode
            mask = _feathered_circle_mask(canvas, inflated_radius, edge_margin_px=edge_margin_px, feather_px=0)
        processed *= mask

    if suppress_nail_heads_px > 0:
        _suppress_nail_heads(processed, canvas, radius, nail_radius_px=suppress_nail_heads_px, nail_count=n)

    processed = np.clip(processed, 0.0, 1.0)

    if outdir:
        # Side-by-side preview (original vs processed)
        original_for_canvas = ImageOps.contain(original_rgb, (canvas, canvas))
        left_panel = Image.new("RGB", (canvas, canvas), (0, 0, 0))
        left_panel.paste(original_for_canvas, ((canvas - original_for_canvas.width)//2, (canvas - original_for_canvas.height)//2))
        processed_vis = Image.fromarray(np.uint8(processed * 255)).convert("L").convert("RGB")
        combined = Image.new("RGB", (canvas*2, canvas), (0,0,0))
        combined.paste(left_panel, (0,0)); combined.paste(processed_vis, (canvas,0))
        combined.save(os.path.join(outdir, "preprocess_preview.png"), "PNG")

    return processed

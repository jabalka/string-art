from __future__ import annotations
import math, os
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from .geometry import generate_nails, index_to_label, generate_default_canvas, NAILS_DEFAULT

def _load_font(size: int):
    for name in ["Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def render_nail_map(
    out_path: str,
    canvas: Optional[int] = None,
    radius: Optional[int] = None,
    n: int = NAILS_DEFAULT,
    show_labels: bool = True,
    show_indices: bool = False,
    stroke_width: int = 2,
    label_every: int = 1,
) -> str:
    canvas, radius = generate_default_canvas(canvas, radius)
    cx = cy = canvas // 2
    img = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    dr = ImageDraw.Draw(img)

    dr.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(0, 0, 0), width=stroke_width)

    nails = generate_nails(n=n, center=(cx, cy), radius=radius)
    r_dot = max(2, radius // 120)
    font = _load_font(size=max(10, canvas // 60))

    for i, (x, y) in enumerate(nails):
        dr.ellipse((x - r_dot, y - r_dot, x + r_dot, y + r_dot), fill=(0, 0, 0))
        if label_every > 1 and (i % label_every != 0):
            continue
        if show_labels or show_indices:
            parts = []
            if show_labels:
                parts.append(index_to_label(i, n=n))
            if show_indices:
                parts.append(str(i))
            text = " / ".join(parts)
            vx, vy = x - cx, y - cy
            norm = math.hypot(vx, vy) or 1.0
            tx = cx + (vx / norm) * (radius + 18)
            ty = cy + (vy / norm) * (radius + 18)
            w, h = dr.textbbox((0, 0), text, font=font)[2:]
            dr.text((tx - w / 2, ty - h / 2), text, font=font, fill=(0, 0, 0))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return out_path

from __future__ import annotations
import math, os
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from .geometry import generate_nails, index_to_label, generate_default_canvas, NAILS_DEFAULT


def _load_font(size: int):
    """Attempt to load a truetype font, falling back to the default bitmap font.

    Tries a small list of common cross‑platform font names; if unavailable, uses the
    Pillow built‑in default so that text rendering always succeeds.
    """
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
    """Render a labeled map of nail positions to an image file.

    Parameters
    ----------
    out_path : str
        Destination PNG path. Directories are created if needed.
    canvas : int, optional
        Canvas size in pixels (square). If omitted, a default size is used.
    radius : int, optional
        Board radius in pixels. If omitted, derived from the canvas.
    n : int
        Number of nails placed evenly on the circle (backward compatible name;
        elsewhere referred to as nail_count).
    show_labels : bool
        If True, draw alphabetic sector + index style labels (A1, B17, ...).
    show_indices : bool
        If True, include the raw numeric index next to the label (e.g. A1 / 0).
    stroke_width : int
        Outline stroke width for the outer circle.
    label_every : int
        Only annotate every k-th nail when > 1 to reduce clutter.

    Returns
    -------
    str
        The output path (same as input `out_path`).
    """
    canvas, radius = generate_default_canvas(canvas, radius)
    center_x = center_y = canvas // 2

    image = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Outer board circle
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        outline=(0, 0, 0),
        width=stroke_width,
    )

    nail_positions = generate_nails(n=n, center=(center_x, center_y), radius=radius)
    nail_dot_radius = max(2, radius // 120)
    font = _load_font(size=max(10, canvas // 60))

    for nail_index, (x, y) in enumerate(nail_positions):
        # Draw the nail head.
        draw.ellipse(
            (x - nail_dot_radius, y - nail_dot_radius, x + nail_dot_radius, y + nail_dot_radius),
            fill=(0, 0, 0),
        )
        if label_every > 1 and (nail_index % label_every != 0):
            continue
        if show_labels or show_indices:
            label_parts = []
            if show_labels:
                # index_to_label expects 'nail_count' keyword (renamed during refactor)
                label_parts.append(index_to_label(nail_index, nail_count=n))
            if show_indices:
                label_parts.append(str(nail_index))
            text = " / ".join(label_parts)

            # Position the text slightly outside the circle along the radial direction.
            vec_x, vec_y = x - center_x, y - center_y
            norm = math.hypot(vec_x, vec_y) or 1.0
            text_x = center_x + (vec_x / norm) * (radius + 18)
            text_y = center_y + (vec_y / norm) * (radius + 18)
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
            draw.text(
                (text_x - text_width / 2, text_y - text_height / 2),
                text,
                font=font,
                fill=(0, 0, 0),
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    image.save(out_path, "PNG")
    return out_path

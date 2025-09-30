from __future__ import annotations
import math
from typing import List, Tuple

# Defaults (kept for backward compatibility)
NAILS_DEFAULT = 240
SECTOR_LETTERS = ("A", "B", "C", "D")

def index_to_label(nail_index: int, nail_count: int = NAILS_DEFAULT) -> str:
    """Return a human readable label for a nail index.

    Rules:
      * If nail_count == 240: EXACT mapping A1..A60, B1..B60, C1..C60, D1..D60 (spec requirement)
      * Else if nail_count divisible by 4: keep sector letters A..D with size = nail_count/4
      * Else: fall back to generic numbering N1..N{n}
    """
    if not (0 <= nail_index < nail_count):
        raise ValueError(f"index out of range: {nail_index} for n={nail_count}")

    if nail_count == 240 or (nail_count % 4 == 0):
        nails_per_sector = nail_count // 4
        sector_idx = nail_index // nails_per_sector
        ordinal_in_sector = (nail_index % nails_per_sector) + 1
        return f"{SECTOR_LETTERS[sector_idx]}{ordinal_in_sector}"

    # Generic (non 4-divisible counts)
    return f"N{nail_index+1}"

def label_to_index(label: str, nail_count: int = NAILS_DEFAULT) -> int:
    """Inverse of ``index_to_label`` for the given ``nail_count``.

    Accepts:
      * A1..D{k} when nail_count % 4 == 0 (or nail_count == 240)
      * N1..N{n}
    Returns the zero-based nail index.
    """
    if not label or len(label) < 2:
        raise ValueError(f"invalid label: {label!r}")

    sector_char = label[0].upper()
    numeric_part = label[1:]

    if (nail_count == 240) or (nail_count % 4 == 0):
        if sector_char in SECTOR_LETTERS:
            try:
                ordinal = int(numeric_part)
            except ValueError as exc:
                raise ValueError(f"invalid numeric part in label: {label!r}") from exc
            nails_per_sector = nail_count // 4
            if not (1 <= ordinal <= nails_per_sector):
                raise ValueError(f"label number must be 1..{nails_per_sector}: {label!r}")
            base_index = SECTOR_LETTERS.index(sector_char) * nails_per_sector
            return base_index + (ordinal - 1)

    if sector_char == "N":
        try:
            ordinal = int(numeric_part)
        except ValueError as exc:
            raise ValueError(f"invalid numeric part in label: {label!r}") from exc
        if not (1 <= ordinal <= nail_count):
            raise ValueError(f"label number must be 1..{nail_count}: {label!r}")
        return ordinal - 1

    raise ValueError(f"invalid label for n={nail_count}: {label!r}")

def generate_default_canvas(canvas_size: int | None, circle_radius: int | None) -> Tuple[int, int]:
    """Derive a sensible (canvas_size, circle_radius) pair.

    If both are ``None`` picks defaults. If one is provided infer the other
    using the historical ratio (radius ≈ 45% of canvas).
    """
    if canvas_size is None and circle_radius is None:
        canvas_size = 1200
        circle_radius = int(canvas_size * 0.45)
    elif canvas_size is None:
        if circle_radius is not None:
            canvas_size = int(circle_radius / 0.45)
        else:
            raise ValueError("circle_radius cannot be None when calculating canvas_size")
    elif circle_radius is None:
        circle_radius = int(canvas_size * 0.45)
    return int(canvas_size), int(circle_radius)

def generate_nails(
    nail_count: int = NAILS_DEFAULT,
    center: Tuple[float, float] = (600.0, 600.0),
    radius: float = 540.0,
    **legacy_kwargs,
) -> List[Tuple[float, float]]:
    """Return a list of ``(x, y)`` positions for equally spaced nails on a circle.

    Backward compatibility: older code called this with ``generate_nails(n=...)``.
    We accept that via **legacy_kwargs and map it to ``nail_count`` if present.

    The first nail (index 0) is placed at 12 o'clock and positions advance clockwise.
    Screen coordinate system: x grows to the right, y grows downward.
    Formula: theta(i) = -π/2 + 2π * (i / nail_count)
    """
    if 'n' in legacy_kwargs and legacy_kwargs['n'] is not None:
        nail_count = legacy_kwargs['n']
    center_x, center_y = center
    nail_positions: List[Tuple[float, float]] = []
    for nail_index in range(nail_count):
        theta = -math.pi / 2.0 + (2.0 * math.pi * nail_index / nail_count)
        x_pos = center_x + radius * math.cos(theta)
        y_pos = center_y + radius * math.sin(theta)
        nail_positions.append((x_pos, y_pos))
    return nail_positions

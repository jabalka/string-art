from __future__ import annotations
import math
from typing import List, Tuple

# Defaults (kept for backward compatibility)
NAILS_DEFAULT = 240
SECTOR_LETTERS = ("A", "B", "C", "D")

def index_to_label(i: int, n: int = NAILS_DEFAULT) -> str:
    """
    Convert 0..n-1 to a human label.
    - If n == 240: EXACTLY A1..A60, B1..B60, C1..C60, D1..D60 (spec requirement).
    - Else if n is divisible by 4: keep A..D with size = n/4.
    - Else: generic labels N1..Nn.
    """
    if not (0 <= i < n):
        raise ValueError(f"index out of range: {i} for n={n}")

    if n == 240 or (n % 4 == 0):
        per = n // 4
        sector_idx = i // per
        num = (i % per) + 1
        # For n==240 this yields the exact A..D + 1..60 mapping
        return f"{SECTOR_LETTERS[sector_idx]}{num}"

    # Generic (non 4-divisible counts)
    return f"N{i+1}"

def label_to_index(label: str, n: int = NAILS_DEFAULT) -> int:
    """
    Inverse of index_to_label for the chosen n.
    Accepts: A1..Dk when n%4==0, or N1..Nn for any n.
    """
    if not label or len(label) < 2:
        raise ValueError(f"invalid label: {label!r}")

    s = label[0].upper()
    rest = label[1:]

    # Case 1: A..D when allowed
    if (n == 240) or (n % 4 == 0):
        if s in SECTOR_LETTERS:
            try:
                num = int(rest)
            except ValueError:
                raise ValueError(f"invalid numeric part in label: {label!r}")
            per = n // 4
            if not (1 <= num <= per):
                raise ValueError(f"label number must be 1..{per}: {label!r}")
            base = SECTOR_LETTERS.index(s) * per
            return base + (num - 1)

    # Case 2: Generic N1..Nn
    if s == "N":
        try:
            num = int(rest)
        except ValueError:
            raise ValueError(f"invalid numeric part in label: {label!r}")
        if not (1 <= num <= n):
            raise ValueError(f"label number must be 1..{n}: {label!r}")
        return num - 1

    raise ValueError(f"invalid label for n={n}: {label!r}")

def generate_default_canvas(canvas: int | None, radius: int | None) -> Tuple[int, int]:
    if canvas is None and radius is None:
        canvas = 1200
        radius = int(canvas * 0.45)
    elif canvas is None:
        if radius is not None:
            canvas = int(radius / 0.45)
        else:
            raise ValueError("radius cannot be None when calculating canvas")
    elif radius is None:
        radius = int(canvas * 0.45)
    return int(canvas), int(radius)

def generate_nails(
    n: int = NAILS_DEFAULT,
    center: Tuple[float, float] = (600.0, 600.0),
    radius: float = 540.0
) -> List[Tuple[float, float]]:
    """
    Generate n nails uniformly on a circle, starting at 12 o'clock (index 0),
    then clockwise. Screen coords: x right, y down.
    theta(i) = -pi/2 + 2*pi*(i/n)
    """
    cx, cy = center
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        theta = -math.pi / 2.0 + (2.0 * math.pi * i / n)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pts.append((x, y))
    return pts

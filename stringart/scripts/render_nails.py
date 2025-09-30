"""
Small helper script: render a nail map image using the library API directly.

Usage (after `pip install -e .`):
  python scripts/render_nails.py --out out --canvas 1200 --radius 540
"""
from __future__ import annotations
import argparse
import os
from stringart.visualize import render_nail_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--canvas", type=int, default=1200)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--show-indices", action="store_true")
    parser.add_argument("--label-every", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "nail_map.png")
    render_nail_map(
        out_path=out_path,
        canvas=args.canvas,
        radius=args.radius,
        show_labels=not args.no_labels,
        show_indices=args.show_indices,
        label_every=args.label_every,
    )
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()

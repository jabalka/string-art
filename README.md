# String Art Generator

Turn an input image into a sequence of nail-to-nail string instructions with progressive previews and quality metrics (MSE / PSNR / SSIM). The solver uses a greedy residual‑reduction algorithm with optional refinement passes and several quality heuristics (coverage plateau detection, average nail hit limit, inner hole handling, etc.).

## Quick Start

Example (auto detect board geometry & nail count, detect inner hole, run 2 refinement rounds):

```
stringart run -i "image_example.jpg" -o out --auto-all --auto-inner-hole --refine-rounds 2 --refine-mse-thresh 0
```

Render only the nail legend (labels around the rim):

```
stringart nails --out legend --canvas 1200 --radius 540 --nails 240
```

Output directory will contain (typical):
* instructions.csv / instructions.txt – ordered list of nail indices
* preview_overlay.png / preview_final.png – visualization of the pattern
* metrics.json – final MSE / PSNR / SSIM
* params.json – serialized solver parameters
* nail_detect_debug.png – (when --debug-detect) nail count FFT plot
* hole_detect_debug.png – (when --hole-debug) radial profile plot

## Notable Features

* Automatic board circle & nail count detection (FFT-based angular frequency with harmonic suppression)
* Adaptive maximum step count when not specified
* Plateau & coverage based early stopping
* Optional refinement rounds that re-evaluate earlier chords with decreasing alpha
* Inner hole automatic detection (skip or dampen scoring inside a detected void)
* Endpoint taper + slight angle smoothing to reduce over-darkening & ringing artifacts
* Lightweight CLI built with `click`

## Important CLI Flags (run)

* `--auto-all` – Infer nail count, radius and default params.
* `--auto-nails/--no-auto-nails` – Control only nail count auto detection.
* `--steps` – Force a fixed maximum number of greedy steps (disables auto heuristic).
* `--refine-rounds` / `--refine-mse-thresh` – Post greedy refinement cycles and stopping threshold.
* `--auto-inner-hole` – Detect an inner void; may set mode to skip or dampen.
* `--inner-hole-mode skip|dampen` – Manual override when not auto detecting.
* `--avg-hits-stop` – Stop when average nail usage reaches a threshold (helps distribution).
* `--save-every N` – Periodically save overlay previews.

## Variable & Naming Refactor (2025-09)

The codebase was refactored for clarity. Key improvements:

* Replaced short / cryptic variable names with descriptive ones (`cx, cy` → `center_x, center_y`, `prof` → `radial_profile`, etc.)
* Added docstrings to every public function.
* Introduced consistent terminology: `nails` (count), `nail_index`, `nail_positions`, `radius`, `canvas`.
* Backward compatibility maintained for legacy parameters where feasible (e.g., internal functions still accept `n=` in some preprocessing paths).

If you previously relied on internal symbol names, scan the updated modules; public APIs (`SolverParams`, `greedy_solver`, `solve_with_refinement`, CLI flags) remain stable.

## Developer Notes

Core modules:
* `geometry.py` – Nail generation & labeling.
* `preprocess.py` – Image loading, normalization, masking.
* `detect.py` – Circle & nail count detection.
* `hole.py` – Inner void estimation.
* `solver.py` – Greedy + refinement logic.
* `chords.py` – Line rasterization with anti-aliasing & blur.
* `metrics.py` – MSE / PSNR / SSIM utilities.
* `visualize.py` – Legend rendering.

## Example Workflow

1. Prepare a high-contrast source image (preferably square or centrally framed).
2. Run automatic mode:
	```
	stringart run -i input.jpg -o run1 --auto-all --auto-inner-hole
	```
3. Inspect `preview_final.png` & metrics.
4. Increase refinement if fine detail is lacking:
	```
	stringart run -i input.jpg -o run2 --auto-all --refine-rounds 3 --refine-mse-thresh 0
	```

## Troubleshooting

* Nail count seems off – try `--no-auto-nails` and specify `--nails` manually.
* Over-dark center – enable `--auto-inner-hole` or set a manual `--inner-hole-frac`.
* Pattern too light – increase steps (`--steps`) or allow more refinement rounds.
* Performance – reduce canvas size or nails; step count grows roughly with nails^2.

## License

Add license information here (currently unspecified).

---
Refactor (naming + docs) completed 2025-09-30.
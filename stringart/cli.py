from __future__ import annotations
import os
import click
import numpy as np

from .preprocess import preprocess_image
from .solver import greedy_solver, SolverParams, solve_with_refinement
from .autotune import auto_config
from .geometry import generate_nails
from .visualize import render_nail_map


@click.group()
def main():
        """Command line interface for the string‑art generator.

        Provides two subcommands:
            nails  – render a legend image of nail positions.
            run    – execute the full solver on an input image.
        """
        # Click requires a function body – no action needed here.
        return


@main.command()
@click.option("--out", type=click.Path(file_okay=False), required=True, help="Output directory")
@click.option("--canvas", type=int, default=1200, help="Canvas size (px)")
@click.option("--radius", type=int, default=540, help="Circle radius (px)")
@click.option("--nails", type=int, default=240, help="Number of nails")
def nails(out, canvas, radius, nails):  # pragma: no cover - thin wrapper
    """Render only the nail legend image.

    Useful to preview labeling / indexing before running an expensive solve.
    """
    os.makedirs(out, exist_ok=True)
    render_nail_map(
        os.path.join(out, "legend_nails.png"),
        canvas=canvas,
        radius=radius,
        n=nails,
        show_labels=True,
        show_indices=False,
    )
    click.echo(f"Nail map saved to {out}/legend_nails.png")


@main.command()
@click.option("-i", "--image", type=click.Path(exists=True), required=True, help="Input image")
@click.option("-o", "--out", type=click.Path(file_okay=False), required=True, help="Output directory")
@click.option("-c", "--canvas", type=int, default=1200, help="Canvas size (px)")
@click.option("-r", "--radius", type=int, default=None, help="Manual radius (px)")
@click.option("-n", "--nails", type=int, default=None, help="Manual nail count")
@click.option("--invert/--no-invert", default=True, help="Invert input image (default: True)")
@click.option("--auto-all/--no-auto-all", default=False, help="Auto-configure nails, radius, params")
@click.option("--auto-nails/--no-auto-nails", default=True, help="Automatically detect nail count from rim")
@click.option("--min-nails", type=int, default=20, help="Lower bound for auto nail detection")
@click.option("--max-nails", type=int, default=360, help="Upper bound for auto nail detection")
@click.option("--debug-detect/--no-debug-detect", default=False, help="Save nail detection debug plots")
@click.option("-s", "--steps", type=int, default=None, help="Fixed max steps (omit for auto)")
@click.option("--mask-inflate-frac", type=float, default=0.02, help="Inflate detected/used radius fraction before masking")
@click.option("--inner-hole-frac", type=float, default=0.0, help="Inner hole exclusion fraction of radius (0 disables)")
@click.option("--inner-hole-mode", type=click.Choice(["skip", "dampen"]), default="skip")
@click.option("--inner-hole-dampen", type=float, default=0.3, help="Score multiplier when mode=dampen")
@click.option("--refine-rounds", type=int, default=0, help="Number of refinement passes after baseline")
@click.option("--refine-mse-thresh", type=float, default=0.012, help="Stop refining when MSE <= threshold")
@click.option("--refine-alpha-scale", type=float, default=0.7, help="Alpha scale per refinement round")
@click.option("--avg-hits-stop", type=float, default=None, help="Optional stop when average nail hits reaches this (omit to disable)")
@click.option("--auto-inner-hole/--no-auto-inner-hole", default=False, help="Automatically detect inner hole (overrides manual inner-hole-frac/mode if strong)")
@click.option("--hole-debug/--no-hole-debug", default=False, help="Save hole radial profile debug plot")
@click.option("--save-every", type=int, default=200, help="Save preview every N steps (0=never)")
def run(
    image,
    out,
    canvas,
    radius,
    nails,
    invert,
    auto_all,
    auto_nails,
    min_nails,
    max_nails,
    debug_detect,
    steps,
    mask_inflate_frac,
    inner_hole_frac,
    inner_hole_mode,
    inner_hole_dampen,
    refine_rounds,
    refine_mse_thresh,
    refine_alpha_scale,
    avg_hits_stop,
    auto_inner_hole,
    hole_debug,
    save_every,
):
    """Execute the full string‑art pipeline on an input image.

    The solver either uses a fully automatic configuration (--auto-all) or
    respects manual overrides for radius / nail count / steps.
    """
    os.makedirs(out, exist_ok=True)

    if auto_all:
        # Auto infer geometry. Manual radius / nails arguments still override.
        params_dict = auto_config(
            image_path=image,
            canvas_size=canvas,
            invert=invert,
            auto_nails=auto_nails and (nails is None),
            verbose=True,
            min_nails=min_nails,
            max_nails=max_nails,
            debug_dir=(out if debug_detect else None),
        )
        if nails is not None:
            params_dict["nails"] = nails
        if radius is not None:
            params_dict["radius"] = radius
        params = SolverParams(**params_dict)
    else:
        # Manual / semi-manual path
        if nails is None:
            nails = 240
        if radius is None:
            radius = canvas // 2 - 20
        # Auto steps heuristic unless user passed --steps
        auto_steps_enabled = steps is None
        if auto_steps_enabled:
            estimated_steps = int(min(12000, max(800, 18 * nails)))
            max_steps = estimated_steps
        else:
            max_steps = steps
        params = SolverParams(
            canvas=canvas,
            radius=radius,
            nails=nails,
            invert=invert,
            auto_steps=auto_steps_enabled,
            max_steps=max_steps,
        )

    params.save_every = save_every
    # If user provided --steps explicitly while auto_all selected, enforce it.
    if steps is not None and params.auto_steps:
        params.auto_steps = False
        params.max_steps = steps

    # Preprocess input image to target grayscale board region.
    target_image = preprocess_image(
        image,
        canvas=params.canvas,
        radius=params.radius,
        invert=params.invert,
        apply_mask=params.apply_mask,
        outdir=out,
        rim="keep",
        n=params.nails,  # legacy arg name inside preprocess
        mask_inflate_frac=mask_inflate_frac,
        defer_mask=False,
    )

    # Inner hole settings (may be overwritten by auto detection below).
    params.inner_hole_frac = inner_hole_frac
    params.inner_hole_mode = inner_hole_mode
    params.inner_hole_dampen = inner_hole_dampen

    # Optional automatic inner hole detection.
    if auto_inner_hole:
        from .hole import estimate_inner_hole, auto_adjust_hole

        hole_estimate = estimate_inner_hole(target_image, params.radius)
        modified = auto_adjust_hole(params, hole_estimate)
        if hole_debug:
            try:  # Best-effort debug figure generation.
                import matplotlib.pyplot as plt

                height, width = target_image.shape
                center_x = center_y = height // 2
                grid_y, grid_x = np.indices(target_image.shape)
                radial_dist = np.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
                inside_mask = radial_dist <= params.radius
                values = target_image[inside_mask]
                radii = radial_dist[inside_mask]
                bins = np.linspace(0, params.radius, 512)
                bin_index = np.digitize(radii, bins) - 1
                radial_profile = np.zeros(len(bins), dtype=float)
                for i in range(len(bins)):
                    mask_i = bin_index == i
                    if mask_i.any():
                        radial_profile[i] = values[mask_i].mean()
                fig, ax = plt.subplots(figsize=(5, 3))
                norm_prof = (radial_profile - radial_profile.min()) / (radial_profile.max() - radial_profile.min() + 1e-6)
                ax.plot(bins / params.radius, norm_prof)
                ax.axvline(
                    hole_estimate.radius_fraction,
                    color="r",
                    ls="--",
                    label=f"hole≈{hole_estimate.radius_fraction:.3f} conf={hole_estimate.confidence:.2f}",
                )
                ax.set_xlabel("r / R")
                ax.set_ylabel("norm mean")
                ax.legend()
                fig.savefig(os.path.join(out, "hole_detect_debug.png"), dpi=130)
                plt.close(fig)
            except Exception:
                pass
        if modified:
            click.echo(
                f"[auto-hole] inner_hole_frac={params.inner_hole_frac:.3f} mode={params.inner_hole_mode}"
            )

    params.refine_rounds = refine_rounds
    # Dynamic refine threshold auto-selection if requested via <= 0.
    if refine_rounds > 0 and (refine_mse_thresh <= 0):
        estimated_mse = float(np.mean((target_image - target_image.mean()) ** 2))
        refine_mse_thresh = estimated_mse * 0.6
        click.echo(
            f"[auto-refine] set refine_mse_thresh≈{refine_mse_thresh:.4f} (est_mse={estimated_mse:.4f})"
        )
    params.refine_mse_thresh = refine_mse_thresh
    params.refine_alpha_scale = refine_alpha_scale
    params.avg_hits_stop = avg_hits_stop

    # Solve (baseline + optional refinement cycles).
    steps_out, metrics = solve_with_refinement(target_image, params, out)

    click.echo(
        f"Done. {len(steps_out)} steps (after refinement), metrics={metrics}"
    )
    click.echo(f"Outputs in {out}/")


if __name__ == "__main__":
    main()

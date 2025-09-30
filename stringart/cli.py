from __future__ import annotations
import click, os
from .geometry import generate_default_canvas, index_to_label, label_to_index
from .preprocess import preprocess_image
from .solver import SolverParams, greedy_solver
from .visualize import render_nail_map
import cv2

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """String-art generator CLI."""
    pass

@main.command()
@click.option("--image", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "outdir", type=click.Path(file_okay=False), default="out", show_default=True)
@click.option("--canvas", type=int, default=900, show_default=True, help="Working canvas (px)")
@click.option("--steps", type=int, default=None, help="If provided, run fixed number of steps (disables auto steps)")
@click.option("--auto-all/--manual", default=True, show_default=True,
              help="Decide invert, rim, suppression, nails, bootstrap, fast-mode, and auto-stopping automatically")
# manual overrides (used only when --manual)
@click.option("--nails", type=int, default=240, show_default=True)
@click.option("--radius", type=int, default=None)
@click.option("--invert/--no-invert", default=False, show_default=True)
@click.option("--rim", type=click.Choice(["keep","feather","erode"]), default="keep", show_default=True)
@click.option("--edge-margin", type=int, default=12, show_default=True)
@click.option("--feather", type=int, default=8, show_default=True)
@click.option("--suppress-nail-heads", type=int, default=0, show_default=True)
@click.option("--alpha", type=float, default=0.22, show_default=True)
@click.option("--thickness", type=int, default=1, show_default=True)
@click.option("--blur", type=float, default=0.6, show_default=True)
@click.option("--min-chord", type=float, default=30.0, show_default=True)
@click.option("--candidate-stride", type=int, default=3, show_default=True)
@click.option("--coarse-scale", type=int, default=6, show_default=True)
@click.option("--sample-stride", type=int, default=2, show_default=True)
@click.option("--topk-refine", type=int, default=24, show_default=True)
@click.option("--bootstrap", type=click.Choice(["none","star"]), default="star", show_default=True)
@click.option("--bootstrap-steps", type=int, default=600, show_default=True)
@click.option("--bootstrap-k", type=int, default=0, show_default=True, help="0=auto")
# auto-stop controls (used in both modes; values auto-set in auto-all)
@click.option("--auto-steps/--fixed-steps", default=True, show_default=True)
@click.option("--max-steps", type=int, default=4000, show_default=True)
@click.option("--coverage", type=float, default=0.87, show_default=True)
@click.option("--rel-improve", type=float, default=1e-4, show_default=True)
@click.option("--abs-improve", type=float, default=8e-7, show_default=True)
@click.option("--patience", type=int, default=60, show_default=True)
@click.option("--window", type=int, default=20, show_default=True)
@click.option("--save-every", type=int, default=0, show_default=True)
@click.option("--seed", type=int, default=1, show_default=True)
@click.option("--endpoint-taper", type=float, default=0.20, show_default=True, help="Fraction of each chord end to taper (0..1)")
@click.option("--angle-smooth", type=float, default=0.25, show_default=True, help="Penalty strength for sharp angle changes (0..1)")
def run(image, outdir, canvas, steps, auto_all, nails, radius, invert, rim, edge_margin, feather, suppress_nail_heads, endpoint_taper, angle_smooth,
        alpha, thickness, blur, min_chord, candidate_stride, coarse_scale, sample_stride, topk_refine,
        bootstrap, bootstrap_steps, bootstrap_k,
        auto_steps, max_steps, coverage, rel_improve, abs_improve, patience, window, save_every, seed):
    import os
    os.makedirs(outdir, exist_ok=True)
    from .autotune import auto_config
    from .preprocess import preprocess_image
    from .geometry import generate_default_canvas, generate_nails
    from .pattern_init import bootstrap_steps as make_boot
    from .solver import SolverParams, greedy_solver

    canvas, _ = generate_default_canvas(canvas, radius)

    if auto_all:
        ac = auto_config(image, canvas=canvas)
        nails = ac.nails
        radius = ac.radius
        invert = ac.invert
        rim = ac.rim
        edge_margin = ac.edge_margin
        feather = ac.feather
        suppress_nail_heads = ac.suppress_nail_heads
        min_chord = ac.min_chord_px
        candidate_stride = ac.candidate_stride
        coarse_scale = ac.coarse_scale
        sample_stride = ac.sample_stride
        topk_refine = ac.topk_refine
        bootstrap = "star"
        bootstrap_steps = ac.bootstrap_steps
        bootstrap_k = ac.bootstrap_k
        coverage = ac.coverage
        rel_improve = ac.rel_improve
        abs_improve = ac.abs_improve
        patience = ac.patience
        window = ac.window
    # keep user-specified endpoint_taper / angle_smooth (no tuple!)
    endpoint_taper = endpoint_taper
    angle_smooth = angle_smooth

    click.echo(f"[auto] nails={nails}  invert={invert}  rim={rim}  R≈{radius}  k_boot={bootstrap_k}")

    # preprocess
    target = preprocess_image(
        image_path=image, canvas=canvas, radius=radius, invert=invert, apply_mask=True, outdir=outdir,
        rim=rim, edge_margin_px=edge_margin, feather_px=feather, suppress_nail_heads_px=suppress_nail_heads, n=nails
    )

    # bootstrap steps
    init_steps = None
    if bootstrap == "star":
        k = bootstrap_k
        if k == 0:
            # fallback if user forced manual without providing k
            small = max(256, canvas // 3)
            import cv2
            tgt_small = cv2.resize(target, (small, small), interpolation=cv2.INTER_AREA)
            nails_xy_small = generate_nails(n=nails, center=(small//2, small//2), radius=int(small*0.45))
            from .pattern_init import choose_best_k_by_correlation
            ks = list(range(2, max(3, min(80, nails//3))))
            k, _ = choose_best_k_by_correlation(nails, ks, boots=min(nails*2, 800),
                                                target_small=tgt_small, nails_xy_small=nails_xy_small)
        init_steps = make_boot(nails, k, boots=bootstrap_steps)

    # If user explicitly provides --steps, force fixed steps mode
    if steps is not None:
        auto_steps = False
        max_steps = steps

    params = SolverParams(
        alpha=alpha, line_thickness_px=thickness, blur_sigma=blur,
        min_chord_px=min_chord, per_nail_max_hits=999999,
        canvas=canvas, radius=radius, nails=nails, start_label="A1",
        invert=invert, apply_mask=True, seed=seed, save_every=save_every,
        candidate_stride=candidate_stride, coarse_scale=coarse_scale,
        sample_stride=sample_stride, topk_refine=topk_refine,
        auto_steps=auto_steps, max_steps=max_steps, coverage=coverage,
        rel_improve=rel_improve, abs_improve=abs_improve, patience=patience, window=window,
        endpoint_taper=endpoint_taper, angle_smooth=angle_smooth
    )

    steps_out, metrics = greedy_solver(target, params, outdir, init_steps=init_steps)
    click.echo(f"[stringart] Done. Steps={len(steps_out)}  MSE={metrics['mse']:.6f}  PSNR={metrics['psnr']:.2f}dB  SSIM={metrics['ssim']:.4f}")



@main.command()
@click.option("--out", "outdir", type=click.Path(file_okay=False), default="out", show_default=True)
@click.option("--canvas", type=int, default=1200, show_default=True)
@click.option("--radius", type=int, default=None)
@click.option("--nails", type=int, default=240, show_default=True)
@click.option("--show-labels/--no-labels", default=True, show_default=True)
@click.option("--show-indices/--no-indices", default=False, show_default=True)
@click.option("--label-every", type=int, default=1, show_default=True)
def nails(outdir, canvas, radius, nails, show_labels, show_indices, label_every):
    """Render a labeled nail map preview PNG."""
    os.makedirs(outdir, exist_ok=True)
    render_nail_map(os.path.join(outdir, "nail_map.png"), canvas=canvas, radius=radius, n=nails,
                    show_labels=show_labels, show_indices=show_indices, label_every=label_every)
    click.echo(f"[stringart] Wrote {os.path.join(outdir,'nail_map.png')}")

@main.command()
@click.argument("value")
@click.option("--nails", type=int, default=240, show_default=True, help="Nail count to interpret labels")
def label(value: str, nails: int):
    """Convert between index and label for the given nail count."""
    s = value.strip()
    try:
        i = int(s)
        click.echo(index_to_label(i, n=nails))
        return
    except Exception:
        pass
    try:
        click.echo(str(label_to_index(s, n=nails)))
        return
    except Exception as e:
        raise click.ClickException(str(e))

if __name__ == "__main__":
    main()

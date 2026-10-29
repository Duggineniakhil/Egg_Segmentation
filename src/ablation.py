"""Ablation study: remove one component of the final classical pipeline at a time (dev split only).

Each variant is the final config (configs/t16_nest_cap4.yaml) with one override. All variants x all dev
images run in parallel across CPU cores. Writes outputs/ablation/:
  ablation.csv          one row per variant: metrics and the drop vs. the full pipeline
  per_image.csv         per-image IoU for every variant (for paired comparisons)
  fig_ablation.png      mIoU drop per removed component

    python src/ablation.py                 # all variants
    python src/ablation.py --workers 6
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from classical_seg import _merge, load_config, segment
from metrics import image_metrics, summarize

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "configs" / "t16_nest_cap4.yaml"
OUT = ROOT / "outputs" / "ablation"

# name -> (what is removed, override of the final pipeline)
VARIANTS = {
    "full": ("nothing (final pipeline)", {}),
    "no_smoothness": ("smoothness channel", {"sweep": {"extra_channels": ["lab_whiteness"]}}),
    "no_lab_whiteness": ("Lab-whiteness channel", {"sweep": {"extra_channels": ["smoothness"]}}),
    "gray_only": ("both extra channels (gray only)", {"sweep": {"extra_channels": []}}),
    "no_dark_polarity": ("dark-polarity sweep (egg darker than background)", {"sweep": {"polarity": "up"}}),
    "no_edge_cut": ("Canny edge cut", {"sweep": {"edge_cut": False}}),
    "edge_support_score": ("shape x contrast score (back to ellipse x edge support)",
                           {"sweep": {"quality": "edge_support", "min_quality": 0.35}}),
    "no_border_penalty": ("border penalty", {"sweep": {"border_penalty": 1.0}}),
    "no_nested_upgrade": ("nested upgrade", {"sweep": {"nest_ratio": 0.0}}),
    "no_nest_cap": ("4x cap on the nested upgrade", {"sweep": {"nest_max_growth": 0}}),
    "no_grabcut": ("GrabCut refinement", {"refine": {"grabcut_iters": 0}}),
    "no_blur": ("Gaussian blur", {"blur": {"type": "none"}}),
    "no_forced_egg": ("at-least-one-egg rule", {"sweep": {"min_eggs": 0}}),
}
# v2 candidates (additions, not removals): run with --variants full v2_...
EXTRA = {
    "v2_complete": ("+ ellipse completion", {"refine": {"complete": True}}),
    "v2_complete_g15": ("+ ellipse completion, max growth 1.5", {"refine": {"complete": True, "complete_max_growth": 1.5}}),
    "v2_complete_g25": ("+ ellipse completion, max growth 2.5", {"refine": {"complete": True, "complete_max_growth": 2.5}}),
    "v2_complete_sup35": ("+ ellipse completion, >=35% outline support", {"refine": {"complete": True, "complete_min_support": 0.35}}),
}
EXTRA.update({
    "v2_rel03": ("+ ellipse completion (relative edges 0.3)",
                 {"refine": {"complete": True, "complete_mode": "relative", "complete_rel": 0.3, "complete_max_growth": 2.5}}),
    "v2_rel05": ("+ ellipse completion (relative edges 0.5)",
                 {"refine": {"complete": True, "complete_mode": "relative", "complete_rel": 0.5, "complete_max_growth": 2.5}}),
    "v2_rel03_g2": ("+ ellipse completion (relative 0.3, growth 2.0)",
                    {"refine": {"complete": True, "complete_mode": "relative", "complete_rel": 0.3, "complete_max_growth": 2.0}}),
    "v2_rel03_nogc": ("+ ellipse completion (relative 0.3), no GrabCut",
                      {"refine": {"complete": True, "complete_mode": "relative", "complete_rel": 0.3,
                                  "complete_max_growth": 2.5, "grabcut_iters": 0}}),
})
_V2 = {"complete": True, "complete_mode": "relative", "complete_rel": 0.3, "complete_max_growth": 2.5}
EXTRA.update({
    "v2_rel03_chroma8": ("+ completion, added pixels chroma-limited (>=8)", {"refine": {**_V2, "complete_chroma": 8}}),
    "v2_rel03_chroma12": ("+ completion, added pixels chroma-limited (>=12)", {"refine": {**_V2, "complete_chroma": 12}}),
})
VARIANTS_ALL = {**VARIANTS, **EXTRA}

_PIPES = None


def _init():
    global _PIPES
    base = load_config(FINAL)["pipeline"]
    _PIPES = {k: _merge(base, ov) for k, (_, ov) in VARIANTS_ALL.items()}


def _run(job):
    variant, stem = job
    img = cv2.imread(str(ROOT / "data" / "images" / f"{stem}.jpg"))
    gt = cv2.imread(str(ROOT / "outputs" / "sam_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    pred = segment(img, _PIPES[variant])
    return {"variant": variant, "image": stem, **image_metrics(gt, pred)}


def plot(t, path):
    """Horizontal bars: mIoU drop when each component is removed (largest first)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = t[t.variant != "full"].sort_values("delta_miou")
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(d) + 1.0))
    y = range(len(d))
    ax.barh(list(y), d.delta_miou, height=0.62, color="#2a78d6")
    for yy, v in zip(y, d.delta_miou):
        ax.text(v - 0.004 if v < 0 else v + 0.004, yy, f"{v:+.3f}", va="center", ha="right" if v < 0 else "left",
                fontsize=8, color="#52514e")
    ax.set_yticks(list(y), d.removed, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#52514e", lw=1)
    ax.set_xlim(d.delta_miou.min() - 0.05, max(0.03, d.delta_miou.max() + 0.03))
    ax.set_facecolor("#fcfcfb")
    fig.suptitle("Ablation (dev): mIoU change when one component is removed", x=0.01, ha="left",
                 fontsize=10, color="#0b0b0b")
    ax.set_title(f"full pipeline: {t[t.variant == 'full'].miou.iloc[0]:.3f} mIoU", loc="left", fontsize=8,
                 color="#52514e")
    ax.set_xlabel("change in dev mIoU", fontsize=9, color="#52514e")
    ax.grid(True, axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--plot-only", action="store_true", help="redraw fig_ablation.png from ablation.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(pd.read_csv(OUT / "ablation.csv"), OUT / "fig_ablation.png")
        return

    dev = json.loads((ROOT / "data" / "splits.json").read_text())["dev"]
    jobs = [(v, s) for v in args.variants for s in dev]
    rows = []
    with Pool(args.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(_run, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                print(f"  {k + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    tag = "" if set(args.variants) == set(VARIANTS) else "_subset"   # a partial run never overwrites the full table
    df.to_csv(OUT / f"per_image{tag}.csv", index=False)

    if "full" not in args.variants:
        raise SystemExit("include 'full' so deltas can be computed")
    full = df[df.variant == "full"].set_index("image").iou
    table = []
    for v in args.variants:
        d = df[df.variant == v]
        s = summarize(d.to_dict("records"))
        paired = d.set_index("image").iou.reindex(full.index) - full
        table.append({"variant": v, "removed": VARIANTS_ALL[v][0], **{k: round(x, 4) for k, x in s.items()},
                      "delta_miou": round(s["miou"] - full.mean(), 4),
                      "images_worse": int((paired < -0.05).sum()), "images_better": int((paired > 0.05).sum())})
    t = pd.DataFrame(table)
    t.to_csv(OUT / f"ablation{tag}.csv", index=False)
    if not tag:
        plot(t, OUT / "fig_ablation.png")
    print(t[["variant", "miou", "delta_miou", "mboundary_f1", "inst_f1", "count_acc",
             "images_worse", "images_better"]].to_string(index=False))


if __name__ == "__main__":
    main()

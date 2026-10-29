"""Task 3: evaluate classical predictions (outputs/cv_masks) against SAM pseudo ground truth.

    python src/run_experiment.py configs/t16_nest_cap4.yaml --split eval --save-masks
    python src/run_experiment.py configs/t16_nest_cap4.yaml --save-masks --no-log
    python src/evaluate.py

Writes to outputs/eval/:
  summary.json         metrics for dev, eval and all images
  per_image.csv        metrics, image attributes and failure category for each image
  per_egg.csv          one row per SAM egg: best-match IoU, size, contrast, border/touching flags
  fig_*.png            analysis plots
  gallery_*.jpg        examples of each failure category
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from metrics import image_metrics, summarize
from run_experiment import error_overlay

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR, GT_DIR, PRED_DIR = ROOT / "data" / "images", ROOT / "outputs" / "sam_masks", ROOT / "outputs" / "cv_masks"
OUT = ROOT / "outputs" / "eval"

# reference palette (light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DEV, EVAL = "#2a78d6", "#eb6834"

CATEGORY_ORDER = ["good", "under-segmented", "leak", "missed / wrong object", "mixed"]


# ----------------------------------------------------------------------------- measurements

def hd95(gt, pred):
    """95th-percentile symmetric Hausdorff distance between mask boundaries, in pixels."""
    g, p = gt > 0, pred > 0
    if not g.any() or not p.any():
        return np.nan
    eg = g ^ ndimage.binary_erosion(g)
    ep = p ^ ndimage.binary_erosion(p)
    dg = ndimage.distance_transform_edt(~eg)
    dp = ndimage.distance_transform_edt(~ep)
    return float(np.percentile(np.r_[dp[eg], dg[ep]], 95))


def ring(mask, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2)
    return (cv2.dilate(mask.astype(np.uint8), k) > 0) & ~mask


def image_attributes(img, gt):
    """What makes an image easy or hard: contrast with the surroundings, white or textured background, backlighting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    eggs = gt > 0
    r = ring(eggs, 40)
    whitish = (lab[..., 0] > 170) & (chroma < 12)
    return {
        "egg_area_frac": eggs.mean(),
        "contrast": gray[eggs].mean() - gray[r].mean(),      # < 0: egg darker than surroundings (backlit)
        "bg_white_frac": whitish[r].mean(),                    # share of nearby background that is white
        "bg_texture": lap[r].mean(),
        "egg_brightness": gray[eggs].mean(),
    }


def egg_rows(stem, img, gt, pred):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gt.shape
    rows = []
    for g in np.unique(gt)[1:]:
        m = gt == g
        best = 0.0
        for p in np.unique(pred[m]):
            if p:
                pm = pred == p
                best = max(best, (m & pm).sum() / (m | pm).sum())
        ys, xs = np.nonzero(m)
        others = (gt > 0) & ~m
        rows.append({
            "image": stem, "egg": int(g), "best_iou": best, "area_frac": m.mean(),
            "contrast": gray[m].mean() - gray[ring(m, 15)].mean(),
            "touches_border": bool(ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1),
            "touches_egg": bool((ring(m, 3) & others).any()),
        })
    return rows


def categorize(r):
    if r["iou"] >= 0.8:
        return "good"
    if r["recall"] < 0.3:
        return "missed / wrong object"
    if r["precision"] >= 0.9 and r["recall"] < 0.8:
        return "under-segmented"
    if r["precision"] < 0.75 and r["recall"] >= 0.75:
        return "leak"
    return "mixed"


# ----------------------------------------------------------------------------- plots

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def binned_mean(ax, x, y, bins):
    idx = np.digitize(x, bins)
    cx, cy = [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() >= 3:
            cx.append(x[sel].mean())
            cy.append(y[sel].mean())
    ax.plot(cx, cy, color=INK, linewidth=2, label="binned mean")


def make_plots(img_df, egg_df):
    # 1) IoU distribution, dev vs eval
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, 1, 21)
    for split, col in (("dev", DEV), ("eval", EVAL)):
        v = img_df[img_df.split == split].iou
        ax.hist(v, bins=bins, histtype="step", linewidth=2, color=col, density=True,
                label=f"{split} (n={len(v)}, mean {v.mean():.3f})")
    style(ax, "Per-image IoU vs SAM masks", "IoU", "density")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "fig_iou_hist.png")

    # 2) precision vs recall per image: where the error comes from
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.axhspan(0.9, 1.02, xmin=0, xmax=0.8 / 1.02, color=GRID, alpha=0.5, linewidth=0)
    for split, col in (("dev", DEV), ("eval", EVAL)):
        d = img_df[img_df.split == split]
        ax.scatter(d.recall, d.precision, s=42, color=col, edgecolor=SURFACE, linewidth=1.5, label=split, zorder=3)
    ax.text(0.03, 0.965, "under-segmented: everything predicted is egg,\nbut part of the egg is missing",
            fontsize=7.5, color=INK2, va="top")
    ax.text(0.98, 0.05, "leak: egg found, plus background", fontsize=7.5, color=INK2, ha="right")
    ax.text(0.03, 0.05, "missed /\nwrong object", fontsize=7.5, color=INK2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style(ax, "Pixel precision vs recall per image", "recall (share of SAM egg pixels found)",
          "precision (share of predicted pixels that are egg)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left")
    save(fig, "fig_precision_recall.png")

    # 3) failure categories
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    counts = img_df.groupby(["category", "split"]).size().unstack(fill_value=0).reindex(CATEGORY_ORDER, fill_value=0)
    y = np.arange(len(CATEGORY_ORDER))
    hgt = 0.38
    for off, split, col in ((-hgt / 2 - 0.01, "dev", DEV), (hgt / 2 + 0.01, "eval", EVAL)):
        vals = counts.get(split, pd.Series(0, index=CATEGORY_ORDER)).values
        ax.barh(y + off, vals, height=hgt, color=col, label=split)
        for yy, v in zip(y + off, vals):
            ax.text(v + 0.4, yy, str(v), va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, CATEGORY_ORDER)
    ax.invert_yaxis()
    style(ax, "Failure categories (images)", "number of images", "")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    save(fig, "fig_categories.png")

    # 4) per-egg IoU vs contrast with surroundings
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(egg_df.contrast, egg_df.best_iou, s=26, color=DEV, alpha=0.75, edgecolor=SURFACE, linewidth=1)
    binned_mean(ax, egg_df.contrast.values, egg_df.best_iou.values, np.linspace(-80, 160, 13))
    ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(-4, 0.04, "egg darker than\nsurroundings", fontsize=7.5, color=INK2, ha="right")
    style(ax, "Egg IoU vs contrast with its surroundings (all images)",
          "egg brightness minus 15 px ring around it (gray levels)", "best-match IoU")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    save(fig, "fig_iou_vs_contrast.png")

    # 5) per-image IoU vs white background share
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(img_df.bg_white_frac, img_df.iou, s=30, color=DEV, alpha=0.8, edgecolor=SURFACE, linewidth=1)
    binned_mean(ax, img_df.bg_white_frac.values, img_df.iou.values, np.linspace(0, 1, 8))
    style(ax, "Image IoU vs how white the background around the eggs is",
          "share of white pixels in a 40 px ring around the eggs", "IoU")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    save(fig, "fig_iou_vs_white_bg.png")

    # 6) per-egg IoU vs size
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(egg_df.area_frac * 100, egg_df.best_iou, s=26, color=DEV, alpha=0.75, edgecolor=SURFACE, linewidth=1)
    binned_mean(ax, egg_df.area_frac.values * 100, egg_df.best_iou.values, np.linspace(0, 40, 9))
    style(ax, "Egg IoU vs egg size", "egg area (% of image)", "best-match IoU")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    save(fig, "fig_iou_vs_size.png")

    # 7) IoU by scene
    g = img_df.groupby("scene").iou
    order = g.median().sort_values().index
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(order) + 1.2))
    for i, sc in enumerate(order):
        d = img_df[img_df.scene == sc]
        col = EVAL if (d.split == "eval").all() else DEV
        ax.scatter(d.iou, np.full(len(d), i), s=24, color=col, alpha=0.85, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.plot([d.iou.median()] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    style(ax, "IoU by scene (dot = image, bar = median; orange = eval scene)", "IoU", "")
    save(fig, "fig_iou_by_scene.png")


def gallery(img_df, category, path, k=6, tile=240):
    d = img_df[img_df.category == category].sort_values("iou").head(k)
    if d.empty:
        return
    rows = []
    for _, r in d.iterrows():
        img = cv2.imread(str(IMG_DIR / f"{r.image}.jpg"))
        gt = cv2.imread(str(GT_DIR / f"{r.image}.png"), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(PRED_DIR / f"{r.image}.png"), cv2.IMREAD_GRAYSCALE)
        sam = img.copy()
        cnts, _ = cv2.findContours((gt > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(sam, cnts, -1, (255, 0, 255), 3)
        a = cv2.resize(sam, (tile, tile))
        b = cv2.resize(error_overlay(img, gt, pred), (tile, tile))
        cv2.putText(a, f"#{r.image.split('_')[-1]} ({r.split})", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(b, f"IoU {r.iou:.2f} P {r.precision:.2f} R {r.recall:.2f}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        rows.append(np.hstack([a, b]))
    while len(rows) % 2:
        rows.append(np.full_like(rows[0], 255))
    sheet = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
    legend = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(legend, "left: image + SAM outline (magenta)   right: green = correct, red = false positive, "
                "blue = missed, yellow = predicted outline", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.imwrite(str(path), np.vstack([legend, sheet]))


# ----------------------------------------------------------------------------- main

def scene_names(splits):
    """Readable scene label: manual override name if any, else the frame numbers it contains."""
    overrides = splits["params"].get("scene_overrides", {})
    by_frame = {n: name for name, ns in overrides.items() for n in ns}
    groups = {}
    for stem, g in splits["groups"].items():
        groups.setdefault(g, []).append(int(stem.split("_")[-1]))
    names = {}
    for g, frames in groups.items():
        named = {by_frame[f] for f in frames if f in by_frame}
        names[g] = named.pop() if len(named) == 1 else "frames " + ",".join(map(str, sorted(frames)[:4])) + \
            ("..." if len(frames) > 4 else "")
    return {stem: names[g] for stem, g in splits["groups"].items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = json.loads((ROOT / "data" / "splits.json").read_text())
    scenes = scene_names(splits)
    img_rows, eggs = [], []
    for split in ("dev", "eval"):
        for stem in splits[split]:
            img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
            gt = cv2.imread(str(GT_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(PRED_DIR / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if pred is not None and pred.dtype != np.uint8:
                raise SystemExit(f"{stem}.png is {pred.dtype}; re-save masks as uint8 label maps")
            if pred is None:
                raise SystemExit(f"missing {PRED_DIR / stem}.png: run run_experiment.py --save-masks first")
            m = image_metrics(gt, pred)
            img_rows.append({"image": stem, "split": split, "scene": scenes[stem], **m, "hd95": hd95(gt, pred),
                             **image_attributes(img, gt)})
            eggs += [{"split": split, **e} for e in egg_rows(stem, img, gt, pred)]
    img_df = pd.DataFrame(img_rows)
    img_df["category"] = img_df.apply(categorize, axis=1)
    egg_df = pd.DataFrame(eggs)
    img_df.to_csv(OUT / "per_image.csv", index=False)
    egg_df.to_csv(OUT / "per_egg.csv", index=False)

    summary = {}
    for name, d in (("dev", img_df[img_df.split == "dev"]), ("eval", img_df[img_df.split == "eval"]), ("all", img_df)):
        s = summarize(d.to_dict("records"))
        s["mhd95_px"] = float(d.hd95.mean())
        s["n_images"] = len(d)
        s["n_eggs"] = int(d.n_gt.sum())
        s["categories"] = d.category.value_counts().to_dict()
        summary[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    make_plots(img_df, egg_df)
    for cat in CATEGORY_ORDER[1:]:
        gallery(img_df, cat, OUT / f"gallery_{cat.split()[0].replace('-', '_')}.jpg")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

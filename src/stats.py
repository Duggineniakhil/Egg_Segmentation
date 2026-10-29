"""How certain are the Task 2/3 numbers? Scene-level bootstrap and scene-grouped folds.

Images of the same scene are strongly correlated (same background, lighting, eggs), so resampling
single images would understate the uncertainty. Here whole *scenes* are resampled (cluster bootstrap).
Uses outputs/eval/per_image.csv from src/evaluate.py. Writes outputs/stats/:
  bootstrap.csv       mean and 95% CI per metric, for dev / eval / all, scene-level and image-level
  folds.csv           final algorithm per scene-grouped fold (5 folds over all 110 images)
  per_scene.csv       mean IoU per scene
  fig_ci.png          CIs for dev / eval / all

    python src/stats.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stats"
B = 10000
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"dev": "#2a78d6", "eval": "#eb6834", "all": "#1baf7a"}
METRICS = {"miou": "iou", "mdice": "dice", "mboundary_f1": "boundary_f1", "count_acc": "count_correct"}


def stat(d):
    """Mean metrics of a set of images; instance F1 pooled over eggs, as in metrics.summarize."""
    out = {k: d[c].mean() for k, c in METRICS.items()}
    tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
    out["inst_f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return out


def bootstrap(d, by_scene, rng):
    groups = d.groupby("scene").indices if by_scene else {i: [i] for i in range(len(d))}
    keys = list(groups)
    samples = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[k]] for k in pick])
        samples.append(stat(d.iloc[idx]))
    return pd.DataFrame(samples)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img = pd.read_csv(ROOT / "outputs" / "eval" / "per_image.csv")
    rng = np.random.default_rng(0)

    rows = []
    for split in ("dev", "eval", "all"):
        d = (img if split == "all" else img[img.split == split]).reset_index(drop=True)
        point = stat(d)
        for level in ("scene", "image"):
            bs = bootstrap(d, level == "scene", rng)
            for m in point:
                lo, hi = np.percentile(bs[m], [2.5, 97.5])
                rows.append({"split": split, "resample": level, "metric": m, "n_images": len(d),
                             "n_scenes": d.scene.nunique(), "mean": round(point[m], 4),
                             "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / "bootstrap.csv", index=False)

    # scene-grouped 5 folds over all images. No parameters are re-fitted, so this measures how much the
    # result varies between different sets of scenes, not the effect of tuning.
    folds = []
    for k, (_, test) in enumerate(GroupKFold(n_splits=5).split(img, groups=img.scene)):
        d = img.iloc[test]
        folds.append({"fold": k + 1, "n_images": len(d), "n_scenes": d.scene.nunique(),
                      **{m: round(v, 4) for m, v in stat(d).items()}})
    fd = pd.DataFrame(folds)
    fd.to_csv(OUT / "folds.csv", index=False)

    ps = img.groupby(["scene", "split"]).agg(n=("iou", "size"), miou=("iou", "mean")).reset_index()
    ps.sort_values("miou").round(3).to_csv(OUT / "per_scene.csv", index=False)

    # figure: scene-level 95% CIs
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    for ax, (m, title) in zip(axes, [("miou", "mean IoU"), ("mdice", "mean Dice"), ("mboundary_f1", "boundary F1")]):
        labels = []
        for i, split in enumerate(("dev", "eval", "all")):
            # bt["resample"], not bt.resample: that name is a DataFrame method
            r = bt[(bt["split"] == split) & (bt["resample"] == "scene") & (bt["metric"] == m)].iloc[0]
            ax.errorbar(i, r["mean"], yerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]], fmt="o",
                        color=COLORS[split], ms=8, capsize=5, lw=2)
            ax.text(i + 0.12, r["mean"], f"{r['mean']:.3f}", va="center", fontsize=8, color=INK2)
            labels.append(f"{split}\n{int(r.n_scenes)} scenes")
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_xlim(-0.5, 2.7)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Final algorithm vs SAM: mean and 95% CI (scene-level bootstrap)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ci.png", dpi=160, facecolor=SURFACE)

    show = bt[bt.metric.isin(["miou", "mdice", "inst_f1"])].copy()
    show["ci"] = show.apply(lambda r: f"[{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
    print(show.pivot_table(index=["split", "metric"], columns="resample", values="ci", aggfunc="first").to_string())
    print(fd.to_string(index=False))
    print(f"folds: mIoU {fd.miou.mean():.3f} +- {fd.miou.std():.3f}")
    (OUT / "summary.json").write_text(json.dumps({"fold_miou_mean": round(fd.miou.mean(), 4),
                                                   "fold_miou_std": round(fd.miou.std(), 4)}, indent=1))


if __name__ == "__main__":
    main()

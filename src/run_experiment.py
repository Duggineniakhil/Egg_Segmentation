"""Run one classical-segmentation trial and log it.

    python src/run_experiment.py configs/t01_gray_otsu.yaml            # dev split (default)
    python src/run_experiment.py configs/t07_final.yaml --split eval   # eval: final model only

Writes outputs/experiments/<name>/{per_image.csv, worst.jpg}. Dev runs also append a row
to outputs/experiments/log.csv (the experiment log for the report).
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from classical_seg import load_config, segment
from metrics import image_metrics, summarize

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"
GT_DIR = ROOT / "outputs" / "sam_masks"
EXP_DIR = ROOT / "outputs" / "experiments"
LOG_COLS = ["name", "base", "change", "why", "split", "n_images", "miou", "mdice", "mprecision", "mrecall",
            "mboundary_f1", "inst_precision", "inst_recall", "inst_f1", "count_acc", "decision", "config"]


def error_overlay(img, gt, pred):
    """Green = agreed egg pixels, red = false positive, blue = missed (BGR)."""
    vis = (img * 0.5).astype(np.uint8)
    g, p = gt > 0, pred > 0
    vis[g & p] = (0.5 * img[g & p] + [0, 110, 0]).clip(0, 255)
    vis[~g & p] = (0.5 * img[~g & p] + [0, 0, 150]).clip(0, 255)
    vis[g & ~p] = (0.5 * img[g & ~p] + [150, 0, 0]).clip(0, 255)
    for lab in (pred,):   # predicted instance outlines in yellow
        for i in np.unique(lab)[1:]:
            cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
    return vis


def worst_sheet(df, preds, k=8, tile=256):
    rows = []
    for stem in df.nsmallest(k, "iou")["image"]:
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        gt = cv2.imread(str(GT_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        ov = cv2.resize(error_overlay(img, gt, preds[stem]), (tile, tile))
        im = cv2.resize(img, (tile, tile))
        r = df.set_index("image").loc[stem]
        cv2.putText(im, f"#{stem.split('_')[-1]} IoU {r.iou:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(ov, f"gt {int(r.n_gt)} pred {int(r.n_pred)}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        rows.append(np.hstack([im, ov]))
    while len(rows) % 2:
        rows.append(np.zeros_like(rows[0]))
    return np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--decision", default="", help="keep / discard, filled into the log")
    ap.add_argument("--save-masks", action="store_true", help="write predicted label maps to outputs/cv_masks/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stems = json.loads((ROOT / "data" / "splits.json").read_text())[args.split]
    out = EXP_DIR / (cfg["name"] + ("" if args.split == "dev" else "_eval"))
    out.mkdir(parents=True, exist_ok=True)

    rows, preds = [], {}
    for stem in stems:
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        gt = cv2.imread(str(GT_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        pred = segment(img, cfg["pipeline"])
        preds[stem] = pred
        rows.append({"image": stem, **image_metrics(gt, pred)})
        if args.save_masks:
            (ROOT / "outputs" / "cv_masks").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ROOT / "outputs" / "cv_masks" / f"{stem}.png"), pred)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    cv2.imwrite(str(out / "worst.jpg"), worst_sheet(df, preds))
    s = summarize(rows)
    print(f"{cfg['name']} [{args.split}, {len(stems)} images]")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in s.items()))

    if args.split == "dev":
        log = EXP_DIR / "log.csv"
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_COLS)
            if new:
                w.writeheader()
            w.writerow({"name": cfg["name"], "base": cfg.get("base", ""), "change": cfg.get("change", ""),
                        "why": cfg.get("why", ""), "split": args.split, "n_images": len(stems),
                        **{k: round(v, 4) for k, v in s.items()}, "decision": args.decision,
                        "config": json.dumps(cfg["pipeline"], separators=(",", ":"))})


if __name__ == "__main__":
    main()

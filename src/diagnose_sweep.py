"""Diagnose the sweep stage on dev: is a failure caused by candidate generation or by selection?

For each image, oracle_iou = best IoU between any single candidate and any GT egg. If the oracle
is high but the final IoU is low, selection picked the wrong blob; if the oracle is low too, the
egg never came out of the sweep as a clean blob.

    python src/diagnose_sweep.py configs/t05_sweep.yaml
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from classical_seg import blur, load_config, score_map, sweep_candidates, sweep_channels

ROOT = Path(__file__).resolve().parents[1]

cfg = load_config(sys.argv[1])
p = cfg["pipeline"]
per = pd.read_csv(ROOT / "outputs" / "experiments" / cfg["name"] / "per_image.csv").set_index("image")
rows = []
for stem in json.loads((ROOT / "data" / "splits.json").read_text())["dev"]:
    img = cv2.imread(str(ROOT / "data" / "images" / f"{stem}.jpg"))
    gt = cv2.imread(str(ROOT / "outputs" / "sam_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    x = blur(img, p["blur"])
    cands = sweep_candidates(x, sweep_channels(x, score_map(x, p["score"]), p), p["sweep"], return_candidates=True)
    best = 0.0
    for g in np.unique(gt)[1:]:
        gm = gt == g
        for cd in cands:
            best = max(best, (gm & cd["mask"]).sum() / (gm | cd["mask"]).sum())
    rows.append({"image": stem, "final_iou": per.loc[stem, "iou"], "oracle_iou": best, "n_cands": len(cands)})
df = pd.DataFrame(rows)
bad = df[df.final_iou < 0.5]
print(f"images with final IoU < 0.5: {len(bad)}")
print(f"  of these, oracle IoU >= 0.8 (selection problem):  {(bad.oracle_iou >= 0.8).sum()}")
print(f"  of these, oracle IoU <  0.8 (generation problem): {(bad.oracle_iou < 0.8).sum()}")
print(f"mean oracle IoU over all dev: {df.oracle_iou.mean():.3f}")
print(bad.sort_values("oracle_iou").to_string(index=False))

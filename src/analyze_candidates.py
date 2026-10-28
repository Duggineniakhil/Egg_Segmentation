"""Which candidate features separate real eggs from background blobs? (dev only)

For every sweep candidate on dev: compute hand-designed features and its best IoU with a SAM
egg. Then report, per feature, the AUC between eggs (IoU >= 0.8) and non-eggs (IoU < 0.3).
AUC 0.5 = useless; near 1 (or near 0) = separates well. This guides the hand-built quality
score; nothing is trained.

    python src/analyze_candidates.py configs/t08_smoothness.yaml
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from classical_seg import blur, candidate_features, load_config, score_map, sweep_candidates, sweep_channels

ROOT = Path(__file__).resolve().parents[1]

cfg = load_config(sys.argv[1])
p = cfg["pipeline"]
rows = []
for stem in json.loads((ROOT / "data" / "splits.json").read_text())["dev"]:
    img = cv2.imread(str(ROOT / "data" / "images" / f"{stem}.jpg"))
    gt = cv2.imread(str(ROOT / "outputs" / "sam_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    x = blur(img, p["blur"])
    for cd in sweep_candidates(x, sweep_channels(x, score_map(x, p["score"]), p), p["sweep"], return_candidates=True):
        iou = max(((gt == g) & cd["mask"]).sum() / ((gt == g) | cd["mask"]).sum() for g in np.unique(gt)[1:])
        rows.append({"image": stem, "iou": iou, "quality": cd["quality"], "e_iou": cd["e_iou"],
                     "support": cd["support"], "solidity": cd["solidity"], **candidate_features(img, cd["mask"])})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "outputs" / "experiments" / f"{cfg['name']}_candidates.csv", index=False)
pos, neg = df[df.iou >= 0.8], df[df.iou < 0.3]
print(f"{len(df)} candidates: {len(pos)} eggs (IoU>=0.8), {len(neg)} non-eggs (IoU<0.3)")
y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
for f in [c for c in df.columns if c not in ("image", "iou")]:
    auc = roc_auc_score(y, np.r_[pos[f], neg[f]])
    print(f"  {f:14s} AUC {auc:.3f}   egg median {pos[f].median():8.3f}   non-egg median {neg[f].median():8.3f}")

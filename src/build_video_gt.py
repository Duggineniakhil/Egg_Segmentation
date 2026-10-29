"""Turn the SAM video detections (outputs/video_gt/meta.json + masks/) into identity-labelled ground truth.

Which egg is which was established by watching the video frame by frame (close-ups of the 30 s handover in
outputs/tracking_debug/). Egg A is the raw egg, egg B the boiled one:
  - until frame 724, B sits still at about (157, 219); A starts on the left, is picked up (~476), spun on
    the right (~557-649), then picked up again (~650) and held at the far left;
  - frames 725-750: one hand holds both; A is the left one;
  - from frame 751, A stays at the far left (about (71, 194)) and B is carried right (~726-808) and spun.
These rules are applied to every sampled frame and the result is drawn on gt_review.jpg for checking.
Detections that are neither egg (none were found) would get gt_id "ignore".

    python src/build_video_gt.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "outputs" / "video_gt"
B_SPOT, A_SPOT, NEAR = (157, 219), (71, 194), 45
MAX_EGG_W = 140
HELD = {"A": [(476, 556), (650, 748)], "B": [(726, 808)]}
# frame -> {box number k: "A" | "B" | "ignore"} for anything the rules get wrong (checked on gt_review_*.jpg)
MANUAL = {
    746: {1: "A", 2: "ignore", 3: "B"},   # k2 is one box around both eggs; SAM's mask for B (k3) is only 10 px
}


def assign(f, dets):
    """dets: list of dicts with 'k' and 'c'. Returns {k: id}."""
    if f in MANUAL:
        return MANUAL[f]
    near = lambda d, p: np.hypot(d["c"][0] - p[0], d["c"][1] - p[1]) <= NEAR
    out = {}
    # a box much wider than one egg (~85 px) is Grounding DINO grouping eggs and hand together: not one egg
    for d in dets:
        if d["box"][2] - d["box"][0] > MAX_EGG_W:
            out[d["k"]] = "ignore"
    dets = [d for d in dets if d["k"] not in out]
    if f <= 724:
        b = [d for d in dets if near(d, B_SPOT)]
        for d in dets:
            out[d["k"]] = "B" if b and d is b[0] else "A"
    elif f <= 750:
        if len(dets) == 1:   # one egg visible: A is held at the far left, B starts at its spot (x ~157)
            out[dets[0]["k"]] = "A" if dets[0]["c"][0] < 110 else "B"
        else:
            for i, d in enumerate(sorted(dets, key=lambda d: d["c"][0])):
                out[d["k"]] = "A" if i == 0 else "B"
    else:
        a = [d for d in dets if near(d, A_SPOT)]
        for d in dets:
            out[d["k"]] = "A" if a and d is a[0] else "B"
    return out


def main():
    meta = {int(k): v for k, v in json.loads((GT / "meta.json").read_text()).items()}
    rows = []
    for f, dets in sorted(meta.items()):
        lab = cv2.imread(str(GT / "masks" / f"{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
        for d in dets:
            ys, xs = np.nonzero(lab == d["k"])
            # mask centroid; box centre if SAM's mask is a sliver (it failed on that box)
            d["c"] = (float(xs.mean()), float(ys.mean())) if len(xs) >= 200 else ((d["box"][0] + d["box"][2]) / 2,
                                                                                (d["box"][1] + d["box"][3]) / 2)
        ids = assign(f, dets)
        for d in dets:
            g = ids[d["k"]]
            held = any(a <= f <= b for a, b in HELD.get(g, []))
            x0, y0, x1, y1 = d["box"]
            rows.append({"frame": f, "k": d["k"], "gt_id": g, "cx": round(d["c"][0], 1), "cy": round(d["c"][1], 1),
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1, "area": d["area"], "held": held})
    df = pd.DataFrame(rows)
    df.to_csv(GT / "gt_tracks.csv", index=False)
    dup = df.groupby(["frame", "gt_id"]).size()
    print(f"{df.frame.nunique()} frames, {len(df)} GT eggs: {df.gt_id.value_counts().to_dict()}, "
          f"held: {int(df.held.sum())}; frames with a duplicated identity: {int((dup > 1).sum())}")

    # review sheet: every sampled frame with A/B drawn on it
    cap = cv2.VideoCapture(str(ROOT / "data" / "video" / "egg_video.mp4"))
    col = {"A": (214, 120, 42), "B": (52, 104, 235), "ignore": (128, 128, 128)}
    tiles = []
    for f in sorted(meta):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        for _, r in df[df.frame == f].iterrows():
            c = col[r.gt_id]
            cv2.rectangle(im, (int(r.x0), int(r.y0)), (int(r.x1), int(r.y1)), c, 3 if r.held else 2)
            cv2.putText(im, r.gt_id + ("*" if r.held else ""), (int(r.x0), max(28, int(r.y0) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3)
        cv2.putText(im, str(f), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        tiles.append(cv2.resize(im, (256, 144)))
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows_img = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    for j in range(0, len(rows_img), 13):
        cv2.imwrite(str(GT / f"gt_review_{j // 13}.jpg"), np.vstack(rows_img[j:j + 13]))


if __name__ == "__main__":
    main()

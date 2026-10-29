"""Task 4: quantitative tracking evaluation against SAM ground truth on sampled video frames.

Ground truth (outputs/video_gt/gt_tracks.csv, built by src/build_video_gt.py): one row per egg per sampled
frame with its identity (A = raw egg, B = boiled egg), centroid, box, and whether it is held in a hand.
Predictions: a tracks.csv from track_video.py (centroid per egg per frame, status detected/held/occluded).

Matching: per frame, Hungarian assignment on centroid distance divided by the GT box diagonal; a pair counts as a
match if that normalized distance is < MATCH_DIST. Centroid distance (not box IoU) is used because a held egg is
only partly visible: its GT box is the visible part, while the tracker reports an egg-sized box.

Metrics (standard CLEAR-MOT and identity metrics, implemented here so every number can be explained):
  MOTA  = 1 - (FN + FP + IDSW) / GT       detection and identity errors together
  MOTP  = mean normalized distance of matches (lower is better)
  IDSW  = times a GT egg's matched track ID differs from the ID it was last matched to
  IDF1  = 2 IDTP / (2 IDTP + IDFP + IDFN), with GT and predicted trajectories paired one-to-one so as to
          maximise IDTP (frames where the pair matches): "how much of the time is each egg labelled with *its*
          ID". It is the metric that punishes a lasting swap.
Each is reported for all GT eggs and separately for eggs on the table vs held in a hand. False positives are
only counted in frames where the GT contains both eggs (SAM misses an egg lifted near the shirt; estimating
that egg is correct, not a false positive).

    python src/eval_tracking.py outputs/video/v2/tracks.csv outputs/video/old_tracker_tracks.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
MATCH_DIST = 0.5


def load_pred(path, scale):
    p = pd.read_csv(path)
    p["cx"], p["cy"] = p.cx / scale, p.cy / scale
    return p


def match_frames(gt, pred):
    """Per-frame Hungarian matching. Returns a list of (frame, gt_id, pred_id or None, dist, held)."""
    out, fps = [], {}
    for f, g in gt.groupby("frame"):
        pr = pred[pred.frame == f]
        cost = np.full((len(g), len(pr)), 1e6)
        for i, (_, a) in enumerate(g.iterrows()):
            diag = np.hypot(a.x1 - a.x0, a.y1 - a.y0)
            for j, (_, b) in enumerate(pr.iterrows()):
                d = np.hypot(a.cx - b.cx, a.cy - b.cy) / max(diag, 1)
                if d < MATCH_DIST:
                    cost[i, j] = d
        pairs = {}
        if cost.size:
            pairs = {i: j for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e5}
        for i, (_, a) in enumerate(g.iterrows()):
            j = pairs.get(i)
            out.append((f, a.gt_id, int(pr.iloc[j].id) if j is not None else None,
                        cost[i, j] if j is not None else np.nan, bool(a.held)))
        # predictions with no GT egg. Only counted when the GT has every egg (A and B): in frames where SAM
        # missed an egg (e.g. lifted near the shirt), a tracker correctly estimating it is not a false positive
        if set(g.gt_id) >= {"A", "B"}:
            fps[f] = len(pr) - len(pairs)
    return out, fps


def clear_mot(matches, fps, subset=None):
    m = [x for x in matches if subset is None or x[4] == subset]
    gt_n = len(m)
    fn = sum(x[2] is None for x in m)
    last, idsw = {}, 0
    for f, g, p, d, h in sorted(matches):      # identity history uses every frame, counted on the subset
        if p is None:
            continue
        if g in last and last[g] != p and (subset is None or h == subset):
            idsw += 1
        last[g] = p
    fp = sum(fps.values()) if subset is None else None
    d = [x[3] for x in m if x[2] is not None]
    res = {"gt": gt_n, "matched": gt_n - fn, "fn": fn, "idsw": idsw, "motp": round(float(np.mean(d)), 3) if d else None}
    if fp is not None:
        res["fp"] = fp
        res["mota"] = round(1 - (fn + fp + idsw) / max(gt_n, 1), 3)
    return res


def idf1(matches, pred_frames_per_id, subset=None):
    m = [x for x in matches if subset is None or x[4] == subset]
    gts = sorted({x[1] for x in m})
    pids = sorted(pred_frames_per_id)
    tp = np.zeros((len(gts), len(pids)))
    for f, g, p, d, h in m:
        if p is not None and p in pids:
            tp[gts.index(g), pids.index(p)] += 1
    r, c = linear_sum_assignment(-tp) if tp.size else ([], [])
    idtp = float(tp[r, c].sum()) if len(r) else 0.0
    n_gt = len(m)
    n_pred = sum(pred_frames_per_id.values()) if subset is None else sum(x[2] is not None for x in m)
    idfn, idfp = n_gt - idtp, n_pred - idtp
    pairing = {gts[i]: int(pids[j]) for i, j in zip(r, c) if tp[i, j] > 0}
    return {"idf1": round(2 * idtp / max(2 * idtp + idfp + idfn, 1), 3), "idtp": int(idtp), "pairing": pairing}


def evaluate(gt, pred):
    sampled = sorted(gt.frame.unique())
    pred = pred[pred.frame.isin(sampled)]
    matches, fps = match_frames(gt, pred)
    per_id = pred.groupby("id").size().to_dict()
    out = {}
    for name, sub in (("all", None), ("on_table", False), ("held", True)):
        out[name] = {**clear_mot(matches, fps, sub), **idf1(matches, per_id, sub)}
    # an "ID switch" in the sense that matters here: frames where an egg carries the other egg's ID
    pairing = out["all"]["pairing"]
    wrong = [x for x in matches if x[2] is not None and pairing.get(x[1]) not in (None, x[2])]
    out["all"]["frames_with_wrong_id"] = len(wrong)
    out["all"]["first_wrong_frame"] = int(min(x[0] for x in wrong)) if wrong else None
    return out, matches


def timeline(gt, preds, path):
    """Per GT egg and sampled frame: correct ID, wrong ID or missed, for each tracker (one row per egg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = {"correct": "#2a78d6", "wrong ID": "#e34948", "missed": "#c9c8c3"}
    fig, ax = plt.subplots(figsize=(10, 0.55 * 2 * len(preds) + 1.3))
    ylabels, y = [], 0
    for name, pred in preds.items():
        res, matches = evaluate(gt, pred)
        pairing = res["all"]["pairing"]
        for egg in ("A", "B"):
            for f, g, p, d, h in matches:
                if g != egg:
                    continue
                st = "missed" if p is None else ("correct" if pairing.get(g) == p else "wrong ID")
                ax.barh(y, 4, left=f / 24, height=0.7, color=colors[st], linewidth=0)
                if h:
                    ax.plot(f / 24 + 2 / 24 * 0, y - 0.42, "|", color="#52514e", ms=4)
            ylabels.append(f"{name}: egg {egg}")
            y += 1
        y += 0.4
    for b in ax.patches:
        b.set_width(4 / 24)
    ax.set_yticks([i + (0.4 if i >= 2 else 0) for i in range(len(ylabels))], ylabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time in the video (s).  Tick above a bar: egg held in the hand.  Gap: SAM found no egg there, so no ground truth.", fontsize=8, color="#52514e")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], frameon=False, fontsize=8, ncol=3,
              loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.set_title("Identity per sampled frame vs SAM ground truth", loc="left", fontsize=10, color="#0b0b0b", pad=18)
    ax.set_facecolor("#fcfcfb")
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="+", help="tracks.csv files to evaluate (e.g. new and old tracker)")
    ap.add_argument("--gt", type=Path, default=ROOT / "outputs" / "video_gt" / "gt_tracks.csv")
    ap.add_argument("--scale", type=float, default=1.0, help="tracks.csv pixels per 640-px pixel (3 for 1080p)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "video_gt" / "tracking_metrics.json")
    ap.add_argument("--names", nargs="*", help="short names for the timeline figure, one per tracks file")
    args = ap.parse_args()
    gt = pd.read_csv(args.gt)
    gt = gt[gt.gt_id.isin(["A", "B"])]
    results = {}
    for t in args.tracks:
        res, _ = evaluate(gt, load_pred(t, args.scale))
        results[t] = res
        a = res["all"]
        print(f"\n{t}")
        print(f"  all       MOTA {a['mota']:.3f}  IDF1 {a['idf1']:.3f}  IDSW {a['idsw']}  FN {a['fn']}  FP {a['fp']}  "
              f"MOTP {a['motp']}  GT {a['gt']}  wrong-ID frames {a['frames_with_wrong_id']} (first {a['first_wrong_frame']})")
        for k in ("on_table", "held"):
            r = res[k]
            print(f"  {k:9s} recall {r['matched'] / max(r['gt'], 1):.3f}  IDF1 {r['idf1']:.3f}  IDSW {r['idsw']}  "
                  f"MOTP {r['motp']}  GT {r['gt']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=str))
    if args.names:
        timeline(gt, {n: load_pred(t, args.scale) for n, t in zip(args.names, args.tracks)},
                 args.out.with_name("fig_id_timeline.png"))


if __name__ == "__main__":
    main()

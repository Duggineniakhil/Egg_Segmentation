"""Task 4: track individual eggs in a video with the classical segmentation + a SORT-style tracker.

    python src/track_video.py --video data/video/egg_video.mp4 --config configs/video_v1.yaml
    python src/track_video.py --video ... --start 240 --end 480      # quick test on a clip
    python src/track_video.py --video ... --cache dets.pkl           # re-run only the tracker later

Steps
1. Detection: the final Task 2 algorithm, adapted for video (configs/video_v1.yaml), runs on every
   frame (in parallel across CPU cores). Every egg in a frame becomes a detection: mask, box,
   centroid, area.
2. Tracking: one Kalman filter per egg (state x, y, vx, vy) predicts where the egg will be in the
   next frame. Detections are matched to predictions with the Hungarian algorithm on a cost that
   combines centroid distance and box IoU. Matches further than a gate are rejected.
     - New tracks are confirmed after MIN_HITS consecutive detections, so one-frame blips get no ID.
     - Unmatched tracks coast on their prediction (velocity damped) for up to MAX_AGE frames,
       which covers a hand passing over or picking up an egg.
     - Matches are gated on the egg's typical size (median of recent detections): centroid distance
       <= GATE sizes, vertical move <= GATE_Y heights (eggs slide sideways on the table; straight
       down is the reflection), area within AREA_RATIO. A missed egg coasts instead of jumping onto
       its reflection, a fragment or the other egg.
     - At a hard scene cut all tracks are dropped; a hidden egg is also dropped once the picture has
       changed completely since it was last seen (gradual transitions such as the iris wipe).
     - A detection that overlaps a visible tracked egg, or sits in its reflection zone just below it
       (the table is glossy), may not start a new track.
     - A new track that confirms near the last seen position of an egg that is hidden (coasting)
       or recently lost takes over that ID (re-identification), so an egg picked up by the hand
       and put down again keeps its number.
3. Output: an annotated video (mask, outline, ID, trail), tracks.csv (one row per egg per frame)
   and summary.json.
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]

MIN_HITS = 3          # consecutive detections before a track gets an ID
MAX_AGE = 48          # frames a track may coast without detections (2 s at 24 fps)
REID_FRAMES = 240     # how long a lost track can be revived (10 s)
GATE = 1.0            # max centroid distance, in units of the egg's size (box diagonal)
REID_GATE = 5.0       # max distance for a new track to take over a hidden/lost egg's id
VEL_DAMP = 0.5        # velocity decay per frame while coasting (a held egg moves with the hand, not on its own)
TRAIL = 36            # trail length drawn in the output video
REFLECT_H = 1.5       # reflection zone below a visible egg, in egg heights (glossy table)
AREA_RATIO = 2.0      # a match may differ from the egg's typical area by at most this factor
GATE_Y = 0.5          # max vertical move per frame, in egg heights: with the camera near table level, eggs on the
                      # table move sideways in the image; a jump straight down is the reflection
SIZE_WINDOW = 10      # detections used for an egg's typical size
CUT_DIFF = 40.0       # mean abs. frame difference (0-255, on 64x36 thumbnails) that counts as a scene cut

PALETTE = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232), (0, 131, 0)]  # BGR


# ----------------------------------------------------------------------------- detection

_PIPE = None


def _init(cfg_path):
    global _PIPE
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from classical_seg import load_config
    _PIPE = load_config(cfg_path)["pipeline"]


def _detect(args):
    idx, frame = args
    from classical_seg import segment
    lab = segment(frame, _PIPE)
    dets = []
    for i in np.unique(lab)[1:]:
        m = lab == i
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        dets.append({"box": (int(x0), int(y0), int(x1), int(y1)), "c": (float(xs.mean()), float(ys.mean())),
                     "area": int(m.sum()), "crop": np.packbits(m[y0:y1, x0:x1], axis=None), "shape": (y1 - y0, x1 - x0)})
    return idx, dets


def read_frames(path, start, end, proc_width):
    """Frames for detection, downscaled to proc_width (the algorithm's pixel parameters are tuned for
    ~640 px wide images). Returns frames, fps, and scale = original width / processing width."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, i, scale = [], start, 1.0
    while end is None or i < end:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > proc_width:
            scale = f.shape[1] / proc_width
            f = cv2.resize(f, (proc_width, round(f.shape[0] / scale)), interpolation=cv2.INTER_AREA)
        frames.append(f)
        i += 1
    return frames, fps, scale


def detect_all(frames, cfg, workers):
    jobs = list(enumerate(frames))
    out = [None] * len(frames)
    with Pool(workers, initializer=_init, initargs=(str(cfg),)) as pool:
        for k, (idx, dets) in enumerate(pool.imap_unordered(_detect, jobs, chunksize=4)):
            out[idx] = dets
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                print(f"  detected {k + 1}/{len(jobs)} frames", flush=True)
    return out


def unpack(d):
    h, w = d["shape"]
    return np.unpackbits(d["crop"], count=h * w).reshape(h, w).astype(bool)


# ----------------------------------------------------------------------------- tracking

class Track:
    def __init__(self, det, tid):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.diag([1, 1, 4, 4]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        self.kf.errorCovPost = np.diag([10, 10, 100, 100]).astype(np.float32)
        self.kf.statePost = np.array([[det["c"][0]], [det["c"][1]], [0], [0]], np.float32)
        self.tid = tid               # provisional id until confirmed
        self.det = det
        self.hits, self.miss = 1, 0
        self.confirmed = False
        self.trail = [det["c"]]
        self.dims = [_dims(det)]     # recent (width, height, area): a stable size even when one blob is off

    def predict(self):
        if self.miss:                # coasting: let the egg slow down instead of flying off
            self.kf.statePost[2:] *= VEL_DAMP
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def update(self, det):
        self.kf.correct(np.array([[det["c"][0]], [det["c"][1]]], np.float32))
        self.det = det
        self.hits += 1
        self.miss = 0
        self.trail = (self.trail + [det["c"]])[-TRAIL:]
        self.dims = (self.dims + [_dims(det)])[-SIZE_WINDOW:]

    def ref(self):
        """Median width, height and area over recent detections."""
        w, h, a = np.median(np.array(self.dims), axis=0)
        return float(w), float(h), float(a)

    def size(self):
        w, h, _ = self.ref()
        return float(np.hypot(w, h))


def _dims(det):
    x0, y0, x1, y1 = det["box"]
    return (x1 - x0, y1 - y0, det["area"])


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def shifted_box(box, c_old, c_new):
    dx, dy = c_new[0] - c_old[0], c_new[1] - c_old[1]
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def shadow_of_egg(d, visible):
    """True if a detection is a duplicate or a reflection of an egg that is already tracked and visible.

    - Overlaps a visible egg: an oversized blob containing that egg (egg + table below it).
    - Directly below a visible egg, within REFLECT_H of its height: its mirror image on the glossy table.
    Such detections may not start a new track (they can still be matched to an existing one).
    """
    x0, y0, x1, y1 = d["box"]
    for e in visible:
        ex0, ey0, ex1, ey1 = e["box"]
        if box_iou(d["box"], e["box"]) > 0.1 or (ex0 <= d["c"][0] <= ex1 and ey0 <= d["c"][1] <= ey1):
            return True
        eh = ey1 - ey0
        if ex0 - 0.25 * (ex1 - ex0) <= d["c"][0] <= ex1 + 0.25 * (ex1 - ex0) and ey1 - 0.2 * eh <= d["c"][1] <= ey1 + REFLECT_H * eh:
            return True
    return False


def scene_cuts(video, start, n):
    """Frames where the picture changes completely (title cards, wipes): all tracks are dropped there."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    cuts, thumbs = set(), []
    for f in range(n):
        ok, im = cap.read()
        if not ok:
            break
        thumbs.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
                      .astype(np.float32))
        if f and np.abs(thumbs[f] - thumbs[f - 1]).mean() > CUT_DIFF:
            cuts.add(f)
    return cuts, thumbs


def track(all_dets, cuts=frozenset(), thumbs=None):
    tracks, lost, next_id, next_tmp = [], [], 1, -1
    per_frame = []
    for f, dets in enumerate(all_dets):
        if f in cuts:            # new shot: nothing carries over, but eggs can still be re-identified
            for t in tracks:
                if t.confirmed:
                    lost.append({"tid": t.tid, "c": t.det["c"], "size": t.size(), "frame": f, "trail": t.trail})
            tracks = []
        preds = [t.predict() for t in tracks]
        # cost: centroid distance relative to egg size, minus a bonus for box overlap
        cost = np.full((len(tracks), len(dets)), 1e6)
        for i, (t, p) in enumerate(zip(tracks, preds)):
            pbox = shifted_box(t.det["box"], t.det["c"], p)
            for j, d in enumerate(dets):
                rw, rh, ra = t.ref()
                dist = np.hypot(d["c"][0] - p[0], d["c"][1] - p[1]) / max(t.size(), 1)
                ratio = d["area"] / max(ra, 1)
                slack = min(1.5, 1 + 0.25 * t.miss)          # a little more room after missed frames
                dy_ok = abs(d["c"][1] - p[1]) <= GATE_Y * rh
                if dist <= GATE * slack and dy_ok and 1 / AREA_RATIO <= ratio <= AREA_RATIO:
                    cost[i, j] = dist - box_iou(pbox, d["box"])
        matched_t, matched_d = set(), set()
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    tracks[i].update(dets[j])
                    matched_t.add(i)
                    matched_d.add(j)
        for i, t in enumerate(tracks):
            if i not in matched_t:
                t.miss += 1
                if not t.confirmed:      # a tentative track must be hit every frame
                    t.hits = 0
        visible = [t.det for t in tracks if t.confirmed and t.miss == 0]
        for j, d in enumerate(dets):
            if j not in matched_d and not shadow_of_egg(d, visible):
                tracks.append(Track(d, next_tmp))
                next_tmp -= 1
        # confirm tracks; take over the id of a nearby egg that is hidden (still coasting) or recently lost
        for t in list(tracks):
            if not t.confirmed and t.hits >= MIN_HITS:
                t.confirmed = True
                hidden = [{"tid": o.tid, "c": o.det["c"], "size": o.size(), "trail": o.trail, "track": o}
                          for o in tracks if o.confirmed and o is not t and o.miss > 0]
                cands = [(np.hypot(t.det["c"][0] - l["c"][0], t.det["c"][1] - l["c"][1]) / max(l["size"], 1), l)
                         for l in hidden + [l for l in lost if f - l["frame"] <= REID_FRAMES]]
                cands = [c for c in cands if c[0] <= REID_GATE]
                if cands:
                    best = min(cands, key=lambda c: c[0])[1]
                    t.tid = best["tid"]      # trail starts fresh: the egg was carried, not tracked
                    if "track" in best:
                        tracks.remove(best["track"])
                    else:
                        lost.remove(best)
                else:
                    t.tid = next_id
                    next_id += 1
        # retire stale tracks: hidden too long, or the picture has changed completely since the egg was
        # last seen (a gradual transition such as the iris wipe, which is not a hard cut)
        keep = []
        for t in tracks:
            changed = (thumbs is not None and t.miss and
                       np.abs(thumbs[f] - thumbs[f - t.miss]).mean() > CUT_DIFF)
            if t.confirmed and (t.miss > MAX_AGE or changed):
                lost.append({"tid": t.tid, "c": t.det["c"], "size": t.size(), "frame": f, "trail": t.trail})
            elif not t.confirmed and t.hits == 0:
                pass
            else:
                keep.append(t)
        tracks = keep
        lost = [l for l in lost if f - l["frame"] <= REID_FRAMES]
        per_frame.append([(t.tid, t.det if t.miss == 0 else None, t.kf.statePost[:2, 0].tolist(), list(t.trail))
                          for t in tracks if t.confirmed])
    return per_frame


# ----------------------------------------------------------------------------- output

def color(tid):
    return PALETTE[(tid - 1) % len(PALETTE)]


def render(video, per_frame, out_path, fps, start, scale):
    """Draw the tracks on the ORIGINAL frames, streamed from disk (full HD never sits in memory).
    Detection coordinates are in processing pixels and are multiplied by `scale`."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    lw = max(2, round(2 * scale))                 # line width
    fs = 0.55 * scale                             # font scale
    S = lambda v: int(round(v * scale))
    rows = []
    for f, objs in enumerate(per_frame):
        ok, vis = cap.read()
        if not ok:
            break
        for tid, det, state, trail in objs:
            c = color(tid)
            pts = (np.array(trail) * scale).astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(vis, [pts], False, c, lw, cv2.LINE_AA)
            if det is not None:
                x0, y0, x1, y1 = det["box"]
                X0, Y0, X1, Y1 = S(x0), S(y0), min(S(x1), w), min(S(y1), h)
                crop = cv2.resize(unpack(det).astype(np.float32), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR) > 0.5
                m = np.zeros((h, w), bool)
                m[Y0:Y1, X0:X1] = crop
                vis[m] = (0.55 * vis[m] + 0.45 * np.array(c)).astype(np.uint8)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, cnts, -1, c, lw, cv2.LINE_AA)
                tx, ty = X0, max(S(18), Y0 - S(6))
                rows.append({"frame": f + start, "id": tid, "cx": round(det["c"][0] * scale, 1),
                             "cy": round(det["c"][1] * scale, 1), "area": int(det["area"] * scale ** 2),
                             "x0": X0, "y0": Y0, "x1": X1, "y1": Y1, "status": "detected"})
            else:      # hidden (e.g. in the hand): dashed ellipse at the predicted position
                cx, cy = S(state[0]), S(state[1])
                for a in range(0, 360, 30):
                    cv2.ellipse(vis, (cx, cy), (S(18), S(14)), 0, a, a + 15, c, lw, cv2.LINE_AA)
                tx, ty = cx - S(18), cy - S(20)
                rows.append({"frame": f + start, "id": tid, "cx": round(state[0] * scale, 1),
                             "cy": round(state[1] * scale, 1), "area": 0,
                             "x0": None, "y0": None, "x1": None, "y1": None, "status": "occluded"})
            label = f"Egg {tid}" + ("" if det is not None else " (hidden)")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
            cv2.rectangle(vis, (tx - S(2), ty - th - S(4)), (tx + tw + S(2), ty + S(3)), c, cv2.FILLED)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), lw, cv2.LINE_AA)
        n_vis = sum(o[1] is not None for o in objs)
        cv2.putText(vis, f"frame {f + start}  eggs tracked: {len(objs)} (visible {n_vis})", (S(8), h - S(10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), lw, cv2.LINE_AA)
        vw.write(vis)
    vw.release()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "video_v1.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "video")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--proc-width", type=int, default=640, help="detection resolution (output keeps the original)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="pickle of detections: reused if it exists, so tracker changes re-run in seconds")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.cache and args.cache.exists():
        dets, fps, scale = pickle.loads(args.cache.read_bytes())
        print(f"loaded {len(dets)} frames of detections from {args.cache}")
    else:
        frames, fps, scale = read_frames(args.video, args.start, args.end, args.proc_width)
        print(f"{len(frames)} frames at {fps:.0f} fps, detecting at {frames[0].shape[1]}x{frames[0].shape[0]} "
              f"(scale {scale:.2f}) with {args.workers} workers")
        dets = detect_all(frames, args.config, args.workers)
        del frames
        if args.cache:
            args.cache.write_bytes(pickle.dumps((dets, fps, scale)))
    cuts, thumbs = scene_cuts(args.video, args.start, len(dets))
    print(f"scene cuts at frames {sorted(c + args.start for c in cuts)}")
    per_frame = track(dets, cuts, thumbs)
    df = render(args.video, per_frame, args.out / "tracked_raw.mp4", fps, args.start, scale)
    df.to_csv(args.out / "tracks.csv", index=False)

    summary = {
        "frames": len(dets), "fps": fps, "config": str(args.config.name), "detection_scale": round(scale, 3),
        "params": {"MIN_HITS": MIN_HITS, "MAX_AGE": MAX_AGE, "REID_FRAMES": REID_FRAMES, "GATE": GATE,
                   "REID_GATE": REID_GATE, "REFLECT_H": REFLECT_H,
                   "AREA_RATIO": AREA_RATIO, "CUT_DIFF": CUT_DIFF,
                   "GATE_Y": GATE_Y, "SIZE_WINDOW": SIZE_WINDOW},
        "frames_with_detections": int(sum(bool(d) for d in dets)),
        "track_ids": sorted(int(i) for i in df.id.unique()) if len(df) else [],
        "tracks": {int(i): {"first_frame": int(g.frame.min()), "last_frame": int(g.frame.max()),
                            "frames_detected": int((g.status == "detected").sum()),
                            "frames_occluded": int((g.status == "occluded").sum())}
                   for i, g in df.groupby("id")} if len(df) else {},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

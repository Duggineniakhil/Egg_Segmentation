"""Task 4: track individual eggs in a video with the classical segmentation + a SORT-style tracker.

    python src/track_video.py --video data/video/egg_video.mp4 --config configs/video_v1.yaml
    python src/track_video.py --video ... --start 240 --end 480      # quick test on a clip
    python src/track_video.py --video ... --cache dets.pkl           # re-run only the tracker later

Steps
1. Detection: the final Task 2 algorithm, adapted for video (configs/video_v1.yaml), runs on every
   frame in parallel across CPU cores, at 640 px wide. Every egg in a frame becomes a detection: mask,
   box, centroid, area.
2. Tracking. Each egg has a track with a persistent ID.
   - Visible egg: a Kalman filter (x, y, vx, vy) predicts the next position. Detections are matched to
     tracks with the Hungarian algorithm (order of detections doesn't matter) on centroid distance
     minus box IoU, gated by the egg's typical size (median of recent detections), its vertical move
     (eggs slide sideways on the table; straight down is the reflection) and its area.
   - Hidden egg (missed, covered, or held in a hand): the track is NOT dropped. Every frame its estimate
     is moved by the optical flow (pyramidal Lucas-Kanade, forward-backward checked) of whatever covers
     it. Under a hand that is the hand's motion, so the estimate is carried along with the hand; on a
     still table it is ~0. The estimate then locks onto the visible white part of the held egg when
     there is one (compact egg-white blob near the estimate; blobs are shared out one-to-one between
     hidden eggs, so two hidden eggs can't claim the same blob).
   - Recovery: a hidden egg is matched to a detection near its propagated estimate (REID_GATE) of
     about its own area (RECOVER_AREA), never onto another egg or the reflection under an egg. A new
     track that confirms (MIN_HITS) is first bound to a hidden egg before a new ID is handed out.
   - Tracks are only dropped at a scene change (hard cut, or the picture having changed completely
     since the egg was last seen, as in the iris wipe), when the estimate leaves the frame, or after
     MAX_HIDDEN frames.
   All parameters are in PARAMS and can be overridden in the config's `tracker:` section.
3. Output: an annotated video (mask + ID when visible, dashed box + ID at the estimated position when
   hidden), tracks.csv (one row per egg per frame, status detected/held/occluded), tracking_log.csv
   (new IDs, misses, recoveries with distance and confidence, retirements), summary.json and
   debug_frames/.

Limitation: the two eggs look identical, so identity comes only from motion continuity. When an egg
is fully hidden and does not move with anything visible, its identity on reappearance is a best
guess; such recoveries are flagged as low_confidence in tracking_log.csv.
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
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]

# Tracker parameters (defaults). Override any of them in the config file's `tracker:` section.
PARAMS = {
    "MIN_HITS": 3,          # consecutive detections before a new track gets an ID
    "GATE": 1.0,            # visible egg: max centroid move per frame, in egg sizes (box diagonal)
    "GATE_Y": 0.5,          # visible egg: max vertical move per frame, in egg heights (eggs on the table move
                            # sideways in the image; a jump straight down is the reflection)
    "AREA_RATIO": 2.0,      # a match may differ from the egg's typical area by at most this factor
    "SIZE_WINDOW": 10,      # detections used for an egg's typical size
    "REID_GATE": 1.5,       # hidden egg: max distance between its propagated estimate and a detection, in sizes
    "RECOVER_AREA": [0.65, 1.6],  # a hidden egg is recovered only onto a detection of about its own area
                            # (not onto a fragment of the egg still in the hand)
    "REID_FAR": 3.0,        # fallback when a new track confirms: bind to a hidden egg within this many sizes
    "LOW_CONF": 1.0,        # recoveries further than this (sizes) are logged as low confidence
    "MAX_HIDDEN": 720,      # frames an egg may stay hidden before its track is dropped (30 s at 24 fps)
    "FLOW_PTS": 6,          # min good optical-flow points to move a hidden egg's estimate
    "FLOW_MAX_STEP": 0.6,   # max flow move per frame, in sizes
    "WHITE_V": 170,         # egg-white pixels: HSV value >= this ...
    "WHITE_S": 45,          # ... and saturation <= this
    "WHITE_MIN": 0.12,      # held-egg white blob: area >= this fraction of the egg's typical area ...
    "WHITE_MAX": 1.6,       # ... and <= this fraction
    "WHITE_SOLIDITY": 0.55, # ... and compact (rejects the shirt's check pattern)
    "WHITE_GATE": 1.0,      # ... and within this many sizes of the flow estimate
    "REFLECT_H": 1.5,       # reflection zone below a visible egg, in egg heights (glossy table)
    "CUT_DIFF": 40.0,       # mean abs. frame difference (0-255, 64x36 thumbnails) that counts as a scene change
    "TRAIL": 36,            # trail length drawn in the output video
}

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
    """One egg. While visible, a Kalman filter (x, y, vx, vy) predicts its next position. While hidden
    (missed, or held in a hand), its estimate `est` is moved by optical flow and locked onto the visible
    part of the egg (egg-white blob) when there is one."""

    def __init__(self, det, tid, frame):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.diag([1, 1, 4, 4]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        self._reset_kf(det["c"])
        self.tid = tid               # provisional (negative) id until confirmed
        self.det = det
        self.hits, self.miss = 1, 0
        self.confirmed = False
        self.trail = [det["c"]]
        self.dims = [_dims(det)]     # recent (width, height, area): a stable size even when one blob is off
        self.est = det["c"]          # current position estimate (= detection centroid while visible)
        self.status = "detected"     # detected | held (white blob found while hidden) | occluded (flow only)
        self.last_seen = frame

    def _reset_kf(self, c):
        self.kf.errorCovPost = np.diag([10, 10, 100, 100]).astype(np.float32)
        self.kf.statePost = np.array([[c[0]], [c[1]], [0], [0]], np.float32)

    def predict(self):
        """Where to look for this egg in the new frame."""
        if self.miss == 0:
            p = self.kf.predict()
            return float(p[0, 0]), float(p[1, 0])
        return self.est

    def update(self, det, frame):
        if self.miss:                # recovered after being hidden: restart the motion model here
            self._reset_kf(det["c"])
        self.kf.correct(np.array([[det["c"][0]], [det["c"][1]]], np.float32))
        self.det = det
        self.hits += 1
        self.miss = 0
        self.est = det["c"]
        self.status = "detected"
        self.last_seen = frame
        self.trail = (self.trail + [det["c"]])[-PARAMS["TRAIL"]:]
        self.dims = (self.dims + [_dims(det)])[-PARAMS["SIZE_WINDOW"]:]

    def ref(self):
        """Median width, height and area over recent detections."""
        w, h, a = np.median(np.array(self.dims), axis=0)
        return float(w), float(h), float(a)

    def size(self):
        w, h, _ = self.ref()
        return float(np.hypot(w, h))

    def est_box(self):
        w, h, _ = self.ref()
        return (self.est[0] - w / 2, self.est[1] - h / 2, self.est[0] + w / 2, self.est[1] + h / 2)


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


def shadow_of_egg(d, visible, reflection_only=()):
    """True if a detection is a duplicate or a reflection of an egg that is already tracked.

    - Overlaps an egg in `visible`: an oversized blob containing that egg (egg + table below it).
    - Directly below an egg in `visible` or `reflection_only`, within REFLECT_H of its height: its mirror
      image on the glossy table.
    Such detections may not start a new track or recover a hidden one.
    """
    for e in visible:
        ex0, ey0, ex1, ey1 = e["box"]
        if box_iou(d["box"], e["box"]) > 0.1 or (ex0 <= d["c"][0] <= ex1 and ey0 <= d["c"][1] <= ey1):
            return True
    for e in [*visible, *reflection_only]:
        ex0, ey0, ex1, ey1 = e["box"]
        eh, ew = ey1 - ey0, ex1 - ex0
        if (ex0 - 0.25 * ew <= d["c"][0] <= ex1 + 0.25 * ew and
                ey1 - 0.2 * eh <= d["c"][1] <= ey1 + PARAMS["REFLECT_H"] * eh):
            return True
    return False


def scene_cuts(video, start, n):
    """Thumbnails of every frame, and the frames where the picture changes completely (hard cuts)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    cuts, thumbs = set(), []
    for f in range(n):
        ok, im = cap.read()
        if not ok:
            break
        thumbs.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
                      .astype(np.float32))
        if f and np.abs(thumbs[f] - thumbs[f - 1]).mean() > PARAMS["CUT_DIFF"]:
            cuts.add(f)
    return cuts, thumbs


# ----------------------------------------------------------------------------- following hidden eggs

def flow_shift(prev_gray, gray, box, size):
    """Median optical-flow displacement of the content of `box` (forward-backward checked).

    Under a hand, that content is the hand, so the estimate moves with the hand. For a plain missed
    detection on a still table it is ~0, so the estimate stays put instead of drifting.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    pad = int(0.2 * size)
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0, 0.0, 0
    mask = np.zeros_like(gray)
    mask[y0:y1, x0:x1] = 255
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=60, qualityLevel=0.01, minDistance=4, mask=mask)
    if pts is None or len(pts) < PARAMS["FLOW_PTS"]:
        return 0.0, 0.0, 0
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **lk)
    good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pts - back, axis=2)[:, 0] < 1.0)
    if good.sum() < PARAMS["FLOW_PTS"]:
        return 0.0, 0.0, int(good.sum())
    d = np.median((nxt - pts)[good, 0], axis=0)
    step = np.hypot(*d)
    cap = PARAMS["FLOW_MAX_STEP"] * size
    if step > cap:
        d = d * cap / step
    return float(d[0]), float(d[1]), int(good.sum())


def white_blobs(bgr, exclude_boxes):
    """Compact egg-white blobs (candidate visible parts of held eggs), outside the boxes of visible eggs."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] >= PARAMS["WHITE_V"]) & (hsv[..., 1] <= PARAMS["WHITE_S"])).astype(np.uint8)
    for x0, y0, x1, y1 in exclude_boxes:
        m[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        x, y, bw, bh = stats[i, :4]
        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        solidity = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
        blobs.append({"c": (float(cents[i][0]), float(cents[i][1])), "area": int(area), "solidity": solidity})
    return blobs


def follow_hidden(hidden, prev_gray, gray, bgr, visible_boxes):
    """Move every hidden egg's estimate: optical flow first, then lock onto a held egg's white blob.
    Blobs are shared out one-to-one (Hungarian), so two hidden eggs can't claim the same blob."""
    for t in hidden:
        dx, dy, _ = flow_shift(prev_gray, gray, t.est_box(), t.size())
        t.est = (t.est[0] + dx, t.est[1] + dy)
        t.status = "occluded"
    if not hidden:
        return
    blobs = white_blobs(bgr, visible_boxes)
    cost = np.full((len(hidden), len(blobs)), 1e6)
    for i, t in enumerate(hidden):
        _, _, ra = t.ref()
        for j, b in enumerate(blobs):
            frac = b["area"] / max(ra, 1)
            dist = np.hypot(b["c"][0] - t.est[0], b["c"][1] - t.est[1]) / max(t.size(), 1)
            if (PARAMS["WHITE_MIN"] <= frac <= PARAMS["WHITE_MAX"] and b["solidity"] >= PARAMS["WHITE_SOLIDITY"]
                    and dist <= PARAMS["WHITE_GATE"]):
                cost[i, j] = dist
    if cost.size:
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e5:
                hidden[i].est = blobs[j]["c"]
                hidden[i].status = "held"


# ----------------------------------------------------------------------------- association

def _known_eggs(tracks, exclude=None):
    """Where every tracked egg is believed to be: its detection if visible, its estimated box if hidden. Used for the duplicate/reflection test, so the reflection under a hidden
    (e.g. held) egg is covered too. `exclude` leaves out one track (its own position is not a duplicate)."""
    out = []
    for t in tracks:
        if not t.confirmed or t is exclude:
            continue
        out.append(t.det if t.miss == 0 else {"box": t.est_box(), "c": t.est})
    return out


def _cost(t, p, d):
    """Association cost of track t (predicted at p) with detection d, or None if outside the gates."""
    rw, rh, ra = t.ref()
    size = max(t.size(), 1)
    dist = np.hypot(d["c"][0] - p[0], d["c"][1] - p[1]) / size
    ratio = d["area"] / max(ra, 1)
    if not (1 / PARAMS["AREA_RATIO"] <= ratio <= PARAMS["AREA_RATIO"]):
        return None
    if t.miss == 0:      # visible last frame: small, mostly sideways moves
        if dist > PARAMS["GATE"] or abs(d["c"][1] - p[1]) > PARAMS["GATE_Y"] * rh:
            return None
    elif dist > PARAMS["REID_GATE"] or not (PARAMS["RECOVER_AREA"][0] <= ratio <= PARAMS["RECOVER_AREA"][1]):
        return None      # hidden: compare with the propagated estimate; recover only onto a whole egg
    pbox = shifted_box(t.det["box"], t.det["c"], p)
    return dist - box_iou(pbox, d["box"])


def track(all_dets, video, start, proc_width, cuts=frozenset(), thumbs=None, log=None):
    """Assign a persistent ID to every egg. `log` (a list) receives debug events."""
    log = [] if log is None else log
    ev = lambda f, kind, tid, **kw: log.append({"frame": f + start, "event": kind, "id": tid, **kw})
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    tracks, next_id, next_tmp, prev_gray = [], 1, -1, None
    per_frame = []
    for f, dets in enumerate(all_dets):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > proc_width:
            frame = cv2.resize(frame, (proc_width, round(frame.shape[0] * proc_width / frame.shape[1])),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if f in cuts:            # new shot: nothing carries over
            for t in tracks:
                if t.confirmed:
                    ev(f, "retire", t.tid, reason="scene cut")
            tracks = []

        # 1) associate detections with all tracks (visible: Kalman prediction, hidden: propagated estimate)
        preds = [t.predict() for t in tracks]
        cost = np.full((len(tracks), len(dets)), 1e6)
        for i, (t, p) in enumerate(zip(tracks, preds)):
            if t.confirmed and t.miss:
                others = _known_eggs(tracks, exclude=t)
                own = [t.det, {"box": t.est_box(), "c": t.est}]
            for j, d in enumerate(dets):
                # a hidden egg may not be "recovered" onto another egg, or onto the reflection under another
                # egg or under itself (e.g. the reflection of an egg the hand has just covered)
                if t.confirmed and t.miss and shadow_of_egg(d, others, reflection_only=own):
                    continue
                c = _cost(t, p, d)
                if c is not None:
                    cost[i, j] = c
        matched_t, matched_d = set(), set()
        if cost.size:
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    t, d = tracks[i], dets[j]
                    if t.confirmed and t.miss:
                        dist = np.hypot(d["c"][0] - t.est[0], d["c"][1] - t.est[1]) / max(t.size(), 1)
                        ev(f, "recover", t.tid, hidden_frames=t.miss, dist_sizes=round(dist, 2),
                           low_confidence=bool(dist > PARAMS["LOW_CONF"]))
                    t.update(d, f)
                    matched_t.add(i)
                    matched_d.add(j)
        for i, t in enumerate(tracks):
            if i not in matched_t:
                if t.confirmed and t.miss == 0:
                    ev(f, "miss", t.tid, at=[round(v) for v in t.est])
                t.miss += 1
                if not t.confirmed:      # a tentative track must be hit every frame
                    t.hits = 0

        # 2) unmatched detections start tentative tracks (unless they are a duplicate/reflection)
        known = _known_eggs(tracks)
        for j, d in enumerate(dets):
            if j not in matched_d and not shadow_of_egg(d, known):
                tracks.append(Track(d, next_tmp, f))
                next_tmp -= 1

        # 3) confirm tentative tracks; bind them to hidden eggs before handing out a new ID
        confirming = [t for t in tracks if not t.confirmed and t.hits >= PARAMS["MIN_HITS"]]
        hidden = [t for t in tracks if t.confirmed and t.miss > 0]
        if confirming:
            cost = np.full((len(confirming), len(hidden)), 1e6)
            for i, t in enumerate(confirming):
                for j, o in enumerate(hidden):
                    dist = np.hypot(t.det["c"][0] - o.est[0], t.det["c"][1] - o.est[1]) / max(o.size(), 1)
                    ratio = t.det["area"] / max(o.ref()[2], 1)
                    if dist <= PARAMS["REID_FAR"] and PARAMS["RECOVER_AREA"][0] <= ratio <= PARAMS["RECOVER_AREA"][1]:
                        cost[i, j] = dist
            bound = {}
            if cost.size:
                bound = {i: j for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e5}
            for i, t in enumerate(confirming):
                t.confirmed = True
                if i in bound:
                    o = hidden[bound[i]]
                    ev(f, "recover", o.tid, hidden_frames=o.miss, dist_sizes=round(cost[i, bound[i]], 2),
                       low_confidence=bool(cost[i, bound[i]] > PARAMS["LOW_CONF"]), via="new track")
                    t.tid = o.tid
                    tracks.remove(o)
                else:
                    t.tid = next_id
                    next_id += 1
                    ev(f, "new_id", t.tid, at=[round(v) for v in t.det["c"]])

        # 4) move hidden eggs with the scene (hand) and onto their visible white part, for the next frame
        hidden = [t for t in tracks if t.confirmed and t.miss > 0]
        if prev_gray is not None:
            follow_hidden(hidden, prev_gray, gray, frame,
                          [t.det["box"] for t in tracks if t.confirmed and t.miss == 0])

        for t in hidden:          # the trail follows the estimate too, so it shows the path the hand carried the egg
            t.trail = (t.trail + [t.est])[-PARAMS["TRAIL"]:]

        # 5) retire: hidden too long, out of the picture, or the picture changed completely (iris wipe)
        keep = []
        h, w = gray.shape
        for t in tracks:
            if not t.confirmed:
                if t.hits > 0:
                    keep.append(t)
                continue
            reason = None
            if t.miss > PARAMS["MAX_HIDDEN"]:
                reason = "hidden too long"
            elif t.miss and not (-0.5 * t.size() <= t.est[0] <= w + 0.5 * t.size()
                                 and -0.5 * t.size() <= t.est[1] <= h + 0.5 * t.size()):
                reason = "left the frame"
            elif (thumbs is not None and t.miss and
                  np.abs(thumbs[f] - thumbs[t.last_seen]).mean() > PARAMS["CUT_DIFF"]):
                reason = "scene changed"
            if reason:
                ev(f, "retire", t.tid, reason=reason)
            else:
                keep.append(t)
        tracks = keep

        ids = [t.tid for t in tracks if t.confirmed]
        if len(ids) != len(set(ids)):
            ev(f, "duplicate_id", sorted(ids))
        per_frame.append([(t.tid, t.det if t.miss == 0 else None, t.est_box(), list(t.trail), t.status)
                          for t in tracks if t.confirmed])
        prev_gray = gray
    return per_frame, log


# ----------------------------------------------------------------------------- output

def color(tid):
    return PALETTE[(tid - 1) % len(PALETTE)]


def _dashed_rect(img, p0, p1, color, lw, dash):
    (x0, y0), (x1, y1) = p0, p1
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
        for s0 in range(0, length, 2 * dash):
            s1 = min(s0 + dash, length)
            pa = (int(a[0] + (b[0] - a[0]) * s0 / length), int(a[1] + (b[1] - a[1]) * s0 / length))
            pb = (int(a[0] + (b[0] - a[0]) * s1 / length), int(a[1] + (b[1] - a[1]) * s1 / length))
            cv2.line(img, pa, pb, color, lw, cv2.LINE_AA)


def render(video, per_frame, out_path, fps, start, scale, debug_frames=()):
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
        for tid, det, ebox, trail, status in objs:
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
                cx, cy = det["c"]
            else:      # held in the hand or covered: dashed box at the estimated position (no mask to show)
                X0, Y0, X1, Y1 = (S(v) for v in ebox)
                _dashed_rect(vis, (X0, Y0), (X1, Y1), c, lw, S(8))
                cx, cy = (ebox[0] + ebox[2]) / 2, (ebox[1] + ebox[3]) / 2
            tx, ty = max(0, X0), max(S(18), Y0 - S(6))
            rows.append({"frame": f + start, "id": tid, "cx": round(cx * scale, 1), "cy": round(cy * scale, 1),
                         "area": int(det["area"] * scale ** 2) if det is not None else 0,
                         "x0": X0, "y0": Y0, "x1": X1, "y1": Y1, "status": status})
            label = f"Egg {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
            cv2.rectangle(vis, (tx - S(2), ty - th - S(4)), (tx + tw + S(2), ty + S(3)), c, cv2.FILLED)
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), lw, cv2.LINE_AA)
        cv2.putText(vis, f"frame {f + start}  eggs tracked: {len(objs)}", (S(8), h - S(10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), lw, cv2.LINE_AA)
        vw.write(vis)
        if f + start in debug_frames:
            d = Path(out_path).parent / "debug_frames"
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"frame_{f + start:04d}_{(f + start) / fps:05.1f}s.jpg"), vis)
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
    ap.add_argument("--debug-frames", default="490,510,530,550,670,690,726,734,742,750,758,770,800,812",
                    help="comma-separated frame numbers to save as annotated JPEGs (hand pick-ups and the 30 s handover)")
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
    cfg = yaml.safe_load(args.config.read_text()) or {}
    unknown = set(cfg.get("tracker", {})) - set(PARAMS)
    if unknown:
        raise SystemExit(f"unknown tracker parameters in {args.config}: {sorted(unknown)}")
    PARAMS.update(cfg.get("tracker", {}))
    cuts, thumbs = scene_cuts(args.video, args.start, len(dets))
    print(f"scene cuts at frames {sorted(c + args.start for c in cuts)}")
    per_frame, log = track(dets, args.video, args.start, args.proc_width, cuts, thumbs)
    pd.DataFrame(log).to_csv(args.out / "tracking_log.csv", index=False)
    for e in log:
        if e["event"] in ("new_id", "recover", "retire", "duplicate_id"):
            print("  ", {k: (float(v) if isinstance(v, np.floating) else v) for k, v in e.items()})
    debug = {int(x) for x in args.debug_frames.split(",") if x.strip()}
    df = render(args.video, per_frame, args.out / "tracked_raw.mp4", fps, args.start, scale, debug)
    df.to_csv(args.out / "tracks.csv", index=False)

    ev = pd.DataFrame(log)
    summary = {
        "frames": len(dets), "fps": fps, "config": str(args.config.name), "detection_scale": round(scale, 3),
        "params": PARAMS,
        "frames_with_detections": int(sum(bool(d) for d in dets)),
        "track_ids": sorted(int(i) for i in df.id.unique()) if len(df) else [],
        "tracks": {int(i): {"first_frame": int(g.frame.min()), "last_frame": int(g.frame.max()),
                            **{k: int((g.status == k).sum()) for k in ("detected", "held", "occluded")}}
                   for i, g in df.groupby("id")} if len(df) else {},
        "events": ev.event.value_counts().to_dict() if len(ev) else {},
        "low_confidence_recoveries": int((ev["low_confidence"] == True).sum())
        if len(ev) and "low_confidence" in ev else 0,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

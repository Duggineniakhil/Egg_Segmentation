"""Classical (no learning) egg segmentation.

Pipeline, each stage set in the config's `pipeline:` block:
  1. blur       smooth noise and fine background texture
  2. score      per-pixel "egg-likeness" map (eggs are bright and colourless)
  3. threshold  score map -> binary foreground
  4. morph      opening / closing / hole filling
  5. filter     drop connected components that don't look like an egg (size, solidity, ellipse fit)
  6. split      separate touching eggs (distance transform + watershed)
  (3-5 alt.)   sweep: many thresholds, both polarities; egg-shaped, edge-supported blobs win
  7. refine     snap boundaries to image edges (GrabCut) and/or replace blobs with fitted ellipses

segment(img_bgr, pipeline_cfg) returns a uint16 instance label map (0 = background, 1..N = eggs).
"""
import copy
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULTS = {
    "blur": {"type": "none", "ksize": 5, "sigma_color": 50, "sigma_space": 7},
    "score": {"type": "gray", "chroma_weight": 2.0},
    "threshold": {"type": "otsu", "value": 128, "offset": 0, "block": 201, "C": -5},
    "morph": {"open": 0, "close": 0, "fill_holes": False},
    "filter": {"min_area_frac": 0.0, "max_area_frac": 1.0, "min_solidity": 0.0, "min_ellipse_iou": 0.0},
    # used when threshold.type == "sweep" (replaces threshold + morph + filter)
    "sweep": {"levels": 24, "p_lo": 3, "p_hi": 97, "polarity": "both", "open": 5, "edge_cut": False,
              "canny_sigma": 0.33, "min_area_frac": 0.015, "max_area_frac": 0.5, "min_solidity": 0.85,
              "max_aspect": 2.5, "min_ellipse_iou": 0.75, "min_quality": 0.35, "max_overlap": 0.2,
              "min_eggs": 1, "max_eggs": 6,
              "quality": "edge_support", "shape_lo": 0.75, "contrast_full": 50, "border_penalty": 0.6,
              "nest_ratio": 0.0},
    "split": {"type": "none", "min_distance": 25, "peak_rel": 0.6},
    "refine": {"grabcut_iters": 0, "band": 15, "ellipse": False},
}


# ----------------------------------------------------------------------------- config

def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path) -> dict:
    """Load a trial config. `base: <other config name>` inherits that config's pipeline first."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    pipeline = DEFAULTS
    if cfg.get("base"):
        pipeline = load_config(path.parent / f"{cfg['base']}.yaml")["pipeline"]
    cfg["pipeline"] = _merge(pipeline, cfg.get("pipeline", {}))
    cfg.setdefault("name", path.stem)
    return cfg


# ----------------------------------------------------------------------------- stages

def blur(img, c):
    k = c["ksize"] | 1   # odd
    if c["type"] == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    if c["type"] == "median":
        return cv2.medianBlur(img, k)
    if c["type"] == "bilateral":
        return cv2.bilateralFilter(img, c["sigma_space"], c["sigma_color"], c["sigma_space"])
    return img


def score_map(img, c):
    """uint8 map, high = egg-like."""
    if c["type"] == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if c["type"] == "hsv_whiteness":   # bright AND unsaturated
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
        return (255 * hsv[..., 2] * (1 - hsv[..., 1])).astype(np.uint8)
    if c["type"] == "lab_whiteness":   # lightness minus colourfulness
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        return np.clip(lab[..., 0] - c["chroma_weight"] * chroma, 0, 255).astype(np.uint8)
    if c["type"] == "smoothness":       # eggs are smooth; ash, leaves, cloth patterns, keys are not
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = (c.get("window", 11),) * 2
        std = np.sqrt(np.maximum(cv2.blur(g * g, k) - cv2.blur(g, k) ** 2, 0))
        return (255 - np.clip(std * c.get("gain", 8.0), 0, 255)).astype(np.uint8)
    raise ValueError(f"unknown score type {c['type']}")


def threshold(score, c):
    t = c["type"]
    if t == "otsu":
        thr, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return score > thr + c["offset"]
    if t == "fixed":
        return score > c["value"]
    if t == "adaptive":
        return cv2.adaptiveThreshold(score, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                     c["block"] | 1, c["C"]) > 0
    raise ValueError(f"unknown threshold type {t}")


def fill_holes(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, cv2.FILLED)
    return out > 0


def morph(mask, c):
    m = mask.astype(np.uint8)
    if c["open"]:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["open"],) * 2))
    if c["close"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["close"],) * 2))
    m = m > 0
    return fill_holes(m) if c["fill_holes"] else m


def ellipse_mask(contour, shape):
    m = np.zeros(shape, np.uint8)
    if len(contour) >= 5:
        cv2.ellipse(m, cv2.fitEllipse(contour), 1, cv2.FILLED)
    return m > 0


def filter_components(mask, c):
    """Keep blobs whose size and shape fit an egg. Returns an instance label map."""
    h, w = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    ok = (area >= c["min_area_frac"] * h * w) & (area <= c["max_area_frac"] * h * w)
    ok[0] = False
    for i in np.flatnonzero(ok) if (c["min_solidity"] or c["min_ellipse_iou"]) else []:
        x, y, bw, bh = stats[i, :4]
        comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)   # work on the bounding-box crop
        cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        if c["min_solidity"] and area[i] / max(cv2.contourArea(cv2.convexHull(cnt)), 1) < c["min_solidity"]:
            ok[i] = False
            continue
        if c["min_ellipse_iou"]:
            pad = np.pad(comp, max(bw, bh))   # room for an ellipse that overshoots the box
            e = ellipse_mask(cnt + max(bw, bh), pad.shape)
            p = pad > 0
            if (p & e).sum() / max((p | e).sum(), 1) < c["min_ellipse_iou"]:
                ok[i] = False
    new_id = np.zeros(n, np.uint16)
    new_id[ok] = np.arange(1, ok.sum() + 1)
    return new_id[lab]


def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    return cv2.Canny(gray, int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))


def sweep_candidates(img, score, c, return_candidates=False):
    """Threshold-sweep region search (MSER-like) with egg-shaped scoring.

    One global threshold fails when the egg is as bright as its background (white bag, tray)
    or darker than it (backlit). Instead, try many thresholds in both directions. Every blob
    produced is a candidate, scored by `quality`:
      "edge_support"   (t05-t08) ellipse_iou * fraction of the outline on a Canny edge
      "shape_contrast" (t09+)    shape^2 * contrast with a ring just outside * border penalty,
                                 where shape rescales ellipse IoU from [shape_lo, 0.97] to [0, 1]
    Candidates are then picked greedily, best first, with no overlaps.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = auto_canny(gray, c["canny_sigma"])
    near_edge = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    cut = cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0 if c["edge_cut"] else np.zeros((h, w), bool)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c["open"],) * 2) if c["open"] else None
    gray_f = gray.astype(np.float32)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))   # 12 px ring for the contrast term

    scores = score if isinstance(score, list) else [score]
    cands = []
    jobs = []
    for ch, sc in enumerate(scores):
        levels = np.unique(np.percentile(sc, np.linspace(c["p_lo"], c["p_hi"], c["levels"])).astype(int))
        # the first channel is swept in both directions, extra channels only upwards
        for pol in ([1, -1] if c["polarity"] == "both" and ch == 0 else [1]):
            jobs += [(sc, pol, t, ch) for t in levels]
    for sc, pol, t, ch in jobs:
        m = ((sc > t) if pol > 0 else (sc < t)) & ~cut
        m = m.astype(np.uint8)
        if k_open is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (c["min_area_frac"] * h * w <= area <= c["max_area_frac"] * h * w):
                continue
            x, y, bw, bh = stats[i, :4]
            comp = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
            if c["edge_cut"]:   # give back the rim removed by the edge cut
                comp = np.pad(comp, 2)
                comp = cv2.dilate(comp, np.ones((3, 3), np.uint8))
                x, y = x - 2, y - 2
            comp = fill_holes(comp).astype(np.uint8)
            cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
            if len(cnt) < 5:
                continue
            solidity = comp.sum() / max(cv2.contourArea(cv2.convexHull(cnt)), 1)
            if solidity < c["min_solidity"]:
                continue
            (ex, ey), (ea, eb), ang = cv2.fitEllipse(cnt)
            if min(ea, eb) <= 0 or max(ea, eb) / min(ea, eb) > c["max_aspect"]:
                continue
            pad = int(max(ea, eb))
            cp = np.pad(comp, pad) > 0
            e = np.zeros(cp.shape, np.uint8)
            cv2.ellipse(e, ((ex + pad, ey + pad), (ea, eb), ang), 1, cv2.FILLED)
            e_iou = (cp & (e > 0)).sum() / max((cp | (e > 0)).sum(), 1)
            if e_iou < c["min_ellipse_iou"]:
                continue
            pts = cnt[:, 0] + [x, y]
            inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            on_border = (pts[:, 0] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] <= 1) | (pts[:, 1] >= h - 2)
            use = inside & ~on_border            # the image border is not evidence either way
            support = near_edge[pts[use, 1], pts[use, 0]].mean() if use.any() else 0.0
            full = np.zeros((h, w), bool)
            ys, xs = np.nonzero(comp)
            ys, xs = ys + y, xs + x
            ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            full[ys[ok], xs[ok]] = True
            if c["quality"] == "shape_contrast":
                # sharper shape term, contrast with a ring just outside, penalty for touching the border
                # (chosen from analyze_candidates.py: AUC 0.988 vs 0.760 for ellipse_iou * edge_support)
                ring = (cv2.dilate(full.astype(np.uint8), ring_k) > 0) & ~full
                contrast = abs(gray_f[full].mean() - gray_f[ring].mean()) if ring.any() else 0.0
                shape = np.clip((e_iou - c["shape_lo"]) / (1 - c["shape_lo"] - 0.03), 0, 1)
                touches = ys.min() <= 0 or xs.min() <= 0 or ys.max() >= h - 1 or xs.max() >= w - 1
                quality = shape ** 2 * min(1.0, contrast / c["contrast_full"]) * (c["border_penalty"] if touches else 1.0)
            else:
                quality = e_iou * support
            cands.append({"mask": full, "quality": quality, "area": int(full.sum()),
                          "e_iou": e_iou, "support": support, "solidity": solidity, "t": int(t), "pol": pol,
                          "ch": ch})
    if return_candidates:
        return cands

    # greedy non-overlapping selection, best quality first
    cands.sort(key=lambda d: -d["quality"])
    label = np.zeros((h, w), np.uint16)
    k = 0
    for cd in cands:
        if cd["quality"] < c["min_quality"] and k >= c["min_eggs"]:
            break
        overlap = (cd["mask"] & (label > 0)).sum() / cd["area"]
        if overlap > c["max_overlap"]:
            continue
        if c["nest_ratio"]:
            cd = _largest_nested(cd, cands, label, c)
        k += 1
        label[cd["mask"] & (label == 0)] = k
        if k >= c["max_eggs"]:
            break
    return label


def _largest_nested(cd, cands, label, c):
    """Upgrade a picked blob to the largest candidate that contains it and is nearly as egg-like.

    The sweep produces nested blobs on one egg: a bright cap at a high threshold, the whole egg at
    a lower one. The cap often fits an ellipse slightly better and wins, leaving the egg half
    covered (t10b: 29 of 31 partial images were under-segmented).
    """
    best = cd
    for o in cands:
        if o["area"] <= best["area"] or o["quality"] < c["nest_ratio"] * cd["quality"]:
            continue
        if c.get("nest_max_growth") and o["area"] > c["nest_max_growth"] * cd["area"]:
            continue   # e.g. egg + the white paper it lies on (#32), egg + hand (#20)
        if (o["mask"] & cd["mask"]).sum() < 0.9 * cd["area"]:   # must contain the picked blob
            continue
        if (o["mask"] & (label > 0)).sum() / o["area"] > c["max_overlap"]:
            continue
        best = o
    return best


def candidate_features(img, mask, ring_px=12):
    """Appearance features of one candidate blob, compared with a ring of pixels just outside it."""
    h, w = mask.shape
    m8 = mask.astype(np.uint8)
    ring = (cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1,) * 2)) > 0) & ~mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    inner = cv2.erode(m8, np.ones((7, 7), np.uint8)) > 0
    inner = inner if inner.any() else mask
    ys, xs = np.nonzero(mask)
    return {
        "area_frac": mask.sum() / (h * w),
        "contrast": abs(gray[mask].mean() - gray[ring].mean()) if ring.any() else 0.0,
        "chroma": chroma[mask].mean(),
        "chroma_ring": chroma[ring].mean() if ring.any() else 0.0,
        "texture_in": lap[inner].mean(),
        "texture_ring": lap[ring].mean() if ring.any() else 0.0,
        "lightness": lab[..., 0][mask].mean(),
        "border_touch": float(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1),
    }


def sweep_channels(img, s, p):
    """Main score map plus any extra channels listed in sweep.extra_channels."""
    return [s, *(score_map(img, {**p["score"], "type": t}) for t in p["sweep"].get("extra_channels", []))]


def split_touching(label, c):
    """Distance-transform peaks as markers -> watershed, applied inside each blob."""
    if c["type"] != "watershed" or label.max() == 0:
        return label
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    fg = label > 0
    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    markers = np.zeros(label.shape, np.int32)
    k = 0
    for i in np.unique(label)[1:]:
        d = np.where(label == i, dist, 0)
        peaks = peak_local_max(d, min_distance=c["min_distance"], threshold_abs=c["peak_rel"] * d.max(),
                               exclude_border=False)
        if len(peaks) == 0:
            peaks = np.argwhere(d == d.max())[:1]
        for y, x in peaks:
            k += 1
            markers[y, x] = k
    ws = watershed(-dist, markers, mask=fg)
    return ws.astype(np.uint16)


def refine(img, label, c):
    if c["grabcut_iters"] and label.max():
        before = label
        label = _grabcut(img, label, c["grabcut_iters"], c["band"], c.get("grow_limit", 0))
        if c.get("max_growth"):
            label = _revert_bad_growth(before, label, c["max_growth"], c.get("min_ellipse_iou", 0))
    if c["ellipse"]:
        out = np.zeros_like(label)
        for i in np.unique(label)[1:]:
            comp = (label == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            out[ellipse_mask(max(cnts, key=cv2.contourArea), label.shape) & (out == 0)] = i
        label = out
    return label


def _revert_bad_growth(before, after, max_growth, min_ellipse_iou):
    """Per egg: keep the GrabCut result only if it is plausible, else the blob from before GrabCut.

    Implausible = area grew more than max_growth times, or the result is no longer
    ellipse-like. That is GrabCut running off into a hand, wall or paper of similar colour.
    """
    out = after.copy()
    for i in np.unique(before)[1:]:
        b, a = before == i, after == i
        bad = a.sum() > max_growth * b.sum()
        if not bad and min_ellipse_iou and a.any():
            cnts, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            e = ellipse_mask(max(cnts, key=cv2.contourArea), a.shape)
            bad = (a & e).sum() / max((a | e).sum(), 1) < min_ellipse_iou
        if bad:
            out[a] = 0
            out[b] = i
    return out


def _grabcut(img, label, iters, band, grow_limit=0):
    """GrabCut seeded from the current mask: eroded mask = sure egg, a band around it = unsure.

    grow_limit > 0: GrabCut may only grow inside each blob's fitted ellipse scaled by this
    factor, so it can't run off into a similar-coloured hand or wall.
    """
    fg = (label > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    gc = np.full(fg.shape, cv2.GC_BGD, np.uint8)
    unsure = cv2.dilate(fg, k) > 0
    if grow_limit:
        allowed = np.zeros(fg.shape, np.uint8)
        for i in np.unique(label)[1:]:
            cnts, _ = cv2.findContours((label == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(cnts, key=cv2.contourArea)
            if len(cnt) >= 5:
                (cx, cy), (a, b), ang = cv2.fitEllipse(cnt)
                cv2.ellipse(allowed, ((cx, cy), (a * grow_limit, b * grow_limit), ang), 1, cv2.FILLED)
        unsure &= (allowed > 0) | (fg > 0)
    gc[unsure] = cv2.GC_PR_BGD
    gc[fg > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(fg, k) > 0] = cv2.GC_FGD
    if not (gc == cv2.GC_FGD).any():
        return label
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)   # GrabCut's k-means init is random: seed per image so results don't depend on run order
    cv2.grabCut(img, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    new_fg = fill_holes((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
    # give refined pixels to the nearest original instance
    from skimage.segmentation import watershed
    dist = cv2.distanceTransform((label == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return watershed(dist, label.astype(np.int32), mask=new_fg).astype(np.uint16)


# ----------------------------------------------------------------------------- main entry

def segment(img_bgr, p, debug=False):
    x = blur(img_bgr, p["blur"])
    s = score_map(x, p["score"])
    if p["threshold"]["type"] == "sweep":
        m = None
        lab = sweep_candidates(x, sweep_channels(x, s, p), p["sweep"])
    else:
        m = morph(threshold(s, p["threshold"]), p["morph"])
        lab = filter_components(m, p["filter"])
    lab = split_touching(lab, p["split"])
    lab = refine(img_bgr, lab, p["refine"])
    if debug:
        return lab, {"score": s, "binary": m}
    return lab

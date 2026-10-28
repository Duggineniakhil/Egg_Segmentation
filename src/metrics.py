"""Segmentation metrics: SAM instance label maps are ground truth, classical outputs are predictions.

Both inputs are uint8 instance label maps (0 = background, 1..N = eggs).
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def binary_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    g, p = gt > 0, pred > 0
    tp = np.logical_and(g, p).sum()
    fp = np.logical_and(~g, p).sum()
    fn = np.logical_and(g, ~p).sum()
    eps = 1e-9
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn + eps),
    }


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tol: int = 3) -> float:
    """Boundary F-score: fraction of boundary pixels within `tol` px of the other mask's boundary."""
    def edges(m):
        m = (m > 0).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

    eg, ep = edges(gt), edges(pred)
    if not eg.any() or not ep.any():
        return float(eg.any() == ep.any())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    near_g = cv2.dilate(eg.astype(np.uint8), k) > 0
    near_p = cv2.dilate(ep.astype(np.uint8), k) > 0
    prec = (ep & near_g).sum() / ep.sum()
    rec = (eg & near_p).sum() / eg.sum()
    return 2 * prec * rec / (prec + rec + 1e-9)


def instance_metrics(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5) -> dict:
    """One-to-one Hungarian matching of eggs on mask IoU; a match counts if IoU >= iou_thr."""
    g_ids, g_inv = np.unique(gt, return_inverse=True)
    p_ids, p_inv = np.unique(pred, return_inverse=True)
    # pixel-count table: inter[a, b] = number of pixels with gt id a and pred id b
    inter = np.bincount(g_inv.ravel() * len(p_ids) + p_inv.ravel(),
                        minlength=len(g_ids) * len(p_ids)).reshape(len(g_ids), len(p_ids))
    area_g, area_p = inter.sum(1), inter.sum(0)
    gi, pi = g_ids != 0, p_ids != 0        # drop the background row/column
    inter, area_g, area_p = inter[gi][:, pi], area_g[gi], area_p[pi]
    g_ids, p_ids = g_ids[gi], p_ids[pi]
    ious = inter / np.maximum(area_g[:, None] + area_p[None, :] - inter, 1)
    matched = []
    if ious.size:
        r, c = linear_sum_assignment(-ious)
        matched = [ious[i, j] for i, j in zip(r, c) if ious[i, j] >= iou_thr]
    tp = len(matched)
    return {
        "n_gt": len(g_ids), "n_pred": len(p_ids),
        "tp": tp, "fp": len(p_ids) - tp, "fn": len(g_ids) - tp,
        "matched_iou_mean": float(np.mean(matched)) if matched else 0.0,
    }


def image_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    m = binary_metrics(gt, pred)
    m["boundary_f1"] = boundary_f1(gt, pred)
    m.update(instance_metrics(gt, pred))
    m["count_correct"] = int(m["n_gt"] == m["n_pred"])
    return {k: float(v) for k, v in m.items()}


def summarize(rows: list) -> dict:
    """Mean of per-image scores, plus dataset-level instance precision/recall/F1."""
    keys = ["iou", "dice", "precision", "recall", "boundary_f1", "count_correct"]
    s = {f"m{k}" if k != "count_correct" else "count_acc": float(np.mean([r[k] for r in rows])) for k in keys}
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    s["inst_precision"] = tp / (tp + fp) if tp + fp else 0.0
    s["inst_recall"] = tp / (tp + fn) if tp + fn else 0.0
    s["inst_f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return s

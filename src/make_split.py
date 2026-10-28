"""Task 2: leakage-aware 4:1 dev/eval split.

Leakage here comes from *scenes*, not augmentation. The 110 images are photos of a few
dozen physical setups (the same steel pot, ash heap, marble counter or pink bowl), shot
from slightly different angles and often under non-consecutive file numbers. A random
split would put near-copies of a scene in both dev and eval, so eval would measure how
well the algorithm remembers a background, not how well it generalizes.

Method
1. Blank out the eggs (SAM masks, dilated, inpainted). Every egg looks alike, so they
   would pull unrelated scenes together; the background is what identifies a scene.
2. Embed each background with CLIP ViT-B/32 (L2-normalized image features).
3. Agglomerative clustering (cosine distance, average linkage, distance threshold) gives
   the scene groups.
3b. Review outputs/split/scene_groups.jpg by eye. Clustering splits a few scenes into
   several groups: CLIP encodes what an object is more than its exact look. A coarser
   threshold merges those only after unrelated scenes have collapsed into one huge group,
   and adding an HSV color histogram fixed the cloth but split others. So the scenes
   confirmed by eye are merged by hand (SCENE_OVERRIDES).
4. Assign whole groups to eval: random search over group subsets for one with about 20%
   of the images and a similar share of multi-egg images.
5. Report leakage: for every eval image, its highest cosine similarity to any dev image,
   compared against random image-level splits.

Usage:
    python src/make_split.py                      # writes data/splits.json + outputs/split/
    python src/make_split.py --threshold 0.2      # coarser / finer scene groups
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"
MASK_DIR = ROOT / "outputs" / "sam_masks"
OUT_DIR = ROOT / "outputs" / "split"

# Physical scenes confirmed by eye that clustering split apart (image frame numbers).
# Each list becomes one group; images not listed keep their cluster.
SCENE_OVERRIDES = {
    "striped_cloth": [3, 4, 5, 6],
    "better_health_bag": [104, 109, 111, 112, 123, 125, 143],
    "orange_tub": [108, 110, 120, 127, 130, 139, 141],
    "pink_bowl_cherry_jar": [21, 22, 34, 35, 39, 40],
    "white_egg_tray": [42, 116, 117],
    "green_rug": [154, 162, 164],
    # found from nearest-dev-neighbour pairs of eval images after the first pass
    "steel_pot": [113, 114, 124, 128, 129, 135, 145, 147, 153, 159],
    "blue_lid": [156, 163],
    "green_door_hand": [160, 161],
}


def frame_no(stem: str) -> int:
    return int(re.search(r"(\d+)$", stem).group(1))


def background_only(img_bgr: np.ndarray, label: np.ndarray, dilate_px: int = 15) -> np.ndarray:
    """Remove the eggs (plus a margin for shadows and edges) by inpainting from the surroundings."""
    small = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    m = cv2.resize((label > 0).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    k = max(1, dilate_px * 256 // 640)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return cv2.inpaint(small, m * 255, 5, cv2.INPAINT_TELEA)


def embed(images_bgr: list) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    model, proc = CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)
    feats = []
    with torch.no_grad():
        for i in range(0, len(images_bgr), 16):
            batch = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images_bgr[i:i + 16]]
            f = model.get_image_features(**proc(images=batch, return_tensors="pt"))
            if not isinstance(f, torch.Tensor):   # newer transformers return a model output
                f = f.pooler_output
            feats.append(torch.nn.functional.normalize(f, dim=-1).numpy())
    return np.concatenate(feats)


def cluster(emb: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=threshold).fit_predict(emb)


def apply_overrides(stems, groups: np.ndarray) -> np.ndarray:
    groups = groups.copy()
    nums = [frame_no(s) for s in stems]
    for members in SCENE_OVERRIDES.values():
        gid = groups.max() + 1
        for n in members:
            groups[nums.index(n)] = gid
    # relabel to 0..G-1, largest group first
    ids, counts = np.unique(groups, return_counts=True)
    order = {g: i for i, g in enumerate(ids[np.argsort(-counts, kind="stable")])}
    return np.array([order[g] for g in groups])


def choose_eval_groups(groups: np.ndarray, multi: np.ndarray, eval_frac: float, seed: int, trials: int):
    """Random search over group subsets: about eval_frac of the images, similar multi-egg share."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    target = eval_frac * len(groups)
    best, best_cost = None, np.inf
    for _ in range(trials):
        chosen, n = [], 0
        for g in rng.permutation(ids):
            size = (groups == g).sum()
            if n + size <= target + 2:
                chosen.append(g)
                n += size
        ev = np.isin(groups, chosen)
        cost = abs(ev.sum() - target) + 20 * abs(multi[ev].mean() - multi[~ev].mean())
        if cost < best_cost:
            best, best_cost = ev, cost
    return best


def max_sim_to_dev(emb: np.ndarray, ev: np.ndarray) -> np.ndarray:
    return (emb[ev] @ emb[~ev].T).max(1)


def group_sheet(stems, groups, path: Path, tile: int = 110, per_row: int = 12) -> None:
    rows = []
    for g in sorted(np.unique(groups), key=lambda g: -(groups == g).sum()):
        members = [s for s, gg in zip(stems, groups) if gg == g]
        for i in range(0, len(members), per_row):
            row = []
            for s in members[i:i + per_row]:
                im = cv2.resize(cv2.imread(str(IMG_DIR / f"{s}.jpg")), (tile, tile))
                cv2.putText(im, str(frame_no(s)), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                row.append(im)
            label = np.full((tile, 40, 3), 255, np.uint8)
            cv2.putText(label, f"G{g}" if i == 0 else "", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            row = [label] + row + [np.full((tile, tile, 3), 255, np.uint8)] * (per_row - len(row))
            rows.append(np.hstack(row))
        rows.append(np.zeros((3, rows[-1].shape[1], 3), np.uint8))
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.15, help="cosine distance for merging scenes")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=20000)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stems = sorted((p.stem for p in IMG_DIR.glob("*.jpg")), key=frame_no)
    labels = [cv2.imread(str(MASK_DIR / f"{s}.png"), cv2.IMREAD_GRAYSCALE) for s in stems]
    multi = np.array([lab.max() > 1 for lab in labels])

    cache = OUT_DIR / "embeddings.npz"
    if cache.exists() and list(np.load(cache)["stems"]) == stems:
        emb = np.load(cache)["emb"]
    else:
        bgs = [background_only(cv2.imread(str(IMG_DIR / f"{s}.jpg")), lab) for s, lab in zip(stems, labels)]
        emb = embed(bgs)
        np.savez(cache, emb=emb, stems=np.array(stems))

    groups = apply_overrides(stems, cluster(emb, args.threshold))
    ev =choose_eval_groups(groups, multi, args.eval_frac, args.seed, args.trials)

    # Leakage: nearest-dev similarity of eval images, ours vs. random image-level splits
    ours = max_sim_to_dev(emb, ev)
    rng = np.random.default_rng(args.seed)
    rand = []
    for _ in range(200):
        r = np.zeros(len(stems), bool)
        r[rng.choice(len(stems), ev.sum(), replace=False)] = True
        rand.append(max_sim_to_dev(emb, r).mean())

    split = {
        "dev": [s for s, e in zip(stems, ev) if not e],
        "eval": [s for s, e in zip(stems, ev) if e],
        "groups": {s: int(g) for s, g in zip(stems, groups)},
        "params": vars(args) | {"embedding": "clip-vit-base-patch32 on egg-inpainted background",
                                "clustering": "agglomerative, cosine, average linkage",
                                "scene_overrides": SCENE_OVERRIDES},
        "stats": {
            "n_groups": int(len(np.unique(groups))),
            "n_dev": int((~ev).sum()), "n_eval": int(ev.sum()),
            "multi_egg_frac_dev": round(float(multi[~ev].mean()), 3),
            "multi_egg_frac_eval": round(float(multi[ev].mean()), 3),
            "eval_max_sim_to_dev_mean": round(float(ours.mean()), 4),
            "random_split_max_sim_to_dev_mean": round(float(np.mean(rand)), 4),
        },
    }
    (ROOT / "data" / "splits.json").write_text(json.dumps(split, indent=1))
    group_sheet(stems, groups, OUT_DIR / "scene_groups.jpg")
    print(json.dumps(split["stats"], indent=1))


if __name__ == "__main__":
    main()

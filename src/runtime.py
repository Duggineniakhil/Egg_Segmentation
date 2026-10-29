"""Runtime of the classical pipeline per image and per stage, single process on CPU (dev images).

Stages are timed by running the pipeline with later stages switched off and taking differences:
  sweep only (no GrabCut)  ->  + GrabCut
  without the dark-polarity sweep (which the ablation shows adds nothing)

    python src/runtime.py                 # 20 dev images
    python src/runtime.py --n 87
"""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from classical_seg import _merge, load_config, segment

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "configs" / "t16_nest_cap4.yaml"

VARIANTS = {
    "full pipeline": {},
    "without GrabCut": {"refine": {"grabcut_iters": 0}},
    "without dark polarity": {"sweep": {"polarity": "up"}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    cv2.setNumThreads(1)
    base = load_config(FINAL)["pipeline"]
    dev = json.loads((ROOT / "data" / "splits.json").read_text())["dev"][: args.n]
    imgs = [cv2.imread(str(ROOT / "data" / "images" / f"{s}.jpg")) for s in dev]
    out = {}
    for name, ov in VARIANTS.items():
        p = _merge(base, ov)
        segment(imgs[0], p)                       # warm-up
        times = []
        for im in imgs:
            t0 = time.perf_counter()
            segment(im, p)
            times.append(time.perf_counter() - t0)
        out[name] = {"mean_s": round(float(np.mean(times)), 3), "median_s": round(float(np.median(times)), 3),
                     "max_s": round(float(np.max(times)), 3)}
        print(f"{name:24s} mean {np.mean(times):.2f} s  median {np.median(times):.2f} s  max {np.max(times):.2f} s")
    out["machine"] = f"{platform.processor() or platform.machine()}, 1 thread, {args.n} images 640x640"
    (ROOT / "outputs" / "ablation" / "runtime.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

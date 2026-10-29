"""Check tracking output against hand-annotated identity checkpoints for the Task 4 video.

The checkpoints were written down by watching the video frame by frame (see outputs/viz close-ups):
  Egg A (raw egg): starts on the left, is picked up (~20 s), spun on the right (~23-27 s), picked up
                   again (~27 s), held at the far left, and put down there during the handover (~31 s).
  Egg B (boiled):  sits still at x~157 until ~30 s, is taken by the same hand during the handover,
                   carried right and spun there (~34-42 s).
Coordinates are in 640-px-wide video pixels (tracks.csv is in output pixels; pass --scale for HD).

    python src/check_tracks.py outputs/video/v2/tracks.csv
    python src/check_tracks.py /kaggle/working/tracking/tracks.csv --scale 3
"""
import argparse
import sys

import pandas as pd

# (first frame, last frame, egg, x range, y range) -- egg A should be ID 1, egg B ID 2 (IDs are given in
# order of first appearance: A is the left egg at the start). The y range keeps an egg carried *over*
# another one in the hand from counting as being at its spot.
CHECKPOINTS = [
    (230, 470, "A", (30, 110), (170, 240)),     # both on the table, A left
    (230, 470, "B", (130, 190), (190, 250)),
    (570, 640, "A", (380, 460), (210, 280)),    # A spun on the right
    (570, 700, "B", (130, 190), (190, 250)),    # B still in place
    (760, 990, "A", (40, 110), (170, 230)),     # after the handover: A left at the far left ...
    (820, 990, "B", (360, 460), (220, 290)),    # ... B carried right and spun
]
EXPECTED_ID = {"A": 1, "B": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", help="tracks.csv from track_video.py")
    ap.add_argument("--scale", type=float, default=1.0, help="output width / 640")
    args = ap.parse_args()
    t = pd.read_csv(args.tracks)
    t["x"] = t.cx / args.scale
    t["y"] = t.cy / args.scale
    ok = True
    ids = sorted(t.id.unique())
    print(f"IDs used: {ids}")
    if len(ids) != 2:
        print("  FAIL: expected exactly 2 IDs")
        ok = False
    for f0, f1, egg, (x0, x1), (y0, y1) in CHECKPOINTS:
        seg = t[t.frame.between(f0, f1)]
        at = seg.x.between(x0, x1) & seg.y.between(y0, y1)
        mine = seg[seg.id == EXPECTED_ID[egg]]
        inside = at[mine.index].mean() if len(mine) else 0.0
        others = seg[(seg.id != EXPECTED_ID[egg]) & at]
        res = inside >= 0.95 and len(others) == 0
        ok &= res
        print(f"  frames {f0}-{f1} egg {egg}: ID {EXPECTED_ID[egg]} at x {x0}-{x1}, y {y0}-{y1} in {inside:.0%} of frames, "
              f"other IDs there: {len(others)}  {'OK' if res else 'FAIL'}")
    dup = t.groupby("frame").id.apply(lambda s: s.duplicated().any()).sum()
    print(f"  frames with a duplicated ID: {dup}  {'OK' if dup == 0 else 'FAIL'}")
    ok &= dup == 0
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

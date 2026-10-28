"""Download the egg dataset and flatten it into data/images/.

The Roboflow export is in classification-folder format (train/valid/test, classes
"egg" and "Unlabeled") with no box or mask annotations. Those folders mean nothing
for segmentation, so all 110 images are copied into one folder under their
original names (e.g. not_damaged_101.jpg). The dev/eval split is made later by
make_split.py.

Usage:
    python src/prepare_data.py                # download + extract + flatten
    python src/prepare_data.py --zip path.zip # use an already downloaded zip
"""
import argparse
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

DRIVE_ID = "1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv"
DRIVE_URL = f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def download(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {dst} ...")
    urllib.request.urlretrieve(DRIVE_URL, dst)


def extract(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"Unsafe path in zip: {name}")
        z.extractall(out_dir)


def flatten(raw_dir: Path, img_dir: Path) -> int:
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(raw_dir.rglob("*.jpg")):
        # not_damaged_101_jpg.rf.<hash>.jpg -> not_damaged_101.jpg
        stem = re.sub(r"_jpg\.rf\..*$", "", f.stem)
        shutil.copy2(f, img_dir / f"{stem}.jpg")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DATA / "raw_download" / "dataset.zip")
    args = ap.parse_args()

    if not args.zip.exists():
        download(args.zip)
    raw_dir = DATA / "raw"
    extract(args.zip, raw_dir)
    n = flatten(raw_dir, DATA / "images")
    print(f"{n} images written to {DATA / 'images'}")


if __name__ == "__main__":
    main()

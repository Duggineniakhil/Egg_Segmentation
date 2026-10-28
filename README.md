# Egg Segmentation: SAM vs. Classical Image Processing

CV intern assignment covering:
1. **Task 1**: pseudo ground-truth egg masks from a pretrained segmentation model (SAM).
2. **Task 2**: classical image-processing segmentation with a leakage-aware 4:1 dev/eval split and an experiment log.
3. **Task 3**: evaluation of the classical method against the SAM masks (quantitative and qualitative).
4. **Task 4**: per-egg tracking on a YouTube video.

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Dataset
[Roboflow "egg" v1](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1) (CC BY 4.0):
110 images at 640×640, 1–4 white eggs per image, varied backgrounds. The export has
**no box or mask labels** (classification-folder format only), so the masks for
Task 1 are produced from scratch.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python src/prepare_data.py      # downloads + flattens images into data/images/
```

## Repository layout
```
src/          scripts for each task
configs/      one YAML per classical-segmentation trial (experiment log)
data/         images (downloaded) and splits.json
outputs/      masks, visualizations, metrics, video
report/       PDF report
```

## Running the tasks
_Commands are added here as each task is implemented._

## AI tool disclosure
_To be completed._

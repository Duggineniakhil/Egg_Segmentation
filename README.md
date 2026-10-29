# Egg Segmentation: SAM vs. Classical Image Processing

CV intern assignment covering:
1. **Task 1**: pseudo ground-truth egg masks from a pretrained segmentation model (SAM).
2. **Task 2**: classical image-processing segmentation with a leakage-aware 4:1 dev/eval split and an experiment log.
3. **Task 3**: evaluation of the classical method against the SAM masks (quantitative and qualitative).
4. **Task 4**: per-egg tracking on a YouTube video.

The full write-up is in [`report/report.pdf`](report/report.pdf).
**Tracking output video:** [tracked_eggs.mp4 on Google Drive](https://drive.google.com/file/d/1-TJ6u60Hdvg0gUGsSb1UORi55hf84DDj/view?usp=sharing).

To rebuild the report from `report/report.md` (needs Chrome or Edge for the PDF step):
```bash
python src/build_report.py --drive-link "https://drive.google.com/file/d/1-TJ6u60Hdvg0gUGsSb1UORi55hf84DDj/view?usp=sharing" --author "Duggineni Akhil"
```

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
To pull the data through the Roboflow API instead, set `ROBOFLOW_API_KEY` and run
`python src/prepare_data.py --source roboflow`.

## Repository layout
```
src/          scripts for each task
configs/      one YAML per classical-segmentation trial (experiment log)
data/         images (downloaded) and splits.json
outputs/      masks, visualizations, metrics, video
report/       PDF report
```

## Running the tasks

### Task 1: SAM pseudo ground truth (GPU, Kaggle)
[`notebooks/task1_sam_kaggle.ipynb`](notebooks/task1_sam_kaggle.ipynb) uses Grounding DINO (`"an egg."`)
boxes as prompts for SAM ViT-H.
1. On Kaggle: **New Notebook → File → Import Notebook**, then upload the notebook.
2. Settings: **Accelerator = GPU T4**, **Internet = On**. Under **Add-ons → Secrets**, add `ROBOFLOW_API_KEY`.
3. Run all, look over the review sheets, and add any fixes to `MANUAL_BOXES`, then re-run.
4. Download `sam_outputs.zip` and unzip it into `outputs/`.

Masks are saved as `outputs/sam_masks/<image>.png`. Each is an instance label map: 0 is background and 1..N
are the eggs, so the binary mask is `label > 0`. The masks are committed to the repo, so later tasks don't need a GPU.

### Task 2a: leakage-aware dev/eval split
```bash
python src/make_split.py
```
The images are photos of a few dozen physical scenes, so the split is made by scene, not by image:
1. The eggs are inpainted out using the SAM masks.
2. CLIP ViT-B/32 embeds each background.
3. Agglomerative clustering (cosine distance 0.15) groups them into scenes, and scenes confirmed by eye that clustering
   split apart are merged by hand (`SCENE_OVERRIDES`).
4. Whole scenes are assigned to eval.

The result goes to `data/splits.json`, and `outputs/split/scene_groups.jpg` shows the groups.

| | dev | eval |
|---|---|---|
| images | 87 | 23 |
| scenes | 19 | 4 (steel pot, orange tub, striped cloth, green door) |
| multi-egg images | 13.8% | 21.7% |

Leakage check: an eval image's mean highest CLIP similarity to any dev image is **0.859**, against **0.911** for a
random image-level split.

### Task 2b: classical segmentation and experiment log
```bash
python src/run_experiment.py configs/t16_nest_cap4.yaml        # final algorithm on dev, appends to the log
python src/diagnose_sweep.py configs/t16_nest_cap4.yaml        # failures: candidate generation vs selection
python src/analyze_candidates.py configs/t16_nest_cap4.yaml    # which candidate features separate eggs
```
Each trial is a YAML file in `configs/` that inherits from its predecessor (`base:`) and states what changed and
why. Results, with keep/discard decisions, are in `outputs/experiments/log.csv`, and each trial's worst dev images
are in `outputs/experiments/<trial>/worst.jpg`. The algorithm ([`src/classical_seg.py`](src/classical_seg.py)):
1. Gaussian blur.
2. A threshold sweep (24 levels) over three channels: gray (both polarities), smoothness, and Lab whiteness. The
   masks are cut along Canny edges.
3. Every egg-sized, convex, ellipse-like blob becomes a candidate, scored by
   `shape² × ring contrast × border penalty`.
4. Greedy non-overlapping selection, which upgrades each pick to the largest nested candidate that is nearly as
   egg-like (at most 4× bigger).
5. GrabCut boundary refinement with a 30 px band.

Dev mIoU went from 0.339 (gray + Otsu) to **0.786**. The log shows 0.785 for the final trial, which ran before
GrabCut's random initialisation was seeded per image.

### Task 3: evaluation and analysis
```bash
python src/run_experiment.py configs/t16_nest_cap4.yaml --split eval --save-masks   # final model, eval, run once
python src/run_experiment.py configs/t16_nest_cap4.yaml --save-masks --no-log       # dev masks for the analysis
python src/evaluate.py
```
On held-out eval, the final algorithm scores **0.789 mIoU**, 0.855 Dice, 0.929 instance F1 and 100% egg-count
accuracy (dev: 0.786 mIoU). The write-up is [`outputs/eval/analysis.md`](outputs/eval/analysis.md), and the plots
and failure galleries are in `outputs/eval/`.

**Robustness** (section 6 of the analysis):
```bash
python src/ablation.py      # remove one component at a time (dev) -> outputs/ablation/
python src/stats.py         # scene-level bootstrap CIs and scene-grouped folds -> outputs/stats/
python src/runtime.py       # per-image runtime per stage, 1 CPU thread
```
* **Eval mIoU 0.789, 95% CI [0.683, 0.852]**, resampling whole scenes. Across 5 scene-grouped folds of all
  110 images it is 0.787 ± 0.070.
* **Ablation:** the candidate score (−0.270 when removed) and the smoothness channel (−0.085) matter most. The
  dark-polarity sweep adds nothing (+0.003 when removed).
* **Runtime:** 4.4 s per image on one CPU thread, against 2.6 s per image for SAM on a T4 GPU.

**v2 experiment** (section 7 of the analysis): ellipse completion fitted to the edge-supported outline points,
for eggs cut along their lit/shadow line (`configs/t17c_complete_chroma.yaml`).
* **Dev:** 0.786 → 0.809 mIoU.
* **Eval (run once, after v1's evaluation):** 0.786 vs 0.789. It helps 3 of the 4 eval scenes but extends onto the
  reflections in the steel pot.
* **Decision:** v1 remains the submitted model.

### Task 4: egg tracking in video
[`notebooks/task4_tracking_kaggle.ipynb`](notebooks/task4_tracking_kaggle.ipynb) runs it on Kaggle. It needs
Internet On but no GPU. The notebook clones this repo, downloads the
[video](https://www.youtube.com/watch?v=Y4JyV94y2_A) with yt-dlp, and produces the annotated video. To run locally:
```bash
yt-dlp -f 18 -o data/video/egg_video.mp4 "https://www.youtube.com/watch?v=Y4JyV94y2_A"
python src/track_video.py --video data/video/egg_video.mp4 --config configs/video_v1.yaml
```
* **Detection:** the Task 2 algorithm with video settings (`configs/video_v1.yaml`): smaller size limits, since the
  eggs cover about 0.5% of a frame, and no forced egg, so the title cards stay empty.
* **Tracking ([`src/track_video.py`](src/track_video.py)):** each egg keeps one ID for the whole video.
  * **Visible egg:** a Kalman filter predicts its next position. Detections are matched with Hungarian
    assignment on distance minus box IoU, limited by the egg's typical size, its vertical movement and its area.
  * **Hidden or held egg:** the track is kept. Its estimate is moved by the optical flow of whatever covers it,
    so under a hand it is carried along with the hand. It then locks onto the visible white part of the egg in
    the hand. The label stays on the egg, with a dashed box, at the estimated position.
  * **Recovery:** a hidden egg is matched to a whole-egg detection near its estimate, never to another egg or
    to a reflection. New IDs are only handed out when no hidden egg can explain the detection.
  * **Validation:** `python src/check_tracks.py outputs/video/tracks.csv` checks the IDs against checkpoints
    written down by watching the video frame by frame. That includes the 30 s handover, where one hand swaps
    which egg it holds. All parameters are in `PARAMS`, with overrides in the config's `tracker:` section.
  * **Limitation:** the eggs look identical, so identity comes from motion continuity alone. Recoveries that
    aren't backed by it are flagged `low_confidence` in `tracking_log.csv`.
* **Outputs:** an annotated video, `tracks.csv`, `tracking_log.csv`, `summary.json` and `debug_frames/` in
  `outputs/video/`. Old-vs-new comparison frames are in `outputs/tracking_debug/`.
* **Quantitative evaluation** ([`outputs/video_gt/tracking_eval.md`](outputs/video_gt/tracking_eval.md)): SAM ground
  truth on every 4th frame (`notebooks/task4_video_gt_kaggle.ipynb`, then `python src/build_video_gt.py`), scored
  with `python src/eval_tracking.py <tracks.csv> [--scale 3 for 1080p]`.
  * **Final tracker:** MOTA **0.962**, IDF1 **0.953**, **0 ID switches**, and 85% of held-egg frames followed.
  * **Previous tracker:** MOTA 0.882, IDF1 0.620, with wrong IDs on every sampled frame after the 30 s handover.

## AI tool disclosure
_To be completed._

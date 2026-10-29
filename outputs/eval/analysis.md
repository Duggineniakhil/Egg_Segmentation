# Task 3: Evaluation and analysis

**Ground truth**: the Task 1 SAM masks (`outputs/sam_masks`). **Predictions**: the final classical algorithm,
`configs/t16_nest_cap4.yaml` (`outputs/cv_masks`). The algorithm was frozen after the dev experiments and run on the
eval split once. Everything here is produced by `python src/evaluate.py`.

## 1. Quantitative results

| Metric | dev (87 img, 110 eggs) | **eval (23 img, 28 eggs)** |
|---|---|---|
| mean IoU | 0.785 | **0.788** |
| mean Dice | 0.836 | **0.854** |
| pixel precision | 0.895 | **0.900** |
| pixel recall | 0.825 | **0.834** |
| boundary F1 (3 px tolerance) | 0.702 | **0.635** |
| HD95, median (px) | 19.7 | **23.0** |
| instance F1 (Hungarian matching, IoU ≥ 0.5) | 0.885 | **0.929** |
| correct egg count | 97.7% | **100%** |

**Why these metrics:**
- **IoU and Dice** measure overall mask agreement.
- **Precision and recall** say *which way* a mask is wrong: too much, or too little.
- **Boundary F1 and HD95** measure contour quality, which region overlap hides.
- **Instance F1 and count accuracy** check that each egg is found as a separate object, which matters for tracking in Task 4.

**Dev vs eval:** eval matches dev (0.788 vs 0.785 mIoU). Despite 27 tuning trials on dev, the algorithm did not overfit
to the dev scenes. Two caveats:
- **Small, correlated eval set:** eval is only 4 scenes and 23 images, so its mean has wide error bars. A single
  image (#139) costs it 0.04 mIoU.
- **Boundaries are where eval is weaker:** boundary F1 is 0.635 on eval vs 0.702 on dev. The eval scenes (steel pot
  with water, striped cloth) have reflective or patterned surroundings, which make the contours less exact.

Across all 138 eggs, 88% are matched with IoU ≥ 0.5. Mean HD95 is far higher than the median (59 vs about 20 px)
because a few complete misses dominate it.

![IoU distribution](fig_iou_hist.png)

## 2. Where the error comes from

Each image is put into one category from its pixel precision P and recall R:
- **good:** IoU ≥ 0.8.
- **under-segmented:** P ≥ 0.9 and R < 0.8.
- **leak:** P < 0.75 and R ≥ 0.75.
- **missed / wrong object:** R < 0.3.
- **mixed:** everything else.

| Category | images (dev + eval) | share of all lost IoU |
|---|---|---|
| good | 73 | 14% |
| **under-segmented** | 22 | **39%** |
| missed / wrong object | 8 | 32% |
| leak | 5 | 10% |
| mixed | 2 | 5% |

![precision vs recall](fig_precision_recall.png) ![categories](fig_categories.png)

**The dominant error is under-segmentation, not false detections.** The precision-vs-recall plot shows it: most
imperfect images sit in the top-left band. Almost everything predicted is egg, but part of the egg is missing.

## 3. What makes an image hard

**1. Contrast with the surroundings is the main driver** (Spearman ρ = 0.57 between egg IoU and contrast,
p = 3·10⁻¹³). Contrast here is egg brightness minus a 15 px ring around it.

| Contrast | eggs | mean IoU |
|---|---|---|
| ≥ 50 gray levels | 102 | 0.89 |
| 0–50 | 33 | 0.56 |
| < 0 (backlit: egg darker than background) | 3 | 0.37 |

![IoU vs contrast](fig_iou_vs_contrast.png)

**2. Partly white backgrounds are hardest.**

| Background (share of white pixels in a 40 px ring around the eggs) | images | mean IoU |
|---|---|---|
| < 20% white | 66 | 0.88 |
| 20–60% white | 37 | 0.63 |
| > 60% white | 7 | 0.73 |

Fully white surroundings do a bit better, probably because the egg's shadow then gives a clean edge.

![IoU vs white background](fig_iou_vs_white_bg.png)

**3. Touching eggs:** they score lower (0.68 vs 0.81 mean IoU, 13 eggs). The shared boundary has little contrast.
Images with several eggs still average higher (0.885 vs 0.767), because most of them are on the high-contrast marble
counter.

**4. Egg size has no clear effect** (mean IoU 0.77–0.86 across size bins).

![IoU vs size](fig_iou_vs_size.png)

**5. By scene:** the worst scenes are:

| Scene | images | mean IoU |
|---|---|---|
| eggs held against a bright wall (#20, #24, #44) | 3 | 0.37 |
| red lid / steel tray | 3 | 0.49 |
| white egg tray | 3 | 0.54 |
| "Better Health" bag | 7 | 0.58 |

![IoU by scene](fig_iou_by_scene.png)

## 4. Qualitative analysis: what went wrong

**Under-segmented** (`gallery_under_segmented.jpg`)

In #109, #31, #126, #5 and #142, the prediction stops at the **line between the lit and shadowed sides of the egg**.
A white egg under one-sided light has two very different brightness levels. At every threshold in the sweep the two
halves are separate blobs, and the lit half alone is already a good ellipse, so it gets picked. GrabCut then can't
cross into the shadowed half, because its colour model learned "egg = bright".

The nested-upgrade step (t11) fixes this only when some candidate covers both halves.

**Missed / wrong object** (`gallery_missed.jpg`)

| Cause | Images | What happens |
|---|---|---|
| White on white | #106, #125 | Egg and bag have the same brightness and no texture difference. The edge between them is too weak, so no threshold isolates the egg. The forced pick (at least one egg per image) then lands on a patch of bag. |
| Occlusion | #11, #117 | A twig or the tray rod cuts the egg into pieces, and none of the pieces is egg-shaped on its own. |
| Close-up held egg | #44 | The egg fills most of the frame and merges with the bright background at every threshold. |
| Edge of the frame | #139 (eval) | The egg is cut by the image edge against a light wall, and the forced pick lands on the tub rim. |

**Leak** (`gallery_leak.jpg`): the egg itself is found completely (recall 0.9–1.0), but the mask also takes in
some background.

| Cause | Images | What happens |
|---|---|---|
| Backlit egg in a hand | #20 | The selected dark blob is egg plus hand. GrabCut can't fix a wrong starting region. (#155 is similar but scores as "mixed".) |
| Ash-covered eggs | #27, #78 | The grey, ash-dusted top of the egg looks like the ash heap, so GrabCut spreads into the heap. |
| Reflective steel and water | #145, #159 (eval) | Bright reflections next to the egg join the mask. |

**Backlighting doesn't always fail:** #24 is backlit and scores 0.98, because the egg is a clean dark ellipse against
the bright wall. It fails when the dark egg touches an equally dark hand (#20) or fills the frame (#44).

## 5. Performance gaps and potential improvements

| Gap | Evidence | Improvement |
|---|---|---|
| Lit/shadow split on one egg | 39% of lost IoU; recall 0.6 with precision ≈ 1 | Normalize the lighting before the sweep: divide by a large-blur estimate of the illumination, or use chromaticity (r, g), which ignores shading. Or **merge neighbouring candidates whose union fits one ellipse better** than either part. |
| Low-contrast, white-on-white eggs | IoU 0.56 at contrast < 50 | Use the egg's own **shadow** as a cue (a dark crescent beside a bright blob), or add a gradient-based ellipse detector (e.g. randomized Hough on Canny edges) that doesn't depend on region thresholds. |
| Occlusion | #11, #117 | Fit an ellipse to *several* nearby egg-coloured pieces together (a RANSAC ellipse over their outline points). |
| Forced pick lands on background | #106, #125, #139 | Keep the at-least-one-egg rule (t14 showed dropping it costs 0.17 mIoU), but rank the forced pick by a stricter shape test than the normal threshold. |
| GrabCut leaking into skin, ash or reflections | #20, #27, #78, #145, #159 | Add a skin-colour model (YCrCb range) as GrabCut background seeds. Reject refinements that make the mask less elliptical (t15 tried this with fixed thresholds and saw no effect, because the main leaks then came from selection; after the t16 cap it is worth retrying). |
| Boundary accuracy | boundary F1 0.64–0.70 | Fit active contours (snakes) or an ellipse to the final mask, scored against the image gradient. |

**Limitation of the ground truth:** SAM's masks are not perfect labels. For #117 they needed manual prompts, and in
#123 and #125 SAM leaves out the part of the egg hidden behind paper. In a few images the classical method is marked
wrong where SAM itself is unsure. The numbers above measure agreement with SAM, not with hand-drawn labels.

"""Build report.pdf (repository root) from report/report.md.

The text is in report/report.md (kept locally; only the PDF is published in the repository). This script:
  1. fills in generated parts: {{EXPERIMENT_LOG}} (from outputs/experiments/log.csv), {{DRIVE_LINK}}, {{AUTHOR}}
  2. draws figures that only the report needs (trial progress chart)
  3. copies every figure the report uses into report/figures/, so the report folder is self-contained
  4. converts Markdown to HTML (report/report.html) and prints it to PDF with headless Chrome or Edge

    python src/build_report.py --drive-link "https://drive.google.com/..." --author "Your Name"
"""
import argparse
import html
import re
import shutil
import subprocess
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG = REPORT / "figures"

# figures used in report.md: name in report/figures -> source file
FIGURES = {
    "contact_sheet.jpg": "outputs/viz/contact_sheet.jpg",
    "sam_review.jpg": "outputs/review_sheet_1.jpg",
    "sam_fixed.jpg": "outputs/viz/sam_fixed.jpg",
    "final_worst.jpg": "outputs/experiments/t16_nest_cap4/worst.jpg",
    "precision_recall.png": "outputs/eval/fig_precision_recall.png",
    "categories.png": "outputs/eval/fig_categories.png",
    "iou_vs_contrast.png": "outputs/eval/fig_iou_vs_contrast.png",
    "iou_vs_white_bg.png": "outputs/eval/fig_iou_vs_white_bg.png",
    "iou_by_scene.png": "outputs/eval/fig_iou_by_scene.png",
    "gallery_under.jpg": "outputs/eval/gallery_under_segmented.jpg",
    "gallery_missed.jpg": "outputs/eval/gallery_missed.jpg",
    "gallery_leak.jpg": "outputs/eval/gallery_leak.jpg",
    "ablation.png": "outputs/ablation/fig_ablation.png",
    "ci.png": "outputs/stats/fig_ci.png",
    "v2_eval.jpg": "outputs/eval/v2_eval_changes.jpg",
    "id_timeline.png": "outputs/video_gt/fig_id_timeline.png",
    "track_750.jpg": "outputs/tracking_debug/old_vs_new_frame_750.jpg",
    "track_690.jpg": "outputs/tracking_debug/old_vs_new_frame_690.jpg",
    "handover.jpg": "outputs/viz/handover_closeup.jpg",
    "held_frames.jpg": "outputs/tracking_debug/held_frames_hd.jpg",
}

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; padding-bottom: 3pt; border-bottom: 1.5px solid #2a78d6; page-break-before: always; }
h1.title { font-size: 21pt; border: none; page-break-before: avoid; margin-bottom: 2pt; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 12pt; margin: 12pt 0 3pt 0; color: #123a6b; }
h3 { font-size: 10.5pt; margin: 9pt 0 2pt 0; }
p, li { margin: 2pt 0 4pt 0; }
ul, ol { margin: 2pt 0 5pt 0; padding-left: 16pt; }
table { border-collapse: collapse; margin: 5pt 0 8pt 0; font-size: 8.6pt; width: 100%; page-break-inside: avoid; }
th { background: #eef3fa; text-align: left; font-weight: 600; }
th, td { border: 1px solid #d6dbe3; padding: 2.5pt 4.5pt; vertical-align: top; }
td:not(:first-child) { white-space: normal; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.4pt; background: #f2f3f5; padding: 0 2pt; border-radius: 2px; }
pre { background: #f2f3f5; padding: 5pt 7pt; font-size: 8.2pt; border-radius: 3px; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 5pt auto 2pt auto; page-break-inside: avoid; }
img.half { max-width: 49%; display: inline-block; margin: 3pt 0.5%; }
.caption, p.caption { font-size: 8.3pt; color: #555; text-align: center; margin: 0 0 9pt 0; }
.meta { color: #555; margin-bottom: 10pt; }
.box { background: #f4f8fd; border-left: 3px solid #2a78d6; padding: 5pt 9pt; margin: 6pt 0 9pt 0; }
a { color: #1a5fb4; text-decoration: none; }
.small { font-size: 8.4pt; }
"""


def experiment_log_table():
    d = pd.read_csv(ROOT / "outputs" / "experiments" / "log.csv")
    d = d.iloc[sorted(range(len(d)), key=lambda i: (i if not d.name[i].startswith("t17") else 1000, d.name[i]))]
    rows = ["| Trial | Change (what and why) | dev mIoU | Decision |", "|---|---|---|---|"]
    for _, r in d.iterrows():
        change = str(r.change).strip()
        why = str(r.why).strip()
        cell = f"{change}. <span class='small'><i>Why:</i> {why}</span>" if why and why != "nan" else change
        cell = cell.replace("|", "/")
        dec = str(r.decision) if str(r.decision) != "nan" else ""
        rows.append(f"| `{r['name']}` | {cell} | {r.miou:.3f} | {dec.replace('|', '/')} |")
    return "\n".join(rows)


def trial_progress_chart(path):
    d = pd.read_csv(ROOT / "outputs" / "experiments" / "log.csv")
    d["kept"] = d.decision.fillna("").str.startswith(("keep", "reference"))
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    x = range(len(d))
    best = d.miou.where(d.kept).cummax().ffill()
    ax.plot(x, best, color="#2a78d6", lw=2, label="best kept so far", zorder=2)
    ax.scatter([i for i in x if d.kept[i]], d.miou[d.kept], color="#2a78d6", s=28, zorder=3, label="kept")
    ax.scatter([i for i in x if not d.kept[i]], d.miou[~d.kept], color="#c9c8c3", s=28, zorder=3,
               label="discarded / superseded")
    for i, name in [(0, "gray+Otsu"), (5, "threshold sweep"), (10, "shape x contrast score"), (12, "GrabCut"),
                    (16, "nested upgrade"), (26, "v1 final"), (28, "v2 (after eval)")]:
        if i < len(d):
            ax.annotate(name, (i, d.miou[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5,
                        color="#52514e")
    ax.axvline(26.5, color="#52514e", lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(list(x), [n.split("_")[0] for n in d.name], rotation=90, fontsize=6.5)
    ax.set_ylabel("dev mIoU", fontsize=8, color="#52514e")
    ax.set_ylim(0.2, 0.9)
    ax.grid(True, axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=7, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("Experiment log: dev mIoU of every trial", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def find_browser():
    for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]:
        if Path(p).exists():
            return p
    raise SystemExit("Chrome/Edge/Chromium not found: open report/report.html and print it to PDF")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-link", default="(Google Drive link to be added)")
    ap.add_argument("--author", default="(author)")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    for name, src in FIGURES.items():
        shutil.copy2(ROOT / src, FIG / name)
    trial_progress_chart(FIG / "trial_progress.png")

    text = (REPORT / "report.md").read_text(encoding="utf-8")
    link = args.drive_link
    # HTML, not Markdown: the link also appears inside an HTML block, where Markdown isn't processed
    link_md = f'<a href="{html.escape(link)}">{html.escape(link)}</a>' if link.startswith("http")         else f"<b>{html.escape(link)}</b>"
    text = (text.replace("{{EXPERIMENT_LOG}}", experiment_log_table())
                .replace("{{DRIVE_LINK}}", link_md).replace("{{AUTHOR}}", html.escape(args.author)))
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    body = body.replace("<h1>", "<h1>", 1)
    page = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Egg segmentation report</title>" \
           f"<style>{CSS}</style></head><body>{body}</body></html>"
    out_html = REPORT / "report.html"
    out_html.write_text(page, encoding="utf-8")

    pdf = ROOT / "report.pdf"
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, capture_output=True, timeout=180)
    from pypdf import PdfReader
    print(f"wrote {pdf} ({len(PdfReader(pdf).pages)} pages)")


if __name__ == "__main__":
    main()

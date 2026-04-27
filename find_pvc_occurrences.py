"""
PVC shape finder for iRhythm full-disclosure ECG PDFs.

What this script does:
- Loads a full-disclosure ECG PDF where each row contains 4 minutes of waveform.
- Loads a small reference image of a target PVC-like morphology.
- Performs multi-scale template matching on each ECG row to find similar shapes.
- Converts each match position to a minute timestamp using the row start time.
- De-duplicates overlapping detections and prints date/time occurrences.

Sensitivity:
- Use `--sensitivity high|medium|low` to control detection strictness.
  - high   : strict matching (fewest false positives, may miss weak events)
  - medium : balanced default
  - low    : permissive matching (more possible events, more noise/artifacts)
- Optional `--score-threshold <float>` overrides sensitivity score threshold directly.

Examples:
- python find_pvc_occurrences.py
- python find_pvc_occurrences.py --sensitivity high
- python find_pvc_occurrences.py --sensitivity low --score-threshold 0.44
- python find_pvc_occurrences.py --sensitivity low --output results.txt
"""

import argparse
import re
from datetime import datetime, timedelta

import cv2
import fitz
import numpy as np


PDF_PATH = "full_disclosure_ecg_report.pdf"
TEMPLATE_PATH = "image.png"
YEAR = 2026
SENSITIVITY_THRESHOLDS = {
    "high": 0.55,
    "medium": 0.50,
    "low": 0.47,
}


def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 130)
    return edges


def parse_row_starts(page: fitz.Page):
    words = page.get_text("words")
    # Words: (x0, y0, x1, y1, "text", block, line, wordno)
    by_line = {}
    for w in words:
        text = w[4]
        if re.fullmatch(r"\d{2}/\d{2}", text) or re.fullmatch(r"\d{2}:\d{2}(?:am|pm)", text):
            key = (w[5], w[6])
            by_line.setdefault(key, []).append(w)

    starts = []
    for _, row_words in by_line.items():
        parts = sorted(row_words, key=lambda r: r[0])
        joined = " ".join(p[4] for p in parts)
        m = re.search(r"(\d{2}/\d{2})\s+(\d{2}:\d{2}(?:am|pm))", joined)
        if not m:
            continue
        y0 = min(p[1] for p in parts)
        starts.append((y0, m.group(1), m.group(2)))

    starts.sort(key=lambda x: x[0])
    return starts[:15]


def match_in_row(row_img: np.ndarray, template_gray: np.ndarray, score_threshold: float):
    row_edges = preprocess(row_img)
    hits = []
    for scale in np.linspace(0.35, 0.95, 13):
        t = cv2.resize(template_gray, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_CUBIC)
        template_edges = preprocess(t)
        h, w = template_edges.shape
        if row_edges.shape[0] < h or row_edges.shape[1] < w or h < 8 or w < 8:
            continue

        res = cv2.matchTemplate(row_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= score_threshold)
        for y, x in zip(ys, xs):
            score = float(res[y, x])
            hits.append((x, y, score, w, h))

    # Non-max suppression by distance.
    hits.sort(key=lambda t: t[2], reverse=True)
    kept = []
    for cand in hits:
        cx, cy = cand[0], cand[1]
        if any(abs(cx - kx) < 20 and abs(cy - ky) < 10 for kx, ky, *_ in kept):
            continue
        kept.append(cand)
    return kept


def to_datetime(mmdd: str, hm_ampm: str) -> datetime:
    return datetime.strptime(f"{YEAR}/{mmdd} {hm_ampm}", "%Y/%m/%d %I:%M%p")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find occurrences of a PVC-like waveform shape in a full-disclosure ECG PDF "
            "using multi-scale template matching."
        )
    )
    parser.add_argument(
        "--pdf",
        default=PDF_PATH,
        help=f"Path to ECG PDF (default: {PDF_PATH})",
    )
    parser.add_argument(
        "--template",
        default=TEMPLATE_PATH,
        help=f"Path to reference waveform image (default: {TEMPLATE_PATH})",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=YEAR,
        help=f"Year used to expand MM/DD labels from PDF (default: {YEAR})",
    )
    parser.add_argument(
        "--sensitivity",
        choices=["high", "medium", "low"],
        default="medium",
        help="Detection strictness. high=strict, medium=balanced, low=permissive.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Optional normalized match threshold override (0.0-1.0). "
            "Higher is stricter."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output file path. When set, results are written to this file "
            "and also printed to stdout."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global YEAR
    YEAR = args.year

    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else SENSITIVITY_THRESHOLDS[args.sensitivity]
    )
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("--score-threshold must be between 0.0 and 1.0")

    doc = fitz.open(args.pdf)
    tpl = cv2.imread(args.template, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise RuntimeError("Template image not found")

    all_events = []

    for page_idx in range(1, doc.page_count):
        page = doc[page_idx]
        starts = parse_row_starts(page)
        if len(starts) < 5:
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ECG strip area bounds for this rendering scale.
        x_left, x_right = 110, img.shape[1] - 20
        strip_h = int((starts[1][0] - starts[0][0]) * 2)
        strip_h = max(strip_h, 95)

        for y_pdf, mmdd, hm in starts:
            y_px = int(y_pdf * 2)
            y0 = max(0, y_px - 22)
            y1 = min(img.shape[0], y0 + strip_h)
            row = img[y0:y1, x_left:x_right]
            hits = match_in_row(row, tpl, score_threshold=score_threshold)
            if not hits:
                continue

            row_start = to_datetime(mmdd, hm)
            width = x_right - x_left
            for x, _, score, tw, _ in hits:
                # Convert x position to minute slot within the 4-minute row.
                center = x + tw / 2
                minute_slot = int(np.clip(np.floor((center / width) * 4), 0, 3))
                event_time = row_start + timedelta(minutes=minute_slot)
                all_events.append((event_time, score, page_idx + 1))

    # De-duplicate nearby identical timestamps (from overlapping matches).
    all_events.sort(key=lambda t: (t[0], -t[1]))
    dedup = []
    for evt in all_events:
        if dedup and evt[0] == dedup[-1][0] and (evt[2] == dedup[-1][2]):
            if evt[1] > dedup[-1][1]:
                dedup[-1] = evt
            continue
        dedup.append(evt)

    lines = []
    lines.append(
        f"Detected occurrences (sensitivity={args.sensitivity}, "
        f"score_threshold={score_threshold:.3f}):"
    )
    for dt, score, page_no in dedup:
        lines.append(f"{dt:%m/%d/%Y %I:%M %p}  score={score:.3f}  page={page_no}")
    lines.append("")
    lines.append(f"Total: {len(dedup)}")

    output_text = "\n".join(lines)
    print(output_text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text + "\n")


if __name__ == "__main__":
    main()

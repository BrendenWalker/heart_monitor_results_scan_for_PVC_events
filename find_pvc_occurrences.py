"""
PVC shape finder for iRhythm full-disclosure ECG PDFs.

What this script does:
- Loads a full-disclosure ECG PDF where each horizontal strip is one minute of
  waveform; a timestamp in the margin repeats every few minutes (~4).
- Loads a small reference image of a target PVC-like morphology.
- Performs multi-scale template matching on each strip band to find similar shapes.
- Maps each hit to clock time: strip height is inferred from consecutive labels;
  strips above the first label on a page count back one minute per strip; within
  a strip, time advances with horizontal position across that minute.
- De-duplicates overlapping multi-scale matches (same beat); keeps separate hits
  that are far enough apart for multiple PVCs on one strip.
- Prints one line per one-minute strip window: timestamp, event count, page, best score.

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
import math
import re
from collections import defaultdict
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
    row_h = float(row_edges.shape[0])
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

    # Non-max suppression: merge peaks that land on the same beat (nearby centers,
    # different scales); keep separate peaks far enough apart for multiple PVCs
    # on one strip. Separation scales with template size, not a fixed pixel cap.
    base_w = float(template_gray.shape[1])
    base_h = float(template_gray.shape[0])
    hits.sort(key=lambda t: t[2], reverse=True)
    kept = []
    for cand in hits:
        x, y, sc, tw, th = cand
        cx, cy = x + tw / 2.0, y + th / 2.0
        conflict = False
        for k in kept:
            kx, ky, kw, kh = k[0], k[1], k[3], k[4]
            kcx, kcy = kx + kw / 2.0, ky + kh / 2.0
            min_dx = max(10.0, 0.28 * max(float(tw), float(kw), base_w * 0.35))
            min_dy = max(5.0, 0.32 * max(float(th), float(kh), base_h * 0.35))
            # Thin 1-minute crops: do not require huge vertical separation (all hits
            # share ~the same strip); still merge same-beat duplicates from scales.
            min_dy = min(min_dy, max(6.0, row_h * 0.42))
            if abs(cx - kcx) < min_dx and abs(cy - kcy) < min_dy:
                conflict = True
                break
        if not conflict:
            kept.append(cand)
    return kept


def to_datetime(mmdd: str, hm_ampm: str) -> datetime:
    return datetime.strptime(f"{YEAR}/{mmdd} {hm_ampm}", "%Y/%m/%d %I:%M%p")


def _minutes_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 60.0


def strip_height_pdf_between(anchors: list, i: int) -> float:
    """PDF vertical distance per one-minute strip, using segment starting at anchor i."""
    if i + 1 < len(anchors):
        y0, t0 = anchors[i]
        y1, t1 = anchors[i + 1]
        dy = y1 - y0
        dt = abs(_minutes_between(t0, t1))
        if dt < 0.5:
            dt = 4.0
        return dy / dt
    if i > 0:
        return strip_height_pdf_between(anchors, i - 1)
    return 1.0


def event_time_for_y(
    y_pdf: float,
    frac_x: float,
    anchors: list,
) -> datetime:
    """
    Clock time for a point at PDF y (vertical center of match) and horizontal
    fraction frac_x in [0,1] across the strip (one minute left to right).
    Strips above the first printed anchor use one minute less per strip height.
    """
    frac_x = float(np.clip(frac_x, 0.0, 1.0))
    y0, t0 = anchors[0]
    if y_pdf < y0:
        sh = strip_height_pdf_between(anchors, 0)
        if sh <= 0:
            sh = 1.0
        k = math.floor((y_pdf - y0) / sh)
        return t0 + timedelta(minutes=k) + timedelta(seconds=frac_x * 60.0)

    for i in range(len(anchors) - 1):
        ya, ta = anchors[i]
        yb, tb = anchors[i + 1]
        if y_pdf < yb:
            sh = (yb - ya) / max(abs(_minutes_between(ta, tb)), 0.5)
            if abs(_minutes_between(ta, tb)) < 0.5:
                sh = (yb - ya) / 4.0
            k = math.floor((y_pdf - ya) / sh)
            return ta + timedelta(minutes=k) + timedelta(seconds=frac_x * 60.0)

    y_last, t_last = anchors[-1]
    sh = strip_height_pdf_between(anchors, len(anchors) - 2)
    if sh <= 0:
        sh = 1.0
    k = math.floor((y_pdf - y_last) / sh)
    return t_last + timedelta(minutes=k) + timedelta(seconds=frac_x * 60.0)


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
    parser.add_argument(
        "--per-hit",
        action="store_true",
        help=(
            "After the grouped strip summary, also print each hit on its own line "
            "(timestamp, score, page)."
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

    # Smallest matching scale (see linspace in match_in_row) must fit inside one strip.
    min_scale = 0.35
    min_strip_px = max(
        24,
        int(math.ceil(tpl.shape[0] * min_scale)) + 4,
        int(math.ceil(tpl.shape[1] * min_scale)) + 4,
    )

    all_events = []

    for page_idx in range(1, doc.page_count):
        page = doc[page_idx]
        starts = parse_row_starts(page)
        if len(starts) < 2:
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ECG strip area bounds for this rendering scale.
        x_left, x_right = 110, img.shape[1] - 20
        width = x_right - x_left

        anchors = [(y, to_datetime(mm, hm)) for y, mm, hm in starts]
        dt0 = abs(_minutes_between(anchors[0][1], anchors[1][1]))
        if dt0 < 0.5:
            dt0 = 4.0
        strip_h_pdf = (anchors[1][0] - anchors[0][0]) / dt0
        strip_h_px = max(int(strip_h_pdf * 2), min_strip_px)

        img_h = img.shape[0]
        y_top = 0
        while y_top < img_h:
            y1 = min(img_h, y_top + strip_h_px)
            row = img[y_top:y1, x_left:x_right]
            if row.shape[0] < 15:
                break
            hits = match_in_row(row, tpl, score_threshold=score_threshold)
            if hits:
                for x, y, score, tw, th in hits:
                    center_x = x + tw / 2
                    center_y_pix = y_top + y + th / 2
                    y_pdf = center_y_pix / 2.0
                    frac_x = float(np.clip(center_x / width, 0.0, 1.0))
                    event_time = event_time_for_y(y_pdf, frac_x, anchors)
                    all_events.append((event_time, score, page_idx + 1))
            y_top += strip_h_px

    # De-duplicate identical (time, page) from redundant peaks; sub-second times
    # still distinguish multiple PVCs on one strip.
    all_events.sort(key=lambda t: (t[0], -t[1]))
    dedup = []
    for evt in all_events:
        if dedup and evt[0] == dedup[-1][0] and (evt[2] == dedup[-1][2]):
            if evt[1] > dedup[-1][1]:
                dedup[-1] = evt
            continue
        dedup.append(evt)

    # Group by calendar minute + page (one ECG strip ≈ one minute of trace).
    by_strip = defaultdict(list)
    for dt, score, page_no in dedup:
        minute_key = dt.replace(second=0, microsecond=0)
        by_strip[(page_no, minute_key)].append((dt, score))

    strip_keys = sorted(by_strip.keys(), key=lambda k: (k[1], k[0]))

    lines = []
    lines.append(
        f"Detected occurrences (sensitivity={args.sensitivity}, "
        f"score_threshold={score_threshold:.3f}):"
    )
    lines.append("timestamp\tevents\tmax_score\tpage")
    for page_no, bucket in strip_keys:
        hits_here = by_strip[(page_no, bucket)]
        n = len(hits_here)
        best = max(s for _, s in hits_here)
        lines.append(f"{bucket:%m/%d/%Y %I:%M %p}\t{n}\t{best:.3f}\t{page_no}")
    if args.per_hit:
        lines.append("")
        lines.append("per_hit_timestamp\tper_hit_score\tper_hit_page")
        for dt, score, page_no in dedup:
            lines.append(f"{dt:%m/%d/%Y %I:%M:%S %p}\t{score:.3f}\t{page_no}")
    lines.append("")
    lines.append(f"Strip windows with at least one hit: {len(by_strip)}")
    lines.append(f"Total distinct hits (after de-dup): {len(dedup)}")

    output_text = "\n".join(lines)
    print(output_text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text + "\n")


if __name__ == "__main__":
    main()

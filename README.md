# PVC Waveform Finder

This project scans a full-disclosure ECG PDF and finds occurrences of a target PVC-like waveform shape using image-based template matching.

## What the script does

- Reads a full-disclosure ECG PDF where each strip row has a printed start `MM/DD HH:MMam/pm` and spans 4 minutes.
- Reads a reference waveform image (for example, a cropped PVC shape).
- Runs multi-scale edge-based template matching across each ECG row.
- Maps horizontal match position to one of the four minute slots in that row.
- De-duplicates overlapping hits and outputs date/time detections with score and PDF page.

## File

- Main script: `find_pvc_occurrences.py`

## Requirements

Install dependencies:

```bash
python -m pip install pymupdf opencv-python numpy scipy pillow
```

## Usage

Basic run:

```bash
python find_pvc_occurrences.py
```

Set sensitivity:

```bash
python find_pvc_occurrences.py --sensitivity high
python find_pvc_occurrences.py --sensitivity medium
python find_pvc_occurrences.py --sensitivity low
```

Override score threshold directly:

```bash
python find_pvc_occurrences.py --score-threshold 0.52
```

Write output to file:

```bash
python find_pvc_occurrences.py --sensitivity low --output results.txt
```

Use custom inputs:

```bash
python find_pvc_occurrences.py --pdf ecg_report.pdf --template template_waveform.png --year 2026
```

## Sensitivity behavior

- `high` (`0.55`): strict matching; fewer false positives, may miss weaker events.
- `medium` (`0.50`): balanced default.
- `low` (`0.47`): permissive matching; catches more possible events, but includes more artifact/noise matches.

If `--score-threshold` is supplied, it overrides the sensitivity preset.

## Output format

The script prints:

- a header with active sensitivity and threshold,
- one line per detection:
  - `MM/DD/YYYY HH:MM AM/PM  score=<value>  page=<number>`
- a total count.

If `--output` is set, the same text is written to the output file.

## Notes and limitations

- This is similarity matching, not a clinical diagnosis tool.
- Recorded artifacts, baseline wander, and lead morphology differences can produce false positives/negatives.
- Best practice is to use this as a triage list, then visually review detections in the original PDF.
- The script assumes rows are standard 4-minute strips with left-side timestamps from this report format.
- Do not commit source ECG reports, patient-derived waveform images, or raw detection outputs to public repositories.

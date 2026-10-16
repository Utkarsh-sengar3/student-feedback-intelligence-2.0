# Phase 11 — Aspect / Category Intelligence

This phase establishes the real structure of the EduRABSA aspect annotations before additional modeling.

It does NOT retrain the sentiment model.

Run from the project root:

```powershell
..\.venv\Scripts\python.exe -u scripts\phase11_aspect_category_intelligence.py
```

Input:

`data/processed/edurabsa/annotations_canonical.csv`

Outputs:

```text
data/processed/edurabsa/phase11/
├── phase11_summary.json
├── category_summary.csv
├── category_sentiment_matrix.csv
├── aspect_summary.csv
├── domain_category_summary.csv
├── opinion_summary.csv
└── PHASE_11_README.md
```

The next modeling step should use the results to decide:

1. category classification strategy
2. aspect extraction strategy
3. aspect-sentiment integration
4. sentiment × category analytics

No temporal/emerging-trend claims are made unless a reliable time field is established.

# Phase 5 — Concern, Severity & Priority Intelligence

EduRABSA provides aspect, opinion, category and sentiment annotations, but no ground-truth severity/priority labels. Therefore this phase creates transparent derived heuristics.

Severity:
- High = negative + explicit high-severity language signal
- Medium = negative without high-severity signal
- Low = neutral/positive

Priority combines negative rate, frequency, and severity adjustment. These scores are derived intelligence and must not be presented as supervised labels.

Files:
- annotation_severity.csv
- category_priority.csv
- aspect_priority.csv
- review_priority.csv
- phase5_summary.json

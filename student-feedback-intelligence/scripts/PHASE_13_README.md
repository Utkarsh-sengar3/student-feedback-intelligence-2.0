# Phase 13 — Sentiment × Aspect Intelligence

Run from the project root:

```powershell
..\.venv\Scripts\python.exe -u scripts/phase13_sentiment_aspect_intelligence.py
```

This phase builds the analytical layer from the canonical EduRABSA
annotations. It does not retrain DistilBERT.

It connects:

- aspect
- opinion
- category
- aspect-level sentiment
- overall review sentiment
- domain
- evidence text

The main outputs are written to:

`data/processed/edurabsa/phase13/`

No temporal/emerging-trend claim is made because the current canonical
dataset does not establish a reliable time axis.

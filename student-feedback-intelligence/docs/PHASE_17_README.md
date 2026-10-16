# Phase 17 — Semantic Search — FIXED

The original Phase 17 successfully generated embeddings but the demo searches
returned empty results because sentiment filters were too strict.

This version normalizes:
- Positive / Pos
- Negative / Neg
- Neutral / Neu

It also performs an unfiltered `course workload` sanity query before the demo
queries.

IMPORTANT: Existing `review_embeddings.npy` can be reused. The script
currently rebuilds embeddings when run; this is safe but not necessary if you
later optimize it to load existing embeddings.

Run:

```powershell
..\.venv\Scripts\python.exe -u scripts/phase17_semantic_search.py
```

Expected result: non-empty sanity/demo retrieval results.

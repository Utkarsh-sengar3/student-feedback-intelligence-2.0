# Phase 18 — Retrieval Evaluation

This phase evaluates the Phase 17 semantic search index using a reproducible weakly-supervised benchmark.

Metrics: Precision@K, Recall@K, MRR, NDCG@K.

The benchmark derives relevance from available EduRABSA metadata and intent terms. It is **not human-judged relevance**. Use it for regression testing and comparing retrieval versions, not as a claim of human retrieval quality.

## Run

From the project root:

```powershell
..\.venv\Scripts\python.exe -u scripts\phase18_retrieval_evaluation.py
```

Expected output directory:

```text
data\processed\edurabsa\phase18\
```

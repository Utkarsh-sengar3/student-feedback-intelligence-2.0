# Phase 19 — RAG / Evidence-Grounded Intelligence

## Objective

Build an evidence-grounded question-answering layer over the Phase 17
semantic retrieval index.

---

## Pipeline

Question

↓

Query Embedding

↓

Semantic Retrieval

↓

Evidence Context

↓

Grounded Answer

---

## Phase 17 dependency

Phase 19 uses the existing Phase 17 artifacts:

```text
data/
└── processed/
    └── edurabsa/
        └── phase17/
            ├── review_embeddings.npy
            └── review_metadata.csv
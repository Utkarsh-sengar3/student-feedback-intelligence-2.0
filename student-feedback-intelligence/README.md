# AI-Based Student Feedback Sentiment Intelligence

Professional end-to-end AI/ML project for transforming student feedback into
explainable institutional intelligence.

## Current status

**Architecture initialized — Phase 1 pending.**

EduRABSA is intentionally NOT included in this repository/skeleton.
Place the original dataset under:

`data/raw/edurabsa/`

Do not modify the raw dataset.

## Development principle

1. Inspect the real dataset first.
2. Validate its schema and annotations.
3. Build reproducible experiments in `notebooks/`.
4. Move verified reusable logic into `src/`.
5. Build API and frontend only after the intelligence layer is verified.

## Main layers

- `data/` — raw and processed datasets
- `notebooks/` — experimentation and analysis
- `src/` — production ML/data logic
- `models/` — trained model artifacts
- `api/` — FastAPI backend
- `frontend/` — Next.js/React application
- `tests/` — automated tests
- `docs/` — technical documentation
- `config/` — configuration
- `scripts/` — operational utilities

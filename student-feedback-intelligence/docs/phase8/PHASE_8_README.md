# Phase 8 — Backend API

## Endpoints

GET `/health`
GET `/api/v1/status`
POST `/api/v1/analyze`

Example request:

```json
{
  "text": "The course workload is too heavy and the assignments are difficult to manage."
}
```

## Expected project locations

```text
models/
├── sentiment/
│   └── baseline_sentiment_model.joblib
└── aspects/
    └── category_classifier.joblib

data/
└── processed/
    └── edurabsa/
        ├── phase5/
        │   └── category_priority.csv
        └── phase7/
            └── recommendations.csv
```

## Run

From the project root:

```bash
pip install -r api/requirements-api.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

`http://localhost:8000/docs`

## Important limitation

The Phase 4 aspect component was a lexical baseline rather than a production span-extraction model. Therefore the current API returns `aspect: null`. The aspect extraction model should be upgraded before final deployment.

Severity and priority are derived heuristics from Phase 5, and recommendations come from the Phase 7 evidence-grounded recommendation layer.

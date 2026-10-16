# API Architecture

Frontend
  |
  v
FastAPI
  |
  +--> Sentiment model (Phase 3)
  |
  +--> Category model (Phase 4)
  |
  +--> Severity heuristic (Phase 5)
  |
  +--> Priority data (Phase 5)
  |
  +--> Recommendation data (Phase 7)
  |
  v
JSON response

The API is intentionally thin. Training remains outside the API process.

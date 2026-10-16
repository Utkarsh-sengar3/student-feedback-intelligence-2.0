# Student Feedback Intelligence — Enhanced Frontend

This ZIP contains the website frontend. It does not replace your existing Python API, datasets, trained models, embeddings, or tests.

## Features included
- Premium SaaS layout with collapsible sidebar, dark/light mode, responsive layout and keyboard command palette (`Ctrl+K`).
- Overview dashboard with API-backed review counts, sentiment breakdown from loaded review records, insight cards and navigation actions.
- Feedback analysis with results, categories, recommendations, supporting evidence and a second-review comparison workflow.
- Feedback explorer with search/filter/pagination integration and CSV export of the loaded page.
- Insights & Concerns view.
- Semantic Evidence Explorer.
- RAG assistant interface with supporting evidence and limitations.
- Model Performance view that displays data returned by the backend.
- Browser-based CSV batch analysis (up to 100 rows/run) using the existing `/api/v1/analyze` endpoint, progress and CSV results export.
- Reports & Actions: exports for currently loaded records, saved filter presets, and local browser-based action tracking.
- Settings/About page.

## Important integration note
The frontend expects a FastAPI backend at `http://127.0.0.1:8000` by default. Endpoints used include:
- `GET /health`
- `POST /api/v1/analyze`
- `GET /api/v1/reviews/`
- `GET /api/v1/insights/top?limit=10`
- `GET /api/v1/insights/summary`
- `GET /api/v1/search/?query=...&top_k=10`
- `POST /api/v1/assistant/`
- `GET /api/v1/model-metrics/`

If your actual API uses a different path, inspect `http://127.0.0.1:8000/docs` and update the path in `src/App.jsx`. The UI shows errors when an endpoint is unavailable rather than fabricating live results. Some endpoints may use a different trailing slash or response shape in your current backend and may need adjustment.

Batch analysis sends one request per non-empty row and is intentionally capped at 100 rows per run. For large production jobs, add backend job queues, persistent progress and cancellation.

Action tracking and saved filter presets are stored in the current browser's localStorage. They are not shared between users and are not persisted in your backend database. This package does not claim multi-user or database-backed action management.

The trend area on the overview is explicitly marked as a UI preview and must not be reported as real project trend data.

## Installation
Copy the contents of this archive's `student_feedback_website` folder into your existing project's `frontend` folder. Preserve the existing `api`, `data`, `models`, `src`, `tests`, and virtual environment.

### Start backend
Open PowerShell in the project root (the folder containing `api`):
```powershell
..\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --reload
```
If `.venv` is inside the project root:
```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --reload
```

### Start frontend
Open a second PowerShell:
```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```
Open `http://localhost:3000`.

Set `VITE_API_BASE_URL` in `frontend/.env` if the backend runs at a different address. Restart the Vite server after changing it.

### Production build
```powershell
npm run build
```

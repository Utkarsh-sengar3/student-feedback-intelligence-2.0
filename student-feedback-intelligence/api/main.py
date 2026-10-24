import os

# Limit thread pools to minimize memory footprint on constrained environments (Render 512MB RAM)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas.feedback import (
    FeedbackRequest,
    FeedbackResponse,
)
from api.services.feedback_service import FeedbackService
from api.routes.search import router as search_router
from api.routes.insights import router as insights_router
from api.routes.assistant import router as assistant_router
from api.routes.reviews import router as reviews_router
from api.routes.metrics import router as metrics_router

app = FastAPI(
    title="AI-Based Student Feedback Intelligence API",
    description=(
        "Production AI/ML backend for student feedback "
        "sentiment intelligence, multilabel category classification, "
        "aspect extraction, actionable recommendations, semantic search, "
        "and evidence-grounded RAG assistance."
    ),
    version="2.1.0",
)

# ============================================================
# CORS
# ============================================================

cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]
else:
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://student-feedback-intelligence-2-0.vercel.app",
        "https://student-feedback-intelligence.vercel.app",
    ]

# Always ensure production Vercel frontend domains are included in CORS origins
for prod_origin in [
    "https://student-feedback-intelligence-2-0.vercel.app",
    "https://student-feedback-intelligence.vercel.app",
]:
    if prod_origin not in origins:
        origins.append(prod_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Avoid leaking internal stack traces in production responses
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An internal server error occurred.",
            "detail": str(exc),
        },
    )


# ============================================================
# SERVICES (Loaded once in memory at startup)
# ============================================================

feedback_service = FeedbackService()


@app.on_event("startup")
def startup_warmup():
    import threading
    threading.Thread(target=feedback_service.warmup, daemon=True).start()


# ============================================================
# HEALTH & STATUS
# ============================================================


@app.get(
    "/health",
    tags=["System"],
)
def health():
    return {
        "status": "ok",
        "service": "student-feedback-intelligence-api",
        "version": "2.1.0",
    }


@app.get(
    "/api/v1/status",
    tags=["System"],
)
def status():
    return {
        "status": "ready",
        "service": "student-feedback-intelligence-api",
        "models": feedback_service.status(),
    }


# ============================================================
# FEEDBACK ANALYSIS
# ============================================================


@app.post(
    "/api/v1/analyze",
    response_model=FeedbackResponse,
    tags=["Feedback"],
)
def analyze_feedback(
    payload: FeedbackRequest,
):
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Field 'text' must be a non-empty string.",
        )

    try:
        result = feedback_service.analyze(text)
        return FeedbackResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline error: {str(exc)}",
        )


# ============================================================
# SUB-ROUTERS
# ============================================================

app.include_router(search_router)
app.include_router(insights_router)
app.include_router(assistant_router)
app.include_router(reviews_router)
app.include_router(metrics_router)

# ============================================================
# ROOT
# ============================================================


@app.get(
    "/",
    tags=["System"],
)
def root():
    return {
        "name": "AI-Based Student Feedback Sentiment Intelligence",
        "version": "2.1.0",
        "status": "running",
        "author": "Utkarsh Sengar",
        "documentation": "/docs",
        "endpoints": [
            "/api/v1/status",
            "/api/v1/analyze",
            "/api/v1/search",
            "/api/v1/insights",
            "/api/v1/assistant",
            "/api/v1/reviews",
            "/api/v1/model-metrics",
        ],
    }

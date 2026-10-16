from fastapi import APIRouter, HTTPException

from api.services.insight_service import InsightService

router = APIRouter(
    prefix="/api/v1/insights",
    tags=["Insights"],
)

_service = None


def get_insight_service() -> InsightService:
    global _service

    if _service is None:
        _service = InsightService()

    return _service


@router.get("/")
def get_insights():
    try:
        service = get_insight_service()
        data = service.get_all_insights()

        return {
            "success": True,
            "count": len(data),
            "data": data,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/top")
def get_top_insights(limit: int = 10):
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100",
        )

    try:
        service = get_insight_service()

        return {
            "success": True,
            "data": service.get_top_insights(limit),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/summary")
def get_summary():
    try:
        service = get_insight_service()

        return {
            "success": True,
            "data": service.get_summary(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{concept}")
def get_insight(concept: str):
    try:
        service = get_insight_service()
        result = service.get_insight(concept)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Insight '{concept}' not found.",
            )

        return {
            "success": True,
            "data": result,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
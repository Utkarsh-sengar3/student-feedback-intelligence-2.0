from fastapi import APIRouter, HTTPException

from api.services.insight_service import (
    InsightService
)


router = APIRouter(
    prefix="/api/v1/insights",
    tags=["Insights"]
)


service = InsightService()


@router.get("/")
def get_insights():

    return {
        "success": True,
        "count": len(
            service.get_all_insights()
        ),
        "data": service.get_all_insights()
    }


@router.get("/top")
def get_top_insights(
    limit: int = 10
):

    if limit < 1 or limit > 100:

        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100"
        )

    return {
        "success": True,
        "data": service.get_top_insights(
            limit
        )
    }


@router.get("/summary")
def get_summary():

    return {
        "success": True,
        "data": service.get_summary()
    }


@router.get("/{concept}")
def get_insight(
    concept: str
):

    result = service.get_insight(
        concept
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=f"Insight '{concept}' not found."
        )

    return {
        "success": True,
        "data": result
    }
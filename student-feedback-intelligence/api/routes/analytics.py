from typing import Optional
from fastapi import APIRouter, Query, Response
from fastapi.responses import Response

from api.schemas.analytics import (
    AnalyticsOverviewResponse,
    AnalyticsRecordsResponse,
)
from api.services.analytics_service import AnalyticsService

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["EDA & Analytics"],
)

service = AnalyticsService.get_instance()


@router.get("/overview", response_model=AnalyticsOverviewResponse)
def get_analytics_overview(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment (positive, negative, neutral)"),
    category: Optional[str] = Query(None, description="Filter by aspect category"),
    search: Optional[str] = Query(None, description="Search keyword in review text"),
):
    """
    Returns high-level KPI metrics, sentiment distributions, domain breakdowns,
    category volumes, text length and term distributions, and filter options.
    """
    return service.get_overview(
        domain=domain,
        sentiment=sentiment,
        category=category,
        search_query=search,
    )


@router.get("/records", response_model=AnalyticsRecordsResponse)
def get_analytics_records(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
    category: Optional[str] = Query(None, description="Filter by aspect category"),
    search: Optional[str] = Query(None, description="Search keyword in review text"),
):
    """
    Returns paginated drill-down records with associated aspect categories,
    sentiment, length metrics, and review text matching active filters.
    """
    return service.get_records(
        page=page,
        page_size=page_size,
        domain=domain,
        sentiment=sentiment,
        category=category,
        search_query=search,
    )


@router.get("/export")
def export_analytics_records_csv(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
    category: Optional[str] = Query(None, description="Filter by aspect category"),
    search: Optional[str] = Query(None, description="Search keyword in review text"),
):
    """
    Generates and streams a CSV export of feedback records matching active filters.
    """
    csv_content = service.export_csv(
        domain=domain,
        sentiment=sentiment,
        category=category,
        search_query=search,
    )
    filename = "feedback_records_export.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

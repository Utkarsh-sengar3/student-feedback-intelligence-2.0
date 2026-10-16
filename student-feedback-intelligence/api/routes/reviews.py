from typing import Optional
from fastapi import APIRouter, Query
from api.schemas.feedback import ReviewsListResponse
from api.services.reviews_service import ReviewsService

router = APIRouter(
    prefix="/api/v1/reviews",
    tags=["Reviews Explorer"],
)

service = ReviewsService.get_instance()


@router.get("/", response_model=ReviewsListResponse)
def get_reviews(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
    search: Optional[str] = Query(None, description="Search keyword in review text"),
):
    result = service.get_reviews(
        page=page,
        page_size=page_size,
        domain=domain,
        sentiment=sentiment,
        search_query=search,
    )
    return result

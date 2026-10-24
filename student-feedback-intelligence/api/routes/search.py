import threading
from fastapi import APIRouter, HTTPException, Query

from api.services.search_service import SearchService


router = APIRouter(
    prefix="/api/v1/search",
    tags=["Semantic Search"],
)

_service = None
_service_lock = threading.Lock()


def get_search_service() -> SearchService:
    """Initialize the search service only when a search endpoint is used."""
    global _service

    if _service is None:
        with _service_lock:
            if _service is None:
                _service = SearchService()

    return _service


@router.get("/")
def search_feedback(
    query: str = Query(..., min_length=2, max_length=500),
    top_k: int = Query(10, ge=1, le=50),
    domain: str | None = None,
    sentiment: str | None = None,
):
    query = query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        service = get_search_service()

        results = service.search(
            query=query,
            top_k=top_k,
            domain=domain,
            sentiment=sentiment,
        )

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "filters": {
                "domain": domain,
                "sentiment": sentiment,
            },
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Semantic search is currently unavailable: {exc}",
        ) from exc


@router.get("/stats")
def search_stats():
    try:
        service = get_search_service()

        return {
            "success": True,
            "data": service.stats(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Search statistics are currently unavailable: {exc}",
        ) from exc
from fastapi import APIRouter, HTTPException, Query

from api.services.search_service import SearchService


router = APIRouter(
    prefix="/api/v1/search",
    tags=["Semantic Search"],
)


service = SearchService()


@router.get("/")
def search_feedback(
    query: str = Query(
        ...,
        min_length=2,
        max_length=500,
    ),
    top_k: int = Query(
        10,
        ge=1,
        le=50,
    ),
    domain: str | None = None,
    sentiment: str | None = None,
):

    query = query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

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


@router.get("/stats")
def search_stats():

    return {
        "success": True,
        "data": service.stats(),
    }
from typing import List, Optional

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    review_id: str
    domain: str
    sentiment: str
    similarity_score: Optional[float] = None
    text: str


class InsightResponse(BaseModel):
    concept: str
    priority: str
    severity: str
    priority_score: float
    negative_rate: float
    affected_reviews: int
    recommended_owner: str
    recommendation: str
    evidence: List[EvidenceItem]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8


class SearchResult(BaseModel):
    review_id: str
    similarity_score: float
    domain: str
    sentiment: str
    text: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
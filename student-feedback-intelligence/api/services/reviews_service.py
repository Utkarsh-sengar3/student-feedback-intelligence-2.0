from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase17"
    / "review_metadata.csv"
)


class ReviewsService:
    _instance = None

    def __init__(self):
        if not METADATA_PATH.exists():
            self.df = pd.DataFrame(
                columns=[
                    "row_index",
                    "review_id",
                    "original_id",
                    "text",
                    "domain",
                    "overall_sentiment",
                ]
            )
        else:
            self.df = pd.read_csv(METADATA_PATH)
            self.df["row_index"] = self.df.get(
                "row_index", pd.Series(range(len(self.df)))
            ).astype(int)
            self.df["text"] = self.df["text"].fillna("").astype(str)
            self.df["domain"] = self.df["domain"].fillna("Unknown").astype(str)
            self.df["overall_sentiment"] = (
                self.df["overall_sentiment"].fillna("Unknown").astype(str)
            )

    @classmethod
    def get_instance(cls) -> "ReviewsService":
        if cls._instance is None:
            cls._instance = ReviewsService()
        return cls._instance

    def get_reviews(
        self,
        page: int = 1,
        page_size: int = 20,
        domain: Optional[str] = None,
        sentiment: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self.df.copy()

        if domain and domain.lower() != "all":
            df = df[df["domain"].str.lower() == domain.strip().lower()]

        if sentiment and sentiment.lower() != "all":
            df = df[df["overall_sentiment"].str.lower() == sentiment.strip().lower()]

        if search_query and search_query.strip():
            query_lower = search_query.strip().lower()
            df = df[df["text"].str.lower().str.contains(query_lower, regex=False)]

        total = len(df)
        total_pages = max(1, int(np.ceil(total / page_size))) if page_size > 0 else 1
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        sliced = df.iloc[start_idx:end_idx]

        records = sliced.to_dict(orient="records")

        # Global or filtered distribution counts
        domain_counts = (
            self.df["domain"].value_counts().to_dict()
            if not self.df.empty
            else {}
        )
        sentiment_counts = (
            self.df["overall_sentiment"].value_counts().to_dict()
            if not self.df.empty
            else {}
        )

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "domain_counts": {str(k): int(v) for k, v in domain_counts.items()},
            "sentiment_counts": {
                str(k): int(v) for k, v in sentiment_counts.items()
            },
            "reviews": records,
        }

"""
Production-ready semantic retrieval component.

Used by:
- Phase 19 RAG
- FastAPI
- Dashboard
- AI Assistant

Retrieval pipeline:

Query
  ↓
SentenceTransformer embedding
  ↓
Cosine similarity
  ↓
Optional domain filter
  ↓
Optional sentiment filter
  ↓
Optional similarity threshold
  ↓
Top-K results
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class SemanticRetriever:

    def __init__(
        self,
        embeddings_path,
        metadata_path,
        model_name="all-MiniLM-L6-v2",
    ):
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: "
                f"{self.embeddings_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_path}"
            )

        print(
            f"Loading retrieval embeddings: "
            f"{self.embeddings_path}"
        )

        self.embeddings = np.load(
            self.embeddings_path
        ).astype("float32")

        print(
            f"Loading retrieval metadata: "
            f"{self.metadata_path}"
        )

        self.metadata = pd.read_csv(
            self.metadata_path
        )

        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                "Embeddings and metadata counts differ: "
                f"{len(self.embeddings)} vs "
                f"{len(self.metadata)}"
            )

        required_columns = {
            "review_id",
            "text",
        }

        missing = (
            required_columns
            - set(self.metadata.columns)
        )

        if missing:
            raise ValueError(
                "Metadata is missing required columns: "
                f"{sorted(missing)}"
            )

        if "domain" not in self.metadata.columns:
            self.metadata["domain"] = "Unknown"

        if "overall_sentiment" not in self.metadata.columns:
            self.metadata["overall_sentiment"] = "Unknown"
        else:
            self.metadata["overall_sentiment"] = (
            self.metadata["overall_sentiment"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )
            

        MODEL_NAME = "all-MiniLM-L6-v2"

        self.model = SentenceTransformer(
            MODEL_NAME,
            local_files_only=True,
        )

        print(
            f"Retriever ready: "
            f"{len(self.metadata):,} reviews"
        )

    @staticmethod
    def _normalize_sentiment(value):
        """
        Normalize common sentiment labels.
        """

        value = str(value).strip().lower()

        mapping = {
            "positive": "positive",
            "pos": "positive",
            "negative": "negative",
            "neg": "negative",
            "neutral": "neutral",
            "neu": "neutral",
        }

        return mapping.get(value, value)

    def search(
        self,
        query,
        top_k=8,
        domain=None,
        sentiment=None,
        min_similarity=None,
    ):
        """
        Perform semantic retrieval.

        Parameters
        ----------
        query : str
            User search query.

        top_k : int
            Maximum number of results.

        domain : str, optional
            Filter by review domain.

        sentiment : str, optional
            Filter by sentiment.

        min_similarity : float, optional
            Minimum cosine similarity.

        Returns
        -------
        pandas.DataFrame
            Ranked retrieval results.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if top_k > 100:
            raise ValueError(
                "top_k cannot exceed 100."
            )

        if min_similarity is not None:

            if not -1.0 <= min_similarity <= 1.0:
                raise ValueError(
                    "min_similarity must be between "
                    "-1.0 and 1.0."
                )

        # --------------------------------------------------
        # Query embedding
        # --------------------------------------------------

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype("float32")

        # --------------------------------------------------
        # Cosine similarity
        # --------------------------------------------------

        scores = (
            self.embeddings
            @ query_vector
        )

        # --------------------------------------------------
        # Metadata filtering
        # --------------------------------------------------

        mask = np.ones(
            len(self.metadata),
            dtype=bool,
        )

        if domain:

            mask &= (
                self.metadata["domain"]
                .astype(str)
                .str.lower()
                .eq(str(domain).strip().lower())
                .to_numpy()
            )

        if sentiment:

            requested_sentiment = (
                self._normalize_sentiment(
                    sentiment
                )
            )

            mask &= (
                self.metadata[
                    "overall_sentiment"
                ]
                .map(self._normalize_sentiment)
                .eq(requested_sentiment)
                .to_numpy()
            )

        # --------------------------------------------------
        # Similarity threshold
        # --------------------------------------------------

        if min_similarity is not None:

            mask &= (
                scores >= min_similarity
            )

        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) == 0:

            return pd.DataFrame(
                columns=[
                    *self.metadata.columns,
                    "similarity_score",
                ]
            )

        candidate_scores = (
            scores[candidate_indices]
        )

        k = min(
            top_k,
            len(candidate_indices),
        )

        # Efficient top-k selection
        top_positions = np.argpartition(
            -candidate_scores,
            k - 1,
        )[:k]

        top_positions = top_positions[
            np.argsort(
                -candidate_scores[
                    top_positions
                ]
            )
        ]

        indices = candidate_indices[
            top_positions
        ]

        # --------------------------------------------------
        # Build result
        # --------------------------------------------------

        results = (
            self.metadata
            .iloc[indices]
            .copy()
            .reset_index(drop=True)
        )

        results.insert(
            1,
            "similarity_score",
            scores[indices],
        )

        return results

    def search_negative(
        self,
        query,
        top_k=8,
        domain=None,
        min_similarity=None,
    ):
        """
        Convenience method for negative feedback retrieval.
        """

        return self.search(
            query=query,
            top_k=top_k,
            domain=domain,
            sentiment="Negative",
            min_similarity=min_similarity,
        )

    def search_positive(
        self,
        query,
        top_k=8,
        domain=None,
        min_similarity=None,
    ):
        """
        Convenience method for positive feedback retrieval.
        """

        return self.search(
            query=query,
            top_k=top_k,
            domain=domain,
            sentiment="Positive",
            min_similarity=min_similarity,
        )

    def stats(self):
        """
        Return basic index statistics.
        """

        return {
            "reviews_indexed": int(
                len(self.metadata)
            ),
            "embedding_dimension": int(
                self.embeddings.shape[1]
            ),
            "model": self.model_name,
            "domains": sorted(
                self.metadata[
                    "domain"
                ]
                .astype(str)
                .unique()
                .tolist()
            ),
            "sentiments": sorted(
                self.metadata[
                    "overall_sentiment"
                ]
                .astype(str)
                .unique()
                .tolist()
            ),
        }


_SHARED_RETRIEVER = None


def get_shared_retriever(
    embeddings_path=None,
    metadata_path=None,
    model_name="all-MiniLM-L6-v2",
) -> SemanticRetriever:
    """
    Return a shared singleton SemanticRetriever instance.
    Prevents duplicate loading of SentenceTransformer and review embeddings in memory.
    """
    global _SHARED_RETRIEVER
    if _SHARED_RETRIEVER is None:
        if embeddings_path is None or metadata_path is None:
            root = Path(__file__).resolve().parents[2]
            phase17 = root / "data" / "processed" / "edurabsa" / "phase17"
            embeddings_path = phase17 / "review_embeddings.npy"
            metadata_path = phase17 / "review_metadata.csv"
        _SHARED_RETRIEVER = SemanticRetriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            model_name=model_name,
        )
    return _SHARED_RETRIEVER
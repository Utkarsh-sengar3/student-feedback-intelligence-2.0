"""
Semantic retriever using sentence-transformers and precomputed numpy embeddings.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


class SemanticRetriever:
    """
    Loads precomputed review embeddings and supports cosine-similarity search.
    """

    def __init__(
        self,
        embeddings_path: Union[str, Path],
        metadata_path: Union[str, Path],
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self._embeddings_path = Path(embeddings_path)
        self._metadata_path = Path(metadata_path)
        self._model_name = model_name
        self._model = None
        self._model_lock = threading.Lock()

        if not self._embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {self._embeddings_path}. "
                "Run the Phase 17 embedding script to generate it."
            )
        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self._metadata_path}."
            )

        self._embeddings: np.ndarray = np.load(
            str(self._embeddings_path), allow_pickle=False
        ).astype(np.float32)

        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = self._embeddings / norms

        self.metadata: pd.DataFrame = pd.read_csv(str(self._metadata_path))
        # Ensure row_index column
        if "row_index" not in self.metadata.columns:
            self.metadata["row_index"] = range(len(self.metadata))
        self.metadata["row_index"] = self.metadata["row_index"].astype(int)
        self.metadata["text"] = self.metadata.get("text", pd.Series("", index=self.metadata.index)).fillna("").astype(str)
        self.metadata["domain"] = self.metadata.get("domain", pd.Series("unknown", index=self.metadata.index)).fillna("unknown").astype(str)
        self.metadata["overall_sentiment"] = self.metadata.get("overall_sentiment", pd.Series("unknown", index=self.metadata.index)).fillna("unknown").astype(str)

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0].astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str] = None,
        sentiment: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Cosine-similarity search over precomputed embeddings.

        Returns a DataFrame with columns:
            row_index, similarity_score, text, domain, overall_sentiment
        """
        query_vec = self._encode_query(query)

        # Cosine similarity via dot product (embeddings are L2-normalized)
        scores: np.ndarray = self._embeddings @ query_vec

        # Build candidate DataFrame
        n = len(scores)
        df = pd.DataFrame(
            {
                "row_index": np.arange(n),
                "similarity_score": scores,
            }
        )

        # Merge with metadata for filtering
        meta = self.metadata[["row_index", "text", "domain", "overall_sentiment"]].copy()
        df = df.merge(meta, on="row_index", how="left")

        # Apply filters
        if domain and domain.lower() not in ("all", "none", ""):
            df = df[df["domain"].str.lower() == domain.strip().lower()]

        if sentiment and sentiment.lower() not in ("all", "none", ""):
            df = df[df["overall_sentiment"].str.lower() == sentiment.strip().lower()]

        # Sort by similarity descending
        df = df.sort_values("similarity_score", ascending=False)

        return df.head(top_k).reset_index(drop=True)

    def stats(self) -> dict:
        """Return statistics about the index."""
        n_embeddings = len(self._embeddings)
        embedding_dim = self._embeddings.shape[1] if len(self._embeddings.shape) > 1 else 0

        domains = list(self.metadata["domain"].dropna().unique())
        sentiments = list(self.metadata["overall_sentiment"].dropna().unique())

        return {
            "total_reviews": n_embeddings,
            "embedding_dimension": int(embedding_dim),
            "model": self._model_name,
            "domains": domains,
            "sentiments": sentiments,
        }


# Module-level singleton
_retriever: Optional[SemanticRetriever] = None
_retriever_lock = threading.Lock()
_retriever_key: Optional[tuple] = None


def get_shared_retriever(
    embeddings_path: Union[str, Path],
    metadata_path: Union[str, Path],
    model_name: str = "all-MiniLM-L6-v2",
) -> SemanticRetriever:
    """
    Return a module-level singleton SemanticRetriever.
    Creates the retriever on first call; subsequent calls with the same
    arguments return the cached instance.
    """
    global _retriever, _retriever_key

    key = (str(embeddings_path), str(metadata_path), model_name)

    if _retriever is None or _retriever_key != key:
        with _retriever_lock:
            if _retriever is None or _retriever_key != key:
                _retriever = SemanticRetriever(
                    embeddings_path=embeddings_path,
                    metadata_path=metadata_path,
                    model_name=model_name,
                )
                _retriever_key = key

    return _retriever

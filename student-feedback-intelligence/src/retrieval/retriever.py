"""
Semantic retriever using lightweight ONNX runtime / sentence-transformers and precomputed numpy embeddings.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


class OnnxQueryEncoder:
    """
    Ultra-lightweight query encoder using ONNX Runtime and HF Tokenizers.
    Consumes ~35-40 MB of RAM compared to >500 MB for PyTorch SentenceTransformer,
    allowing reliable execution on constrained environments like Render Free Tier (512MB RAM).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not model_name.startswith("sentence-transformers/") and "/" not in model_name:
            self.model_name = f"sentence-transformers/{model_name}"
        else:
            self.model_name = model_name
        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._session is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._session is not None and self._tokenizer is not None:
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer

            root = Path(__file__).resolve().parents[2]
            local_model = root / "models" / "embeddings" / "model.onnx"
            local_tok = root / "models" / "embeddings" / "tokenizer.json"

            if local_model.exists():
                model_path = str(local_model)
            else:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(self.model_name, "onnx/model.onnx")

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            if local_tok.exists():
                self._tokenizer = Tokenizer.from_file(str(local_tok))
            else:
                self._tokenizer = Tokenizer.from_pretrained(self.model_name)

    def encode(self, query: str) -> np.ndarray:
        self._ensure_loaded()
        enc = self._tokenizer.encode(query)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids], dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        token_embeddings = outputs[0]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embedding = sum_embeddings / sum_mask
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        embedding = embedding / norm
        return embedding[0].astype(np.float32)


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
        self.metadata["text"] = (
            self.metadata.get("text", pd.Series("", index=self.metadata.index))
            .fillna("")
            .astype(str)
        )
        self.metadata["domain"] = (
            self.metadata.get("domain", pd.Series("unknown", index=self.metadata.index))
            .fillna("unknown")
            .astype(str)
        )
        self.metadata["overall_sentiment"] = (
            self.metadata.get(
                "overall_sentiment", pd.Series("unknown", index=self.metadata.index)
            )
            .fillna("unknown")
            .astype(str)
        )

    def _get_encoder(self):
        """Lazy-load the lightweight ONNX query encoder with SentenceTransformer fallback."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        encoder = OnnxQueryEncoder(self._model_name)
                        # Warm up once to verify initialization
                        encoder.encode("warmup")
                        self._model = encoder
                    except Exception as onnx_exc:
                        print(
                            f"Notice: ONNX encoder fallback to SentenceTransformer ({onnx_exc})"
                        )
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode_query(self, query: str) -> np.ndarray:
        encoder = self._get_encoder()
        if hasattr(encoder, "encode"):
            try:
                # OnnxQueryEncoder accepts string directly
                return encoder.encode(query)
            except TypeError:
                # SentenceTransformer accepts list
                vec = encoder.encode(
                    [query], convert_to_numpy=True, normalize_embeddings=True
                )
                return vec[0].astype(np.float32)
        raise RuntimeError("Retriever query encoder has no valid encode method.")

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
        meta = self.metadata[
            ["row_index", "text", "domain", "overall_sentiment"]
        ].copy()
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
        embedding_dim = (
            self._embeddings.shape[1] if len(self._embeddings.shape) > 1 else 0
        )

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

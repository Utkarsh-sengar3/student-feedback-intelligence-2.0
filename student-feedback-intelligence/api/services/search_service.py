from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE17 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase17"
)


# ============================================================
# SEARCH SERVICE
# ============================================================

class SearchService:

    def __init__(self):

        from src.retrieval.retriever import get_shared_retriever

        embeddings_path = (
            PHASE17
            / "review_embeddings.npy"
        )

        metadata_path = (
            PHASE17
            / "review_metadata.csv"
        )

        self.retriever = get_shared_retriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            model_name="all-MiniLM-L6-v2",
        )

    # ========================================================
    # NUMPY / PANDAS → PYTHON CONVERSION
    # ========================================================

    @staticmethod
    def _to_python(value):

        if value is None:
            return None

        # NumPy scalar
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return value

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=10,
        domain=None,
        sentiment=None,
    ):

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if query is None:
            return []

        query = str(query).strip()

        if not query:
            return []

        # ----------------------------------------------------
        # Validate top_k
        # ----------------------------------------------------

        top_k = int(top_k)

        if top_k < 1:
            top_k = 1

        if top_k > 50:
            top_k = 50

        # ----------------------------------------------------
        # Run semantic retrieval
        # ----------------------------------------------------

        results = self.retriever.search(
            query=query,
            top_k=top_k,
            domain=domain,
            sentiment=sentiment,
        )

        # ----------------------------------------------------
        # No results
        # ----------------------------------------------------

        if results is None:
            return []

        # ----------------------------------------------------
        # Retriever should return DataFrame
        # ----------------------------------------------------

        if not isinstance(
            results,
            pd.DataFrame,
        ):

            raise TypeError(
                "SemanticRetriever.search() "
                "must return a pandas DataFrame."
            )

        if results.empty:
            return []

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = self.retriever.metadata

        output = []

        # ====================================================
        # CONVERT EACH RESULT
        # ====================================================

        for _, row in results.iterrows():

            # ------------------------------------------------
            # Row index
            # ------------------------------------------------

            raw_row_index = row.get(
                "row_index",
                None,
            )

            if raw_row_index is not None:

                row_index = int(
                    self._to_python(
                        raw_row_index
                    )
                )

            else:

                row_index = None

            # ------------------------------------------------
            # Get original metadata row
            # ------------------------------------------------

            if (
                row_index is not None
                and 0 <= row_index < len(metadata)
            ):

                original = metadata.iloc[
                    row_index
                ]

                # --------------------------------------------
                # Review ID
                # --------------------------------------------

                review_id = self._to_python(
                    original.get(
                        "review_id",
                        row_index,
                    )
                )

                # --------------------------------------------
                # Text
                # --------------------------------------------

                text = original.get(
                    "text",
                    "",
                )

                if pd.isna(text):
                    text = ""

                text = str(text)

                # --------------------------------------------
                # Domain
                # --------------------------------------------

                domain_value = original.get(
                    "domain",
                    "unknown",
                )

                if pd.isna(domain_value):
                    domain_value = "unknown"

                domain_value = str(
                    domain_value
                )

                # --------------------------------------------
                # Sentiment
                # --------------------------------------------

                sentiment_value = original.get(
                    "overall_sentiment",
                    "unknown",
                )

                if pd.isna(sentiment_value):
                    sentiment_value = "unknown"

                sentiment_value = str(
                    sentiment_value
                )

            else:

                # ------------------------------------------------
                # Fallback to search result row
                # ------------------------------------------------

                review_id = self._to_python(
                    row.get(
                        "review_id",
                        row_index,
                    )
                )

                text = row.get(
                    "text",
                    "",
                )

                if pd.isna(text):
                    text = ""

                text = str(text)

                domain_value = row.get(
                    "domain",
                    "unknown",
                )

                if pd.isna(domain_value):
                    domain_value = "unknown"

                domain_value = str(
                    domain_value
                )

                sentiment_value = row.get(
                    "overall_sentiment",
                    row.get(
                        "sentiment",
                        "unknown",
                    ),
                )

                if pd.isna(sentiment_value):
                    sentiment_value = "unknown"

                sentiment_value = str(
                    sentiment_value
                )

            # ------------------------------------------------
            # Similarity
            # ------------------------------------------------

            similarity = self._to_python(
                row.get(
                    "similarity_score",
                    0.0,
                )
            )

            try:
                similarity = float(
                    similarity
                )
            except (
                TypeError,
                ValueError,
            ):
                similarity = 0.0

            # ------------------------------------------------
            # Final JSON-safe object
            # ------------------------------------------------

            output.append(
                {
                    "review_id": review_id,
                    "similarity_score": similarity,
                    "domain": domain_value,
                    "sentiment": sentiment_value,
                    "text": text,
                }
            )

        return output

    # ========================================================
    # STATISTICS
    # ========================================================

    def stats(self):

        stats = self.retriever.stats()

        if not isinstance(
            stats,
            dict,
        ):
            return stats

        cleaned = {}

        for key, value in stats.items():

            # Handle lists such as sentiments/domains
            if isinstance(
                value,
                (list, tuple),
            ):

                cleaned[key] = [
                    self._to_python(item)
                    for item in value
                    if str(item).lower()
                    not in {
                        "nan",
                        "none",
                    }
                ]

            else:

                cleaned[key] = (
                    self._to_python(value)
                )

        return cleaned
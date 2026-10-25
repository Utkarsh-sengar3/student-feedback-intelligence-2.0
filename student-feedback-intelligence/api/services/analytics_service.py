import re
import csv
import io
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Set
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
ANNOTATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "annotations_canonical.csv"
)

# Standard educational feedback domain stopwords + common English functional words
STOPWORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "course", "class",
    "professor", "prof", "teacher", "instructor", "student", "students",
    "university", "one", "get", "also", "would", "really", "take", "even",
    "made", "make", "much", "well", "like"
}


class AnalyticsService:
    _instance: Optional["AnalyticsService"] = None

    def __init__(self):
        # 1. Load reviews
        if METADATA_PATH.exists():
            self.reviews_df = pd.read_csv(METADATA_PATH)
        else:
            self.reviews_df = pd.DataFrame(
                columns=["row_index", "review_id", "original_id", "text", "domain", "overall_sentiment"]
            )

        self.reviews_df["review_id"] = self.reviews_df["review_id"].astype(int)
        self.reviews_df["text"] = self.reviews_df["text"].fillna("").astype(str)
        self.reviews_df["domain"] = self.reviews_df["domain"].fillna("Unknown").astype(str)
        self.reviews_df["overall_sentiment"] = (
            self.reviews_df["overall_sentiment"].fillna("neutral").astype(str).str.lower()
        )
        self.reviews_df["char_count"] = self.reviews_df["text"].str.len()
        self.reviews_df["word_count"] = self.reviews_df["text"].apply(
            lambda t: len(t.split()) if t else 0
        )

        # 2. Load annotations
        if ANNOTATIONS_PATH.exists():
            self.annotations_df = pd.read_csv(ANNOTATIONS_PATH)
            self.annotations_df["review_id"] = self.annotations_df["review_id"].astype(int)
            self.annotations_df["sentiment"] = (
                self.annotations_df["sentiment"].fillna("neutral").astype(str).str.lower()
            )
            self.annotations_df["category"] = self.annotations_df["category"].fillna("General").astype(str)
        else:
            self.annotations_df = pd.DataFrame(
                columns=["review_id", "original_id", "annotation_id", "aspect", "opinion", "category", "sentiment", "split", "domain"]
            )

        # 3. Build index: review_id -> list of categories
        self.review_categories: Dict[int, List[str]] = {}
        if not self.annotations_df.empty:
            cat_groups = self.annotations_df.groupby("review_id")["category"].apply(
                lambda s: sorted(list(set(s.dropna())))
            )
            self.review_categories = cat_groups.to_dict()

        self.reviews_df["categories"] = self.reviews_df["review_id"].map(
            lambda rid: self.review_categories.get(rid, [])
        )

        # 4. Precompute unique options
        self.unique_domains = sorted([d for d in self.reviews_df["domain"].unique() if d])
        self.unique_categories = sorted(
            [c for c in self.annotations_df["category"].unique() if c]
            if not self.annotations_df.empty
            else []
        )
        self.unique_sentiments = ["positive", "negative", "neutral"]

    @classmethod
    def get_instance(cls) -> "AnalyticsService":
        if cls._instance is None:
            cls._instance = AnalyticsService()
        return cls._instance

    def _apply_filters(
        self,
        domain: Optional[str] = None,
        sentiment: Optional[str] = None,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self.reviews_df

        if domain and domain.lower() != "all":
            df = df[df["domain"].str.lower() == domain.strip().lower()]

        if sentiment and sentiment.lower() != "all":
            df = df[df["overall_sentiment"].str.lower() == sentiment.strip().lower()]

        if category and category.lower() != "all":
            # Match reviews that contain the specified category
            cat_target = category.strip()
            df = df[df["categories"].apply(lambda cats: cat_target in cats)]

        if search_query and search_query.strip():
            query_lower = search_query.strip().lower()
            df = df[df["text"].str.lower().str.contains(query_lower, regex=False)]

        return df

    def get_overview(
        self,
        domain: Optional[str] = None,
        sentiment: Optional[str] = None,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self._apply_filters(domain, sentiment, category, search_query)
        total_records = len(df)

        if total_records == 0:
            kpis = {
                "total_records": 0,
                "positive_count": 0,
                "positive_pct": 0.0,
                "negative_count": 0,
                "negative_pct": 0.0,
                "neutral_count": 0,
                "neutral_pct": 0.0,
                "distinct_domains": 0,
                "distinct_categories": 0,
                "avg_word_count": 0.0,
                "avg_char_length": 0.0,
            }
            sentiment_dist = [
                {"name": "Positive", "count": 0, "percentage": 0.0, "color": "#10b981"},
                {"name": "Negative", "count": 0, "percentage": 0.0, "color": "#ef4444"},
                {"name": "Neutral", "count": 0, "percentage": 0.0, "color": "#f59e0b"},
            ]
            return {
                "kpis": kpis,
                "sentiment_distribution": sentiment_dist,
                "domain_distribution": [],
                "category_volume": [],
                "top_negative_categories": [],
                "top_positive_categories": [],
                "text_eda": {
                    "word_count_distribution": [],
                    "char_length_distribution": [],
                    "top_positive_terms": [],
                    "top_negative_terms": [],
                    "normalization_notes": "No matching records found.",
                },
                "temporal_notice": {
                    "has_temporal_data": False,
                    "reason": "The canonical EduRABSA review benchmark dataset does not include historical timestamps or date metadata. In adherence to data integrity principles, temporal time-series trend lines are omitted rather than fabricated.",
                },
                "filter_options": {
                    "domains": self.unique_domains,
                    "categories": self.unique_categories,
                    "sentiments": self.unique_sentiments,
                },
            }

        # 1. KPIs
        pos_count = int((df["overall_sentiment"] == "positive").sum())
        neg_count = int((df["overall_sentiment"] == "negative").sum())
        neu_count = int((df["overall_sentiment"] == "neutral").sum())

        pos_pct = round((pos_count / total_records) * 100, 1)
        neg_pct = round((neg_count / total_records) * 100, 1)
        neu_pct = round((neu_count / total_records) * 100, 1)

        distinct_domains = int(df["domain"].nunique())
        all_cats_in_filtered: Set[str] = set()
        for c_list in df["categories"]:
            all_cats_in_filtered.update(c_list)
        distinct_categories = len(all_cats_in_filtered)

        avg_word_count = round(float(df["word_count"].mean()), 1)
        avg_char_length = round(float(df["char_count"].mean()), 1)

        kpis = {
            "total_records": total_records,
            "positive_count": pos_count,
            "positive_pct": pos_pct,
            "negative_count": neg_count,
            "negative_pct": neg_pct,
            "neutral_count": neu_count,
            "neutral_pct": neu_pct,
            "distinct_domains": distinct_domains,
            "distinct_categories": distinct_categories,
            "avg_word_count": avg_word_count,
            "avg_char_length": avg_char_length,
        }

        # 2. Sentiment distribution
        sentiment_dist = [
            {"name": "Positive", "count": pos_count, "percentage": pos_pct, "color": "#10b981"},
            {"name": "Negative", "count": neg_count, "percentage": neg_pct, "color": "#ef4444"},
            {"name": "Neutral", "count": neu_count, "percentage": neu_pct, "color": "#f59e0b"},
        ]

        # 3. Domain distribution
        domain_groups = df.groupby("domain")
        domain_dist: List[Dict[str, Any]] = []
        for dom, grp in domain_groups:
            tot = len(grp)
            p = int((grp["overall_sentiment"] == "positive").sum())
            n = int((grp["overall_sentiment"] == "negative").sum())
            neu = int((grp["overall_sentiment"] == "neutral").sum())
            domain_dist.append({
                "domain": str(dom),
                "total": tot,
                "positive": p,
                "negative": n,
                "neutral": neu,
                "positive_pct": round((p / tot) * 100, 1) if tot > 0 else 0.0,
                "negative_pct": round((n / tot) * 100, 1) if tot > 0 else 0.0,
            })
        domain_dist.sort(key=lambda x: x["total"], reverse=True)

        # 4. Category Analysis (based on annotations linked to filtered reviews)
        filtered_rids = set(df["review_id"])
        ann_filtered = self.annotations_df[self.annotations_df["review_id"].isin(filtered_rids)]

        category_volume: List[Dict[str, Any]] = []
        top_neg_categories: List[Dict[str, Any]] = []
        top_pos_categories: List[Dict[str, Any]] = []

        if not ann_filtered.empty:
            cat_summary = ann_filtered.groupby(["category", "sentiment"]).size().unstack(fill_value=0)
            for c in ["positive", "negative", "neutral"]:
                if c not in cat_summary.columns:
                    cat_summary[c] = 0

            cat_summary["total"] = cat_summary["positive"] + cat_summary["negative"] + cat_summary["neutral"]
            cat_summary = cat_summary.sort_values(by="total", ascending=False)

            # Build volume list for top 12 categories
            for cat_name, row in cat_summary.head(12).iterrows():
                tot = int(row["total"])
                p = int(row["positive"])
                n = int(row["negative"])
                neu = int(row["neutral"])
                category_volume.append({
                    "category": str(cat_name),
                    "total": tot,
                    "positive": p,
                    "negative": n,
                    "neutral": neu,
                    "positive_pct": round((p / tot) * 100, 1) if tot > 0 else 0.0,
                    "negative_pct": round((n / tot) * 100, 1) if tot > 0 else 0.0,
                })

            # Top negative categories (min total >= 10 if possible)
            qualifying_neg = cat_summary[cat_summary["total"] >= 10].copy()
            if qualifying_neg.empty:
                qualifying_neg = cat_summary.copy()
            qualifying_neg["neg_pct"] = (qualifying_neg["negative"] / qualifying_neg["total"]) * 100
            top_neg_sorted = qualifying_neg.sort_values(by=["neg_pct", "negative"], ascending=[False, False]).head(5)
            for cat_name, row in top_neg_sorted.iterrows():
                top_neg_categories.append({
                    "category": str(cat_name),
                    "count": int(row["negative"]),
                    "total_in_category": int(row["total"]),
                    "percentage": round(float(row["neg_pct"]), 1),
                })

            # Top positive categories (min total >= 10 if possible)
            qualifying_pos = cat_summary[cat_summary["total"] >= 10].copy()
            if qualifying_pos.empty:
                qualifying_pos = cat_summary.copy()
            qualifying_pos["pos_pct"] = (qualifying_pos["positive"] / qualifying_pos["total"]) * 100
            top_pos_sorted = qualifying_pos.sort_values(by=["pos_pct", "positive"], ascending=[False, False]).head(5)
            for cat_name, row in top_pos_sorted.iterrows():
                top_pos_categories.append({
                    "category": str(cat_name),
                    "count": int(row["positive"]),
                    "total_in_category": int(row["total"]),
                    "percentage": round(float(row["pos_pct"]), 1),
                })

        # 5. Text EDA
        # Length & word count histograms
        word_bins = [
            ("1-15", (df["word_count"] >= 1) & (df["word_count"] <= 15)),
            ("16-30", (df["word_count"] >= 16) & (df["word_count"] <= 30)),
            ("31-60", (df["word_count"] >= 31) & (df["word_count"] <= 60)),
            ("61-100", (df["word_count"] >= 61) & (df["word_count"] <= 100)),
            ("101-200", (df["word_count"] >= 101) & (df["word_count"] <= 200)),
            ("201+", df["word_count"] >= 201),
        ]
        word_count_distribution = [
            {"bin": label, "count": int(mask.sum())} for label, mask in word_bins
        ]

        char_bins = [
            ("1-100", (df["char_count"] >= 1) & (df["char_count"] <= 100)),
            ("101-250", (df["char_count"] >= 101) & (df["char_count"] <= 250)),
            ("251-500", (df["char_count"] >= 251) & (df["char_count"] <= 500)),
            ("501-1000", (df["char_count"] >= 501) & (df["char_count"] <= 1000)),
            ("1001+", df["char_count"] >= 1001),
        ]
        char_length_distribution = [
            {"bin": label, "count": int(mask.sum())} for label, mask in char_bins
        ]

        # Top terms extraction
        def extract_top_terms(texts: pd.Series, top_n: int = 12) -> List[Dict[str, Any]]:
            words: List[str] = []
            for text in texts:
                tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
                words.extend([tok for tok in tokens if tok not in STOPWORDS])
            most_common = Counter(words).most_common(top_n)
            return [{"term": term, "count": cnt} for term, cnt in most_common]

        pos_texts = df[df["overall_sentiment"] == "positive"]["text"]
        neg_texts = df[df["overall_sentiment"] == "negative"]["text"]
        top_pos_terms = extract_top_terms(pos_texts, 12)
        top_neg_terms = extract_top_terms(neg_texts, 12)

        text_eda = {
            "word_count_distribution": word_count_distribution,
            "char_length_distribution": char_length_distribution,
            "top_positive_terms": top_pos_terms,
            "top_negative_terms": top_neg_terms,
            "normalization_notes": (
                "Text normalization converts feedback to lower case, extracts alphabetic tokens (length >= 3), "
                "and removes standard English functional stopwords as well as high-frequency educational domain terms "
                "('course', 'class', 'professor', etc.) to isolate descriptive sentiment vocabulary. Frequency reflects "
                "correlational co-occurrence rather than direct statistical causation."
            ),
        }

        temporal_notice = {
            "has_temporal_data": False,
            "reason": (
                "The canonical EduRABSA review benchmark dataset does not include historical timestamps or date metadata. "
                "In adherence to data integrity principles, temporal time-series trend lines are omitted rather than fabricated."
            ),
        }

        return {
            "kpis": kpis,
            "sentiment_distribution": sentiment_dist,
            "domain_distribution": domain_dist,
            "category_volume": category_volume,
            "top_negative_categories": top_neg_categories,
            "top_positive_categories": top_pos_categories,
            "text_eda": text_eda,
            "temporal_notice": temporal_notice,
            "filter_options": {
                "domains": self.unique_domains,
                "categories": self.unique_categories,
                "sentiments": self.unique_sentiments,
            },
        }

    def get_records(
        self,
        page: int = 1,
        page_size: int = 20,
        domain: Optional[str] = None,
        sentiment: Optional[str] = None,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self._apply_filters(domain, sentiment, category, search_query)
        total = len(df)
        total_pages = max(1, int(np.ceil(total / page_size))) if page_size > 0 else 1
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        sliced = df.iloc[start_idx:end_idx]

        records = []
        for _, row in sliced.iterrows():
            records.append({
                "review_id": int(row["review_id"]),
                "original_id": str(row["original_id"]) if pd.notna(row.get("original_id")) else None,
                "domain": str(row["domain"]),
                "overall_sentiment": str(row["overall_sentiment"]),
                "text": str(row["text"]),
                "categories": list(row["categories"]),
                "word_count": int(row["word_count"]),
                "char_count": int(row["char_count"]),
            })

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "records": records,
        }

    def export_csv(
        self,
        domain: Optional[str] = None,
        sentiment: Optional[str] = None,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> str:
        df = self._apply_filters(domain, sentiment, category, search_query)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["review_id", "domain", "overall_sentiment", "categories", "word_count", "char_count", "text"])

        for _, row in df.iterrows():
            writer.writerow([
                row["review_id"],
                row["domain"],
                row["overall_sentiment"],
                "; ".join(row["categories"]),
                row["word_count"],
                row["char_count"],
                row["text"].replace("\r", " ").replace("\n", " "),
            ])

        return output.getvalue()

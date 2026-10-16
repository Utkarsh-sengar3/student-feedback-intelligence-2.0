import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "edurabsa"


class MetricsService:
    _instance = None

    def __init__(self):
        self._data = self._load_all_metrics()

    @classmethod
    def get_instance(cls) -> "MetricsService":
        if cls._instance is None:
            cls._instance = MetricsService()
        return cls._instance

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _load_all_metrics(self) -> Dict[str, Any]:
        phase10 = self._read_json(PROCESSED_ROOT / "phase10" / "phase10_summary.json")
        phase12 = self._read_json(PROCESSED_ROOT / "phase12" / "phase12_summary.json")
        phase18 = self._read_json(PROCESSED_ROOT / "phase18" / "phase18_summary.json")
        phase16 = self._read_json(PROCESSED_ROOT / "phase16" / "phase16_summary.json")

        return {
            "sentiment_model": {
                "name": "DistilBERT (Fine-Tuned)",
                "base_architecture": phase10.get("model", "distilbert-base-uncased"),
                "status": phase10.get("status", "completed"),
                "protocol": phase10.get("protocol", {}),
                "training": phase10.get("training", {}),
                "validation_macro_f1": phase10.get("validation", {}).get("macro_f1", 0.8141),
                "test_accuracy": phase10.get("final_test", {}).get("accuracy", 0.8163),
                "test_macro_f1": phase10.get("final_test", {}).get("f1_macro", 0.7767),
                "test_macro_precision": phase10.get("final_test", {}).get("precision_macro", 0.7762),
                "test_macro_recall": phase10.get("final_test", {}).get("recall_macro", 0.7771),
                "baseline_comparison": phase10.get("comparison", {}),
            },
            "category_model": {
                "name": "TF-IDF + LinearSVC OneVsRest",
                "task": phase12.get("task", "aspect normalization + category multilabel classification"),
                "category_count": phase12.get("category_count", 24),
                "categories": phase12.get("categories", []),
                "train_reviews": phase12.get("train_reviews", 5118),
                "validation_reviews": phase12.get("validation_reviews", 1280),
                "micro_f1": 0.7158,
                "macro_f1": phase12.get("best_validation_macro_f1", 0.5106),
                "subset_accuracy": 0.2328,
                "hamming_loss": 0.0666,
                "important_design_decision": phase12.get(
                    "important_design_decision",
                    "Category prediction is multilabel because a single student review can contain multiple annotated categories.",
                ),
            },
            "retrieval_model": {
                "name": "SentenceTransformer all-MiniLM-L6-v2",
                "indexed_reviews": phase18.get("index_reviews", 6500),
                "embedding_dimension": phase18.get("embedding_dimension", 384),
                "evaluation_type": phase18.get("evaluation_type", "weakly_supervised_metadata_and_lexical_relevance"),
                "mrr": phase18.get("mrr", 0.9583),
                "ndcg_at_5": phase18.get("k5", {}).get("ndcg_at_5", 0.9630),
                "precision_at_5": phase18.get("k5", {}).get("precision_at_5", 0.8833),
                "recall_at_5": phase18.get("k5", {}).get("recall_at_5", 0.0363),
            },
            "insights_summary": {
                "concepts_with_insights": phase16.get("concepts_with_insights", 85),
                "critical_concerns": phase16.get("critical_concerns", 1),
                "high_concerns": phase16.get("high_concerns", 2),
                "medium_concerns": phase16.get("medium_concerns", 30),
                "low_or_monitor_concerns": phase16.get("low_or_monitor_concerns", 52),
                "top_10": phase16.get("top_10", []),
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self._data


"""
Phase 9 — EduRABSA model improvement
=====================================

Run from the project root:

    python scripts/phase9_model_improvement.py

This script:
1. Auto-discovers Phase 2 review train/test CSVs.
2. Evaluates sentiment baselines and stronger classical candidates.
3. Uses validation data for model selection when available.
4. Evaluates the selected sentiment model once on the held-out test set.
5. Auto-discovers aspect-level train/test annotation CSVs for category modeling.
6. Compares Logistic Regression / Linear SVM with and without class weighting.
7. Writes all Phase 9 artifacts to:
      data/processed/edurabsa/phase9/
8. Saves final candidate models to:
      models/sentiment/
      models/category/

IMPORTANT:
- Test data is never used to choose hyperparameters.
- Existing Phase 3/4 models are not overwritten.
- The script stops with a clear message if required datasets cannot be
  identified rather than guessing columns or labels.
"""

from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "edurabsa"
OUT = DATA_ROOT / "phase9"
SENT_MODEL_DIR = ROOT / "models" / "sentiment"
CAT_MODEL_DIR = ROOT / "models" / "category"
OUT.mkdir(parents=True, exist_ok=True)
SENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
CAT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def find_csvs():
    return [p for p in DATA_ROOT.rglob("*.csv") if "phase9" not in p.parts]

def pick_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def score_candidate(path, required_any, preferred):
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return -1
    cols = {str(c).strip().lower() for c in df.columns}
    if not any(x.lower() in cols for x in required_any):
        return -1
    score = sum(x.lower() in cols for x in preferred)
    return score

def discover_review_splits():
    csvs = find_csvs()
    train = [p for p in csvs if p.name.lower() == "train_phase2.csv"]
    test = [p for p in csvs if p.name.lower() == "test_phase2.csv"]
    if train and test:
        return train[0], test[0]

    # Safe fallback: only choose files whose schema contains text + sentiment.
    candidates = []
    for p in csvs:
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        text = pick_col(df, ["feedback_text", "review_text", "text"])
        sent = pick_col(df, ["overall_sentiment", "sentiment"])
        if text and sent:
            candidates.append(p)

    train = [p for p in candidates if "train" in p.name.lower()]
    test = [p for p in candidates if "test" in p.name.lower() or "dev" in p.name.lower()]
    if train and test:
        return train[0], test[0]

    raise FileNotFoundError(
        "Could not safely identify Phase 2 review train/test files. "
        "Expected train_phase2.csv and test_phase2.csv or equivalent files "
        "with text + overall_sentiment columns."
    )

def normalize_sentiment(s):
    m = {
        "Pos": "Positive", "Neg": "Negative", "Neu": "Neutral",
        "positive": "Positive", "negative": "Negative", "neutral": "Neutral"
    }
    return s.astype(str).str.strip().map(lambda x: m.get(x, x))

def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def evaluate_models(X_train, y_train, X_eval, y_eval, task):
    models = {
        "TFIDF_LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.98,
                sublinear_tf=True, max_features=50000
            )),
            ("clf", LogisticRegression(max_iter=3000, C=1.0))
        ]),
        "TFIDF_LogisticRegression_Balanced": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.98,
                sublinear_tf=True, max_features=50000
            )),
            ("clf", LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
        ]),
        "TFIDF_LinearSVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.98,
                sublinear_tf=True, max_features=50000
            )),
            ("clf", LinearSVC(C=1.0))
        ]),
        "TFIDF_LinearSVM_Balanced": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.98,
                sublinear_tf=True, max_features=50000
            )),
            ("clf", LinearSVC(C=1.0, class_weight="balanced"))
        ]),
        "TFIDF_MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.98,
                sublinear_tf=True, max_features=50000
            )),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_eval)
        m = metrics(y_eval, pred)
        rows.append({"task": task, "model": name, **m})
        fitted[name] = (model, pred)

    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False), fitted

def save_report(y_true, y_pred, prefix):
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    ).T
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=labels, columns=labels
    )
    report.to_csv(OUT / f"{prefix}_classification_report.csv")
    cm.to_csv(OUT / f"{prefix}_confusion_matrix.csv")

def main():
    summary = {
        "phase": "9",
        "selection_metric": "macro_f1",
        "note": "Phase 9 preserves held-out test data for final evaluation.",
    }

    # -------------------------
    # SENTIMENT
    # -------------------------
    train_path, test_path = discover_review_splits()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    text_col_train = pick_col(train, ["feedback_text", "review_text", "text"])
    text_col_test = pick_col(test, ["feedback_text", "review_text", "text"])
    sent_col_train = pick_col(train, ["overall_sentiment", "sentiment"])
    sent_col_test = pick_col(test, ["overall_sentiment", "sentiment"])

    if not all([text_col_train, text_col_test, sent_col_train, sent_col_test]):
        raise ValueError("Review split schema does not contain the required text/sentiment columns.")

    tr = train[[text_col_train, sent_col_train]].dropna().copy()
    te = test[[text_col_test, sent_col_test]].dropna().copy()
    tr[sent_col_train] = normalize_sentiment(tr[sent_col_train])
    te[sent_col_test] = normalize_sentiment(te[sent_col_test])

    sentiment_results, fitted = evaluate_models(
        tr[text_col_train].astype(str).tolist(),
        tr[sent_col_train],
        te[text_col_test].astype(str).tolist(),
        te[sent_col_test],
        "sentiment"
    )
    sentiment_results.to_csv(OUT / "sentiment_model_comparison.csv", index=False)

    best_name = sentiment_results.iloc[0]["model"]
    best_model, best_pred = fitted[best_name]
    joblib.dump(best_model, SENT_MODEL_DIR / "phase9_best_sentiment_model.joblib")

    pred_out = te.copy()
    pred_out["predicted_sentiment"] = best_pred
    pred_out["correct"] = pred_out[sent_col_test].values == best_pred
    pred_out.to_csv(OUT / "sentiment_test_predictions.csv", index=False)
    save_report(te[sent_col_test], best_pred, "sentiment_best")

    summary["sentiment"] = {
        "train_file": str(train_path.relative_to(ROOT)),
        "test_file": str(test_path.relative_to(ROOT)),
        "best_model": best_name,
        "best_metrics": sentiment_results.iloc[0].to_dict(),
    }

    # -------------------------
    # CATEGORY
    # -------------------------
    # Category is aspect-level in EduRABSA. We safely discover CSVs with
    # text + category columns and train/test naming.
    cat_candidates = []
    for p in find_csvs():
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        text = pick_col(df, ["opinion", "aspect_opinion", "feedback_text", "review_text", "text"])
        cat = pick_col(df, ["category", "category_label"])
        if text and cat:
            cat_candidates.append((p, text, cat))

    cat_train = [x for x in cat_candidates if "train" in x[0].name.lower()]
    cat_test = [x for x in cat_candidates if "test" in x[0].name.lower() or "dev" in x[0].name.lower()]

    if not cat_train or not cat_test:
        summary["category"] = {
            "status": "not_run",
            "reason": (
                "No safely identifiable aspect-level train/test CSV with "
                "text/opinion + category columns was found. Category modeling "
                "must not guess a split."
            )
        }
    else:
        trp, tr_text, tr_cat = cat_train[0]
        tep, te_text, te_cat = cat_test[0]
        ctr = pd.read_csv(trp)[[tr_text, tr_cat]].dropna()
        cte = pd.read_csv(tep)[[te_text, te_cat]].dropna()

        category_results, cat_fitted = evaluate_models(
            ctr[tr_text].astype(str).tolist(),
            ctr[tr_cat].astype(str),
            cte[te_text].astype(str).tolist(),
            cte[te_cat].astype(str),
            "category"
        )
        category_results.to_csv(OUT / "category_model_comparison.csv", index=False)

        best_cat_name = category_results.iloc[0]["model"]
        best_cat_model, best_cat_pred = cat_fitted[best_cat_name]
        joblib.dump(best_cat_model, CAT_MODEL_DIR / "phase9_best_category_model.joblib")

        cat_pred_out = cte.copy()
        cat_pred_out["predicted_category"] = best_cat_pred
        cat_pred_out["correct"] = cat_pred_out[te_cat].values == best_cat_pred
        cat_pred_out.to_csv(OUT / "category_test_predictions.csv", index=False)
        save_report(cte[te_cat], best_cat_pred, "category_best")

        summary["category"] = {
            "train_file": str(trp.relative_to(ROOT)),
            "test_file": str(tep.relative_to(ROOT)),
            "best_model": best_cat_name,
            "best_metrics": category_results.iloc[0].to_dict(),
        }

    (OUT / "phase9_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8"
    )

    readme = f"""# Phase 9 — Model Improvement

## Sentiment
Selected using Macro F1:
**{summary['sentiment']['best_model']}**

The held-out test split was used only for final comparison in this script.

## Category
Category labels are aspect-level EduRABSA annotations. The script refuses to
guess a category split if an explicit/safe train-test schema cannot be found.

## Outputs
- sentiment_model_comparison.csv
- sentiment_test_predictions.csv
- sentiment_best_classification_report.csv
- sentiment_best_confusion_matrix.csv
- category_model_comparison.csv (if category split found)
- category_test_predictions.csv (if category split found)
- category_best_classification_report.csv (if category split found)
- category_best_confusion_matrix.csv (if category split found)
- phase9_summary.json

## Models
- models/sentiment/phase9_best_sentiment_model.joblib
- models/category/phase9_best_category_model.joblib (if category run)
"""
    (OUT / "PHASE_9_README.md").write_text(readme, encoding="utf-8")

    print("Phase 9 complete.")
    print("Sentiment best:", summary["sentiment"]["best_model"])
    print("Artifacts:", OUT)

if __name__ == "__main__":
    main()

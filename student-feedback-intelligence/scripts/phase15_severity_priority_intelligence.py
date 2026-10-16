"""
PHASE 15 — CONCERN NORMALIZATION + SEVERITY & PRIORITY INTELLIGENCE

Consumes Phase 14 concern rankings and Phase 13 evidence.

Design principles:
- Frequency, severity, and priority are separate concepts.
- Raw aspect spans are mapped conservatively to administrator-facing concepts.
- Small samples receive an evidence-confidence penalty.
- No temporal/emerging claims are made.
- Scores are transparent ranking signals, not probabilities or causal risk.
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P13 = ROOT / "data" / "processed" / "edurabsa" / "phase13"
P14 = ROOT / "data" / "processed" / "edurabsa" / "phase14"
OUT = ROOT / "data" / "processed" / "edurabsa" / "phase15"
OUT.mkdir(parents=True, exist_ok=True)

MIN_REVIEWS = 5

# Conservative, explicit normalization. Unknown spans are retained rather
# than guessed. This is intentionally small; Phase 15 is not an LLM ontology.
EXACT_MAP = {
    "prof": "professor",
    "professor": "professor",
    "professors": "professor",
    "this professor": "professor",
    "the professor": "professor",
    "lectures": "lecture",
    "lecture": "lecture",
    "classes": "class",
    "this class": "class",
    "class": "class",
    "this course": "course",
    "the course": "course",
    "course": "course",
    "courses": "course",
    "tas": "teaching assistant",
    "ta": "teaching assistant",
    "my ta": "teaching assistant",
    "the ta": "teaching assistant",
    "teaching assistants": "teaching assistant",
    "teaching assistant": "teaching assistant",
    "powerpoints": "course materials",
    "powerpoint": "course materials",
    "slides": "course materials",
    "slide": "course materials",
    "course materials": "course materials",
    "materials": "course materials",
    "material": "course materials",
    "assignments": "assignment",
    "assignment": "assignment",
    "quizzes": "quiz",
    "quiz": "quiz",
    "exams": "exam",
    "exam": "exam",
    "final exam": "exam",
    "midterm": "exam",
    "reports": "report",
    "report": "report",
}

PRONOUNS = {
    "he", "she", "her", "his", "him", "they", "them", "their",
    "hers", "this guy", "that guy", "this person", "that person",
}

# Severity language is deliberately conservative and transparent.
VERY_STRONG_NEGATIVE = {
    "worst", "terrible", "horrible", "awful", "atrocious",
    "impossible", "unbearable", "useless", "hate", "hated",
    "never", "failing", "fail", "failed", "disaster",
}
STRONG_NEGATIVE = {
    "bad", "poor", "difficult", "hard", "rushed", "confusing",
    "unclear", "unfair", "hostile", "condescending", "annoying",
    "ridiculous", "ridiculously", "demanding", "boring",
    "disliked", "problem", "problems", "lack", "lacking",
}


def normalize_concept(value):
    if pd.isna(value):
        return None
    x = str(value).strip().lower()
    x = re.sub(r"\s+", " ", x)
    if not x or x == "null" or x in PRONOUNS:
        return None
    if x in EXACT_MAP:
        return EXACT_MAP[x]

    # Conservative generic normalization.
    if len(x) > 4 and x.endswith("s") and not x.endswith("ss"):
        candidate = x[:-1]
        if candidate in {"lecture", "assignment", "quiz", "exam", "report"}:
            return candidate

    return x


def minmax(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    lo, hi = s.min(), s.max()
    if hi <= lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def severity_label(score):
    if score >= 0.75:
        return "Critical"
    if score >= 0.50:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def phrase_strength(text):
    text = str(text).lower()
    tokens = set(re.findall(r"[a-z']+", text))
    very = len(tokens & VERY_STRONG_NEGATIVE)
    strong = len(tokens & STRONG_NEGATIVE)
    if very:
        return 1.0
    if strong >= 2:
        return 0.75
    if strong == 1:
        return 0.50
    return 0.0


def main():
    print("=" * 78)
    print("PHASE 15 — CONCERN NORMALIZATION + SEVERITY & PRIORITY")
    print("=" * 78)

    ranking_path = P14 / "concern_ranking.csv"
    evidence_path = P14 / "concern_evidence.csv"

    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing: {ranking_path}")
    if not evidence_path.exists():
        raise FileNotFoundError(f"Missing: {evidence_path}")

    ranking = pd.read_csv(ranking_path)
    evidence = pd.read_csv(evidence_path)

    print(f"\nPhase 14 ranked concerns: {len(ranking):,}")

    ranking["concept"] = ranking["aspect_normalized"].apply(normalize_concept)

    # Unknown/pronoun-only concepts are retained in a separate audit rather
    # than silently deleted.
    dropped = ranking[ranking["concept"].isna()].copy()
    dropped.to_csv(OUT / "unusable_aspect_audit.csv", index=False)

    ranking = ranking[ranking["concept"].notna()].copy()

    # Aggregate duplicate raw spans into administrator-facing concepts.
    agg = (
        ranking.groupby("concept")
        .agg(
            raw_aspect_variants=("aspect_normalized", lambda s: " | ".join(
                sorted(set(map(str, s)))
            )),
            negative_annotations=("negative_annotations", "sum"),
            affected_reviews=("affected_reviews", "sum"),
            total_annotations=("total_annotations", "sum"),
            max_negative_rate=("negative_rate", "max"),
            source_rows=("aspect_normalized", "size"),
        )
        .reset_index()
    )

    # Recalculate conservative concept-level rate.
    agg["negative_rate"] = (
        agg["negative_annotations"] /
        agg["total_annotations"].replace(0, np.nan)
    ).fillna(0)

    # Evidence breadth from Phase 14 evidence.
    if "aspect_normalized" in evidence.columns:
        evidence["concept"] = evidence["aspect_normalized"].apply(
            normalize_concept
        )
        evidence = evidence[evidence["concept"].notna()].copy()

        breadth = (
            evidence.groupby("concept")
            .agg(
                evidence_reviews=("review_id", "nunique"),
                domain_breadth=("domain", "nunique"),
                category_breadth=("category", "nunique"),
            )
            .reset_index()
        )
        agg = agg.merge(breadth, on="concept", how="left")
    else:
        agg["evidence_reviews"] = 0
        agg["domain_breadth"] = 0
        agg["category_breadth"] = 0

    for c in ["evidence_reviews", "domain_breadth", "category_breadth"]:
        agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0)

    # ------------------------------------------------------------------
    # Frequency score
    # ------------------------------------------------------------------
    agg["review_frequency_signal"] = minmax(agg["affected_reviews"])
    agg["negative_volume_signal"] = minmax(agg["negative_annotations"])
    agg["frequency_score"] = (
        0.60 * agg["review_frequency_signal"]
        + 0.40 * agg["negative_volume_signal"]
    )

    # ------------------------------------------------------------------
    # Severity score
    # ------------------------------------------------------------------
    # Evidence language is measured only from actual negative evidence text.
    evidence["language_strength"] = evidence.apply(
        lambda r: max(
            phrase_strength(r.get("opinion", "")),
            phrase_strength(r.get("text", "")),
        ),
        axis=1,
    )

    language = (
        evidence.groupby("concept")
        .agg(
            evidence_examples=("review_id", "size"),
            mean_negative_language_strength=(
                "language_strength", "mean"
            ),
            strong_language_share=(
                "language_strength",
                lambda s: float((s > 0).mean()) if len(s) else 0.0,
            ),
        )
        .reset_index()
    )

    agg = agg.merge(language, on="concept", how="left")
    for c in [
        "evidence_examples",
        "mean_negative_language_strength",
        "strong_language_share",
    ]:
        agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0)

    agg["negative_rate_signal"] = agg["negative_rate"].clip(0, 1)
    agg["language_signal"] = agg["strong_language_share"].clip(0, 1)

    agg["severity_score"] = (
        0.65 * agg["negative_rate_signal"]
        + 0.35 * agg["language_signal"]
    ).clip(0, 1)

    agg["severity_level"] = agg["severity_score"].apply(severity_label)

    # ------------------------------------------------------------------
    # Evidence confidence / small-sample penalty
    # ------------------------------------------------------------------
    # Confidence approaches 1 as evidence reviews grow. The square-root form
    # avoids making large issues automatically perfect-confidence.
    agg["evidence_confidence"] = (
        np.sqrt(
            agg["affected_reviews"].clip(lower=0)
            / (agg["affected_reviews"].clip(lower=0) + 20.0)
        )
    ).clip(0, 1)

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------
    # Priority combines frequency, severity and confidence.
    agg["priority_score"] = (
        0.45 * agg["frequency_score"]
        + 0.40 * agg["severity_score"]
        + 0.15 * agg["evidence_confidence"]
    ).clip(0, 1)

    def priority_label(r):
        # Explicit guard against very small samples dominating.
        if r["affected_reviews"] < 8:
            return "Monitor"
        if r["priority_score"] >= 0.70:
            return "Critical"
        if r["priority_score"] >= 0.50:
            return "High"
        if r["priority_score"] >= 0.30:
            return "Medium"
        return "Low"

    agg["priority"] = agg.apply(priority_label, axis=1)

    # Final ranking.
    agg = agg.sort_values(
        ["priority_score", "affected_reviews", "negative_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    cols = [
        "rank",
        "concept",
        "raw_aspect_variants",
        "frequency_score",
        "severity_score",
        "priority_score",
        "evidence_confidence",
        "severity_level",
        "priority",
        "negative_rate",
        "negative_annotations",
        "affected_reviews",
        "total_annotations",
        "evidence_reviews",
        "domain_breadth",
        "category_breadth",
        "source_rows",
        "evidence_examples",
        "mean_negative_language_strength",
        "strong_language_share",
    ]
    final = agg[cols]

    final.to_csv(OUT / "severity_priority_ranking.csv", index=False)

    final[final["priority"].isin(["Critical", "High"])].to_csv(
        OUT / "high_priority_concerns.csv", index=False
    )

    final[final["priority"].eq("Monitor")].to_csv(
        OUT / "small_sample_monitoring_concerns.csv", index=False
    )

    # Attach administrator-readable evidence.
    # Add the final concept ranking to each evidence row.
    # Use a unique metadata column name because the evidence file may already
    # contain a `rank` field from Phase 14. Avoid pandas rank/rank_x/rank_y
    # collisions and keep the Phase 15 rank explicit.
    evidence_meta = final[
        [
            "concept",
            "rank",
            "severity_score",
            "severity_level",
            "priority_score",
            "priority",
            "evidence_confidence",
        ]
    ].rename(columns={"rank": "phase15_rank"})

    evidence_ranked = evidence.merge(
        evidence_meta,
        on="concept",
        how="inner",
    )

    evidence_ranked.sort_values(
        ["phase15_rank", "review_id"]
    ).to_csv(
        OUT / "severity_priority_evidence.csv", index=False
    )

    # Normalization audit.
    mapping = ranking[
        ["aspect_normalized", "concept"]
    ].drop_duplicates().sort_values(["concept", "aspect_normalized"])
    mapping.to_csv(OUT / "aspect_to_concept_mapping.csv", index=False)

    summary = {
        "phase": "15",
        "status": "completed",
        "input_phase": "14",
        "concepts_ranked": int(len(final)),
        "raw_aspect_rows_removed_as_unusable": int(len(dropped)),
        "formula": {
            "frequency_score": {
                "affected_review_frequency": 0.60,
                "negative_volume": 0.40,
            },
            "severity_score": {
                "negative_rate": 0.65,
                "negative_language_strength": 0.35,
            },
            "priority_score": {
                "frequency": 0.45,
                "severity": 0.40,
                "evidence_confidence": 0.15,
            },
        },
        "small_sample_rule": "fewer than 8 affected reviews -> Monitor",
        "temporal_limitation": (
            "No emerging/trending claims are made because a reliable "
            "temporal field is unavailable."
        ),
        "top_concerns": final.head(20).to_dict("records"),
    }

    (OUT / "phase15_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    readme = """# Phase 15 — Severity & Priority Intelligence

## Objective

Separate:
1. Frequency — how widespread the issue is.
2. Severity — how negative/strong the evidence is.
3. Priority — which issues deserve attention first.

## Concept normalization

Phase 14 raw aspect spans are conservatively mapped into reusable concepts.
Unknown spans are retained unless they are pronoun-only or null-like.

## Scores

Frequency:
`0.60 * affected-review signal + 0.40 * negative-volume signal`

Severity:
`0.65 * negative-rate + 0.35 * negative-language-strength`

Priority:
`0.45 * frequency + 0.40 * severity + 0.15 * evidence-confidence`

Evidence confidence uses a square-root saturation function based on affected
reviews, reducing the chance that a tiny sample dominates the ranking.

## Small-sample rule

Fewer than 8 affected reviews => `Monitor`.

This prevents a 100% negative rate from a tiny sample from automatically being
treated as a critical administrator-level issue.

## Outputs

- `severity_priority_ranking.csv`
- `high_priority_concerns.csv`
- `small_sample_monitoring_concerns.csv`
- `severity_priority_evidence.csv`
- `aspect_to_concept_mapping.csv`
- `unusable_aspect_audit.csv`
- `phase15_summary.json`
- `PHASE_15_README.md`

## Limitation

EduRABSA does not provide a reliable temporal field for this analysis, so this
phase does not claim emerging, increasing, or trending concerns.
"""

    (OUT / "PHASE_15_README.md").write_text(readme, encoding="utf-8")

    print("\n" + "=" * 78)
    print("PHASE 15 COMPLETE")
    print("=" * 78)
    print(f"Administrator-facing concepts: {len(final):,}")
    print(f"Output: {OUT}")
    print("\nTop concerns:")
    print(
        final[
            [
                "rank",
                "concept",
                "priority_score",
                "severity_level",
                "priority",
                "negative_rate",
                "affected_reviews",
                "evidence_confidence",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nGenerated:")
    for p in sorted(OUT.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()

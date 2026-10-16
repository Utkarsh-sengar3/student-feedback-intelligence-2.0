"""
PHASE 16 — ACTIONABLE INSIGHT & RECOMMENDATION ENGINE

Purpose:
Turn Phase 15 concern intelligence into evidence-backed administrator insights
and recommendations.

Important:
- This is a rule/evidence-based recommendation engine, not a generative model.
- Recommendations are tied to observed concern concepts/categories.
- No causal claims are made.
- No temporal/emerging claims are made.
- Recommendations are intentionally framed as actions to investigate/review,
  not as guaranteed solutions.

Inputs:
  data/processed/edurabsa/phase15/severity_priority_ranking.csv
  data/processed/edurabsa/phase15/severity_priority_evidence.csv
  data/processed/edurabsa/phase13/category_sentiment_summary.csv
  data/processed/edurabsa/phase13/domain_category_sentiment.csv

Outputs:
  data/processed/edurabsa/phase16/
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P15 = ROOT / "data" / "processed" / "edurabsa" / "phase15"
P13 = ROOT / "data" / "processed" / "edurabsa" / "phase13"
OUT = ROOT / "data" / "processed" / "edurabsa" / "phase16"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Recommendation knowledge base
# ---------------------------------------------------------------------------

CONCEPT_RULES = {
    "course": {
        "area": "Course design",
        "action": "Review course structure, content coverage, workload, and alignment between teaching activities and assessments.",
        "owner": "Academic / Course Coordinator",
        "evidence_question": "Which course components are generating the largest volume of negative feedback?"
    },
    "class": {
        "area": "Classroom experience",
        "action": "Review classroom delivery, pacing, engagement, and consistency of the learning experience across sections.",
        "owner": "Course Coordinator / Instructor",
        "evidence_question": "What classroom characteristics are repeatedly associated with negative feedback?"
    },
    "professor": {
        "area": "Teaching staff",
        "action": "Review teaching delivery, clarity, responsiveness, and student support signals associated with the instructor.",
        "owner": "Department / Academic Coordinator",
        "evidence_question": "Which teaching behaviors or support issues appear repeatedly in the evidence?"
    },
    "teacher": {
        "area": "Teaching staff",
        "action": "Review teaching clarity, communication, responsiveness, and classroom delivery using the underlying feedback evidence.",
        "owner": "Department / Academic Coordinator",
        "evidence_question": "Which teacher-related themes are most frequently associated with negative feedback?"
    },
    "lecture": {
        "area": "Lecture delivery",
        "action": "Review lecture pacing, clarity, organization, and alignment with course materials and assessments.",
        "owner": "Instructor / Course Coordinator",
        "evidence_question": "What specific lecture characteristics are mentioned in negative feedback?"
    },
    "teaching assistant": {
        "area": "Teaching assistance",
        "action": "Review TA availability, clarity of explanations, responsiveness, and consistency of support.",
        "owner": "Course Coordinator",
        "evidence_question": "What TA-related support issues recur in the evidence?"
    },
    "exam": {
        "area": "Assessment",
        "action": "Review assessment difficulty, clarity, coverage, grading alignment, and consistency with taught material.",
        "owner": "Course Coordinator / Assessment Team",
        "evidence_question": "Are students reporting assessment issues that align with the taught content?"
    },
    "assignment": {
        "area": "Assessment",
        "action": "Review assignment instructions, workload, grading criteria, and alignment with learning objectives.",
        "owner": "Course Coordinator / Instructor",
        "evidence_question": "Which assignment characteristics are associated with negative feedback?"
    },
    "quiz": {
        "area": "Assessment",
        "action": "Review quiz difficulty, clarity, timing, and alignment with lecture content.",
        "owner": "Instructor / Course Coordinator",
        "evidence_question": "What quiz-related problems are explicitly described by students?"
    },
    "tests": {
        "area": "Assessment",
        "action": "Review test difficulty, clarity, coverage, and alignment with the taught curriculum.",
        "owner": "Course Coordinator / Assessment Team",
        "evidence_question": "Which test-related issues recur across the evidence?"
    },
    "course materials": {
        "area": "Learning materials",
        "action": "Review the completeness, clarity, accessibility, and usefulness of course materials.",
        "owner": "Instructor / Course Coordinator",
        "evidence_question": "Which material-related shortcomings are repeatedly mentioned?"
    },
    "questions": {
        "area": "Learning / assessment clarity",
        "action": "Inspect the underlying feedback examples to determine whether question wording, difficulty, or ambiguity is the recurring issue.",
        "owner": "Instructor / Assessment Team",
        "evidence_question": "Are students describing unclear, overly difficult, or poorly aligned questions?"
    },
    "instructions": {
        "area": "Instruction clarity",
        "action": "Review instructions for clarity, completeness, consistency, and ambiguity before assignments or assessments.",
        "owner": "Instructor / Course Coordinator",
        "evidence_question": "What specific instruction gaps are described in the evidence?"
    },
    "readings": {
        "area": "Learning materials",
        "action": "Review reading selection, relevance, workload, and accessibility.",
        "owner": "Instructor / Course Coordinator",
        "evidence_question": "Are students reporting workload, relevance, or accessibility problems with readings?"
    },
    "reading": {
        "area": "Learning materials",
        "action": "Review reading selection, relevance, workload, and accessibility.",
        "owner": "Instructor / Course Coordinator",
        "evidence_question": "What recurring issue with reading material appears in the feedback?"
    },
    "marking": {
        "area": "Assessment / grading",
        "action": "Review grading criteria, consistency, transparency, and alignment with assignment or assessment requirements.",
        "owner": "Course Coordinator / Assessment Team",
        "evidence_question": "Are students reporting concerns about grading consistency or transparency?"
    },
}

CATEGORY_RULES = {
    "Course - Workload": (
        "Review course workload and whether required activities are proportionate "
        "to available time and learning objectives.",
        "Course Coordinator"
    ),
    "Course - Assessment": (
        "Review assessment difficulty, clarity, workload, and alignment with "
        "course content and learning objectives.",
        "Course Coordinator / Assessment Team"
    ),
    "Course - Content": (
        "Review content relevance, coverage, sequencing, and alignment with "
        "learning objectives.",
        "Course Coordinator"
    ),
    "Course - Course materials": (
        "Review learning materials for clarity, completeness, relevance, and accessibility.",
        "Instructor / Course Coordinator"
    ),
    "Course - Difficulty": (
        "Review whether perceived difficulty is caused by content complexity, "
        "pace, prerequisites, workload, or assessment design.",
        "Course Coordinator"
    ),
    "Course - Learning activity": (
        "Review whether learning activities are useful, engaging, clearly explained, "
        "and appropriately connected to learning objectives.",
        "Instructor / Course Coordinator"
    ),
    "Staff - Teaching": (
        "Review teaching clarity, pacing, explanation quality, and classroom delivery.",
        "Department / Academic Coordinator"
    ),
    "Staff - Helpfulness": (
        "Review instructor availability, responsiveness, and student support.",
        "Department / Academic Coordinator"
    ),
    "Staff - Attitude": (
        "Review communication and classroom interactions for recurring student concerns.",
        "Department / Academic Coordinator"
    ),
    "Staff - Knowledge & skills": (
        "Review whether students report gaps in subject explanation or demonstrated expertise.",
        "Department / Academic Coordinator"
    ),
    "University - Information & services": (
        "Review the relevant information/service pathway for clarity, accessibility, "
        "and responsiveness.",
        "University Service Owner"
    ),
    "University - Campus & facilities": (
        "Review the specific facilities or campus services referenced in negative evidence.",
        "University Service Owner"
    ),
}


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_filename(x):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(x)).strip("_")


def recommendation_for_concept(concept):
    if concept in CONCEPT_RULES:
        rule = CONCEPT_RULES[concept]
        return (
            rule["action"],
            rule["owner"],
            rule["evidence_question"],
        )

    return (
        "Review the underlying student feedback evidence for this concern and "
        "identify the specific issue before selecting an intervention.",
        "Relevant Academic / Service Owner",
        "What specific issue is repeatedly described in the evidence?",
    )


def category_recommendation(category):
    if category in CATEGORY_RULES:
        return CATEGORY_RULES[category]
    return (
        "Inspect the category-level negative evidence and determine the specific "
        "operational issue before selecting an intervention.",
        "Relevant Category Owner"
    )


def build_insight(row, evidence_count):
    concept = norm(row["concept"])
    priority = norm(row["priority"])
    severity = norm(row["severity_level"])
    neg_rate = float(row["negative_rate"])
    affected = int(row["affected_reviews"])

    if priority == "Critical":
        opening = "This is a high-priority concern requiring administrator attention."
    elif priority == "High":
        opening = "This is a high-priority concern that should be investigated."
    elif priority == "Medium":
        opening = "This is a moderate-priority concern worth targeted review."
    else:
        opening = "This concern should be monitored rather than treated as a major issue from this evidence alone."

    return (
        f"{opening} The concept '{concept}' appears in {affected:,} affected reviews "
        f"with a {neg_rate:.1%} negative rate. Its evidence-based severity is "
        f"{severity.lower()}, supported by {evidence_count:,} linked evidence records. "
        "The result indicates where to investigate; it does not establish causation."
    )


def main():
    print("=" * 78)
    print("PHASE 16 — ACTIONABLE INSIGHT & RECOMMENDATION ENGINE")
    print("=" * 78)

    ranking_path = P15 / "severity_priority_ranking.csv"
    evidence_path = P15 / "severity_priority_evidence.csv"

    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)

    ranking = pd.read_csv(ranking_path)
    evidence = pd.read_csv(evidence_path)

    print(f"\nPhase 15 concepts: {len(ranking):,}")
    print(f"Evidence records: {len(evidence):,}")

    # Evidence count per concept.
    ev_counts = (
        evidence.groupby("concept")
        .size()
        .reset_index(name="linked_evidence_count")
    )

    insights = ranking.merge(ev_counts, on="concept", how="left")
    insights["linked_evidence_count"] = (
        insights["linked_evidence_count"].fillna(0).astype(int)
    )

    rows = []
    for _, r in insights.iterrows():
        concept = norm(r["concept"])
        action, owner, validation_question = recommendation_for_concept(concept)

        rows.append({
            "rank": int(r["rank"]),
            "concept": concept,
            "priority_score": float(r["priority_score"]),
            "severity_score": float(r["severity_score"]),
            "frequency_score": float(r["frequency_score"]),
            "priority": norm(r["priority"]),
            "severity_level": norm(r["severity_level"]),
            "negative_rate": float(r["negative_rate"]),
            "affected_reviews": int(r["affected_reviews"]),
            "linked_evidence_count": int(r["linked_evidence_count"]),
            "evidence_confidence": float(r["evidence_confidence"]),
            "insight": build_insight(r, int(r["linked_evidence_count"])),
            "recommended_action": action,
            "recommended_owner": owner,
            "validation_question": validation_question,
        })

    insight_df = pd.DataFrame(rows)

    insight_df.to_csv(
        OUT / "actionable_insights.csv", index=False
    )

    insight_df[
        insight_df["priority"].isin(["Critical", "High"])
    ].to_csv(
        OUT / "administrator_action_queue.csv", index=False
    )

    # ------------------------------------------------------------------
    # Category-backed recommendations
    # ------------------------------------------------------------------
    category_path = P13 / "category_sentiment_summary.csv"
    category_rows = []

    if category_path.exists():
        cat = pd.read_csv(category_path)

        # Detect common possible column names without changing source data.
        category_col = next(
            (c for c in ["category", "category_normalized"] if c in cat.columns),
            None
        )

        neg_col = next(
            (c for c in ["negative_rate", "neg_rate", "negative_ratio"]
             if c in cat.columns),
            None
        )

        count_col = next(
            (c for c in ["review_count", "reviews", "annotation_count",
                         "review_annotations"]
             if c in cat.columns),
            None
        )

        if category_col:
            for _, r in cat.iterrows():
                category = norm(r[category_col])
                if not category:
                    continue

                action, owner = category_recommendation(category)

                rec = {
                    "category": category,
                    "recommended_action": action,
                    "recommended_owner": owner,
                }

                if neg_col:
                    try:
                        rec["negative_rate"] = float(r[neg_col])
                    except Exception:
                        rec["negative_rate"] = np.nan

                if count_col:
                    try:
                        rec["volume"] = int(r[count_col])
                    except Exception:
                        rec["volume"] = np.nan

                category_rows.append(rec)

    category_df = pd.DataFrame(category_rows)
    if not category_df.empty:
        category_df.to_csv(
            OUT / "category_recommendations.csv", index=False
        )

    # ------------------------------------------------------------------
    # Evidence package for dashboard/API consumption
    # ------------------------------------------------------------------
    evidence_cols = [
        c for c in [
            "concept", "phase15_rank", "priority_score",
            "severity_level", "priority", "review_id",
            "text", "opinion", "category", "domain",
            "language_strength"
        ] if c in evidence.columns
    ]

    dashboard_evidence = evidence[evidence_cols].copy()

    if "concept" in dashboard_evidence.columns:
        dashboard_evidence = dashboard_evidence.merge(
            insight_df[
                ["concept", "recommended_action", "recommended_owner"]
            ],
            on="concept",
            how="left"
        )

    dashboard_evidence.to_csv(
        OUT / "dashboard_evidence_package.csv", index=False
    )

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------
    top = insight_df.head(10)

    summary = {
        "phase": "16",
        "status": "completed",
        "input": "Phase 15 severity/priority intelligence",
        "concepts_with_insights": int(len(insight_df)),
        "critical_concerns": int((insight_df["priority"] == "Critical").sum()),
        "high_concerns": int((insight_df["priority"] == "High").sum()),
        "medium_concerns": int((insight_df["priority"] == "Medium").sum()),
        "low_or_monitor_concerns": int(
            (~insight_df["priority"].isin(["Critical", "High", "Medium"])).sum()
        ),
        "methodology": [
            "Evidence-backed rule-based recommendations",
            "No causal claims",
            "No temporal/emerging claims",
            "Recommendations framed as investigation/action prompts",
        ],
        "top_10": top.to_dict("records"),
    }

    (OUT / "phase16_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    readme = """# Phase 16 — Actionable Insight & Recommendation Engine

## Objective

Convert Phase 15 concern scores into administrator-readable insights and
evidence-backed recommendations.

This phase intentionally uses transparent rules rather than an LLM so every
recommendation can be traced to a concern, metric, and evidence record.

## Outputs

- `actionable_insights.csv`
- `administrator_action_queue.csv`
- `category_recommendations.csv` (when Phase 13 category summary is available)
- `dashboard_evidence_package.csv`
- `phase16_summary.json`
- `PHASE_16_README.md`

## Recommendation philosophy

The engine does NOT claim:
- causation,
- that a recommendation will definitely solve the issue,
- that a concern is emerging/trending.

It produces investigation and improvement actions supported by observed
feedback patterns.

## Next stage

Phase 17 can build semantic retrieval over the evidence package, followed by
RAG and the administrator AI assistant.
"""

    (OUT / "PHASE_16_README.md").write_text(readme, encoding="utf-8")

    print("\n" + "=" * 78)
    print("PHASE 16 COMPLETE")
    print("=" * 78)
    print(f"Insights generated: {len(insight_df):,}")
    print(f"Critical: {(insight_df['priority'] == 'Critical').sum()}")
    print(f"High: {(insight_df['priority'] == 'High').sum()}")
    print(f"Medium: {(insight_df['priority'] == 'Medium').sum()}")

    print("\nTop actionable insights:")
    print(
        insight_df[
            [
                "rank", "concept", "priority_score", "priority",
                "negative_rate", "affected_reviews",
                "recommended_owner"
            ]
        ].head(15).to_string(index=False)
    )

    print("\nGenerated:")
    for p in sorted(OUT.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()

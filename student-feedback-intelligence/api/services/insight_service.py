from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE16 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase16"
)

PHASE15 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "edurabsa"
    / "phase15"
)


class InsightService:

    def __init__(self):

        self.insights = self._load_insights()
        self.evidence = self._load_evidence()


    def _load_insights(self):

        path = (
            PHASE16
            / "actionable_insights.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing Phase 16 artifact: {path}"
            )

        return pd.read_csv(path)


    def _load_evidence(self):

        path = (
            PHASE16
            / "dashboard_evidence_package.csv"
        )

        if not path.exists():

            return pd.DataFrame()

        return pd.read_csv(path)


    def get_all_insights(self):

        return self.insights.to_dict(
            orient="records"
        )


    def get_top_insights(
        self,
        limit=10
    ):

        df = self.insights.copy()

        if "priority_score" in df.columns:

            df = df.sort_values(
                "priority_score",
                ascending=False
            )

        return df.head(limit).to_dict(
            orient="records"
        )


    def get_insight(
        self,
        concept
    ):

        matches = self.insights[
            self.insights["concept"]
            .astype(str)
            .str.lower()
            ==
            concept.lower()
        ]

        if matches.empty:

            return None

        record = matches.iloc[0].to_dict()

        if not self.evidence.empty and "concept" in self.evidence.columns:
            ev_matches = self.evidence[
                self.evidence["concept"].astype(str).str.lower() == concept.lower()
            ]
            record["evidence"] = ev_matches.head(15).to_dict(orient="records")
        else:
            record["evidence"] = []

        return record


    def get_summary(self):

        df = self.insights

        summary = {
            "total_insights": int(len(df)),
        }

        if "priority" in df.columns:

            summary["priority_distribution"] = (
                df["priority"]
                .value_counts()
                .to_dict()
            )

        if "severity_level" in df.columns:

            summary["severity_distribution"] = (
                df["severity_level"]
                .value_counts()
                .to_dict()
            )

        if "priority_score" in df.columns:

            summary["average_priority_score"] = float(
                df["priority_score"].mean()
            )

        return summary
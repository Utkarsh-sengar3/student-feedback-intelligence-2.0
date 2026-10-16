"""Phase 6: Student Feedback Insight Engine.

Reads Phase 5 derived annotation data and produces descriptive insight tables.
Severity/priority are inherited derived heuristics, not supervised labels.
"""
from pathlib import Path
import pandas as pd
import numpy as np

def build_insights(annotation_csv: str, output_dir: str):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    ann=pd.read_csv(annotation_csv)
    ann["sentiment_norm"]=ann["sentiment_norm"].fillna("").astype(str)
    domain_reviews=ann.groupby("domain")["review_id"].nunique().to_dict()

    sent=ann.groupby(["domain","sentiment_norm"]).size().unstack(fill_value=0)
    for c in ["Negative","Neutral","Positive"]:
        if c not in sent.columns: sent[c]=0
    sent["total_annotations"]=sent[["Negative","Neutral","Positive"]].sum(axis=1)
    sent["negative_rate"]=sent["Negative"]/sent["total_annotations"]
    sent.reset_index().to_csv(out/"domain_sentiment_summary.csv",index=False)

    neg=ann[ann["sentiment_norm"]=="Negative"].copy()
    recurring=(neg.groupby(["domain","category"])
        .agg(affected_reviews=("review_id","nunique"),
             negative_mentions=("review_id","size"),
             high_severity_mentions=("severity",lambda s:(s=="High").sum()))
        .reset_index())
    recurring["domain_review_count"]=recurring["domain"].map(domain_reviews)
    recurring["review_recurrence_rate"]=recurring["affected_reviews"]/recurring["domain_review_count"]
    recurring.sort_values(["review_recurrence_rate","affected_reviews"],ascending=False)\
             .to_csv(out/"recurring_concerns.csv",index=False)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--annotations",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    build_insights(a.annotations,a.output)

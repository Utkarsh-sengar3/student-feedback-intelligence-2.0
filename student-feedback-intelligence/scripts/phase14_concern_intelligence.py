from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P13=ROOT/"data"/"processed"/"edurabsa"/"phase13"
OUT=ROOT/"data"/"processed"/"edurabsa"/"phase14"
OUT.mkdir(parents=True,exist_ok=True)

WEIGHTS={"negative_rate":0.35,"affected_review_share":0.30,"negative_volume_share":0.20,"breadth":0.15}
MIN_NEGATIVE_REVIEWS=5

def load(name):
    p=P13/name
    if not p.exists(): raise FileNotFoundError(f"Missing Phase 13 artifact: {p}")
    return pd.read_csv(p)

def minmax(s):
    s=pd.to_numeric(s,errors="coerce").fillna(0.0)
    lo,hi=s.min(),s.max()
    return pd.Series(0.0,index=s.index) if hi<=lo else (s-lo)/(hi-lo)

def severity(x):
    return "Critical" if x>=.75 else "High" if x>=.50 else "Medium" if x>=.25 else "Low"

def main():
    print("="*78); print("PHASE 14 — RECURRING + HIGH-IMPACT CONCERN INTELLIGENCE"); print("="*78)
    df=load("recurring_negative_aspects.csv")
    evidence=load("negative_aspect_evidence.csv")
    req=["aspect_normalized","negative_annotations","affected_reviews","total_annotations","negative_rate"]
    miss=[c for c in req if c not in df.columns]
    if miss: raise ValueError(f"Missing columns: {miss}")
    for c in ["negative_annotations","affected_reviews","total_annotations","negative_rate"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    df=df[df.affected_reviews>=MIN_NEGATIVE_REVIEWS].copy()

    total_reviews=max(df.affected_reviews.sum(),1)
    total_neg=max(df.negative_annotations.sum(),1)
    df["affected_review_share"]=df.affected_reviews/total_reviews
    df["negative_volume_share"]=df.negative_annotations/total_neg

    def nitems(x):
        if pd.isna(x) or not str(x).strip(): return 0
        return len([v for v in str(x).split(",") if v.strip()])
    df["category_breadth"]=df["categories"].apply(nitems) if "categories" in df else 0
    df["domain_breadth"]=df["domains"].apply(nitems) if "domains" in df else 0
    df["breadth_raw"]=df["category_breadth"].clip(upper=6)/6*.6 + df["domain_breadth"].clip(upper=3)/3*.4

    df["concern_score"]=(WEIGHTS["negative_rate"]*df.negative_rate.clip(0,1)
        +WEIGHTS["affected_review_share"]*minmax(df.affected_review_share)
        +WEIGHTS["negative_volume_share"]*minmax(df.negative_volume_share)
        +WEIGHTS["breadth"]*minmax(df.breadth_raw)).clip(0,1)
    df["severity_level"]=df.concern_score.apply(severity)

    def priority(r):
        if r.concern_score>=.75 or (r.negative_rate>=.60 and r.affected_reviews>=20): return "Critical"
        if r.concern_score>=.50 or (r.negative_rate>=.45 and r.affected_reviews>=15): return "High"
        if r.concern_score>=.25: return "Medium"
        return "Low"
    df["priority"]=df.apply(priority,axis=1)
    df=df.sort_values(["concern_score","negative_annotations","negative_rate"],ascending=False).reset_index(drop=True)
    df["rank"]=np.arange(1,len(df)+1)

    cols=["rank","aspect_normalized","concern_score","severity_level","priority","negative_rate","negative_annotations","affected_reviews","total_annotations","category_breadth","domain_breadth","categories","domains"]
    ranked=df[[c for c in cols if c in df]].copy()
    ranked.to_csv(OUT/"concern_ranking.csv",index=False)
    ranked[(ranked.affected_reviews>=10)&(ranked.negative_rate>=.30)].to_csv(OUT/"recurring_concerns.csv",index=False)
    ranked[(ranked.negative_annotations>=15)&(ranked.negative_rate>=.30)].to_csv(OUT/"high_impact_concerns.csv",index=False)
    ranked[(ranked.affected_reviews<20)&(ranked.negative_rate>=.60)].to_csv(OUT/"low_frequency_high_severity_candidates.csv",index=False)

    evcols=[c for c in ["aspect_normalized","review_id","domain","category","opinion","text"] if c in evidence]
    ev=evidence[evcols].merge(ranked[["aspect_normalized","concern_score","severity_level","priority","rank"]],on="aspect_normalized",how="inner").sort_values(["rank","review_id"])
    ev.to_csv(OUT/"concern_evidence.csv",index=False)

    dc=P13/"domain_category_sentiment.csv"
    if dc.exists():
        d=pd.read_csv(dc)
        if "sentiment_norm" in d: d[d.sentiment_norm.eq("Negative")].to_csv(OUT/"domain_category_negative_concerns.csv",index=False)

    summary={"phase":"14","status":"completed","minimum_negative_reviews":MIN_NEGATIVE_REVIEWS,
      "ranking_formula":WEIGHTS,
      "severity_thresholds":{"Critical":">=0.75","High":">=0.50","Medium":">=0.25","Low":"<0.25"},
      "ranked_concerns":len(ranked),
      "temporal_note":"No emerging/trending claims: no reliable temporal field is available.",
      "top_concerns":ranked.head(20).to_dict("records")}
    (OUT/"phase14_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    (OUT/"PHASE_14_README.md").write_text("""# Phase 14 — Concern Intelligence

Consumes Phase 13 artifacts and creates a transparent concern-ranking layer.

Score:
`0.35*negative_rate + 0.30*affected_review_share + 0.20*negative_volume_share + 0.15*breadth`

Severity: Critical >= .75, High >= .50, Medium >= .25, Low < .25.

This is a ranking signal, not a probability or causal risk score. No temporal/emerging claims are made.

Run:
`..\\.venv\\Scripts\\python.exe -u scripts/phase14_concern_intelligence.py`
""",encoding="utf-8")

    print("\n"+"="*78); print("PHASE 14 COMPLETE"); print("="*78)
    print(f"Ranked concerns: {len(ranked):,}")
    print(f"Output: {OUT}")
    if len(ranked): print(ranked[["rank","aspect_normalized","concern_score","severity_level","priority","negative_rate","affected_reviews"]].head(15).to_string(index=False))
    print("\nGenerated:")
    for p in sorted(OUT.iterdir()): print(" -",p.name)

if __name__=="__main__": main()

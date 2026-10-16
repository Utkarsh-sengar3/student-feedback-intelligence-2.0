from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
PHASE10 = ROOT/"data"/"processed"/"edurabsa"/"phase10"
OUT = ROOT/"data"/"processed"/"edurabsa"/"phase10_1"
OUT.mkdir(parents=True, exist_ok=True)

pred_path = PHASE10/"final_test_predictions.csv"
summary_path = PHASE10/"phase10_summary.json"
labels = ["Neg","Neu","Pos"]

if not pred_path.exists():
    raise FileNotFoundError(f"Missing Phase 10 predictions: {pred_path}")

df = pd.read_csv(pred_path)
required = {"overall_sentiment","predicted_sentiment"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df = df[df.overall_sentiment.isin(labels) & df.predicted_sentiment.isin(labels)].copy()
df["correct"] = df.overall_sentiment == df.predicted_sentiment
errors = df[~df.correct].copy()
errors["error_type"] = errors.overall_sentiment + "_to_" + errors.predicted_sentiment

# Confusion matrix
cm = confusion_matrix(df.overall_sentiment, df.predicted_sentiment, labels=labels)
pd.DataFrame(cm,index=labels,columns=labels).to_csv(OUT/"confusion_matrix.csv")

# Class metrics
p,r,f,s = precision_recall_fscore_support(
    df.overall_sentiment, df.predicted_sentiment,
    labels=labels, zero_division=0
)
class_df = pd.DataFrame({
    "sentiment":labels,"precision":p,"recall":r,"f1":f,"samples":s
})
class_df["errors"] = class_df["samples"] - class_df["recall"]*class_df["samples"]
class_df["errors"] = class_df["errors"].round().astype(int)
class_df["error_rate"] = 1-class_df["recall"]
class_df.to_csv(OUT/"class_error_rates.csv",index=False)

# Error types
etype = errors.error_type.value_counts().rename_axis("error_type").reset_index(name="count")
etype["percentage_of_errors"] = etype["count"]/max(len(errors),1)*100
etype.to_csv(OUT/"error_types.csv",index=False)

# Error records
cols = [c for c in ["review_id","original_id","domain","text","text_clean",
                    "overall_sentiment","predicted_sentiment","error_type"] if c in errors.columns]
errors[cols].to_csv(OUT/"all_misclassifications.csv",index=False)

neutral = errors[(errors.overall_sentiment=="Neu") | (errors.predicted_sentiment=="Neu")]
neutral[cols].to_csv(OUT/"neutral_misclassifications.csv",index=False)

# Domain performance
domain_df = pd.DataFrame()
if "domain" in df.columns:
    rows=[]
    for domain,g in df.groupby("domain",dropna=False):
        pp,rr,ff,_ = precision_recall_fscore_support(
            g.overall_sentiment,g.predicted_sentiment,
            labels=labels,zero_division=0
        )
        rows.append({
            "domain":str(domain),"samples":len(g),
            "accuracy":accuracy_score(g.overall_sentiment,g.predicted_sentiment),
            "macro_precision":np.mean(pp),
            "macro_recall":np.mean(rr),
            "macro_f1":np.mean(ff),
            "errors":int((~g.correct).sum()),
            "error_rate":float((~g.correct).mean())
        })
    domain_df=pd.DataFrame(rows).sort_values("macro_f1")
domain_df.to_csv(OUT/"domain_performance.csv",index=False)

errors.head(200).to_csv(OUT/"high_priority_error_review_samples.csv",index=False)

summary={}
if summary_path.exists():
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
test_metrics=summary.get("final_test",{})
macro_f1=test_metrics.get("f1_macro")

result={
    "phase":"10.1","status":"completed",
    "test_rows_analyzed":int(len(df)),
    "total_errors":int(len(errors)),
    "accuracy":float(df.correct.mean()),
    "phase10_macro_f1":macro_f1,
    "phase9_3_macro_f1":0.7049,
    "macro_f1_improvement_vs_phase9_3":None if macro_f1 is None else float(macro_f1-0.7049),
    "most_common_error_type":etype.iloc[0].to_dict() if len(etype) else None,
    "highest_error_class":class_df.sort_values("error_rate",ascending=False).iloc[0].to_dict(),
    "weakest_domain":domain_df.iloc[0].to_dict() if len(domain_df) else None,
    "neutral_related_error_count":int(len(neutral))
}
(OUT/"phase10_1_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")

report = f"""# Phase 10.1 — Transformer Error & Domain Analysis

Post-hoc analysis only. No model retraining.

## Overall
- Test rows analyzed: {len(df)}
- Errors: {len(errors)}
- Accuracy: {df.correct.mean():.6f}
- Phase 10 Macro F1: {macro_f1}
- Phase 9.3 Macro F1: 0.704900
- Neutral-related errors: {len(neutral)}

## Files
- confusion_matrix.csv
- class_error_rates.csv
- error_types.csv
- domain_performance.csv
- all_misclassifications.csv
- neutral_misclassifications.csv
- high_priority_error_review_samples.csv
- phase10_1_summary.json

Note: Phase 10 predictions contain labels, not probabilities, so this phase does not invent confidence scores.
"""
(OUT/"PHASE_10_1_README.md").write_text(report,encoding="utf-8")

print("PHASE 10.1 COMPLETE")
print("Rows:",len(df),"Errors:",len(errors),"Accuracy:",round(df.correct.mean(),6))
print("\nConfusion Matrix:\n",pd.DataFrame(cm,index=labels,columns=labels))
print("\nClass Metrics:\n",class_df[["sentiment","samples","errors","error_rate","f1"]].to_string(index=False))
print("\nDomain Performance:\n",domain_df.to_string(index=False))
print("\nError Types:\n",etype.to_string(index=False))
print("\nOutput:",OUT)

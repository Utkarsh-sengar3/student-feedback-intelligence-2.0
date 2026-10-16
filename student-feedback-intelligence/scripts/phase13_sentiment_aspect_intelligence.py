from pathlib import Path
import json, re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT/'data/processed/edurabsa/annotations_canonical.csv'
R = ROOT/'data/processed/edurabsa/reviews_canonical.csv'
OUT = ROOT/'data/processed/edurabsa/phase13'; OUT.mkdir(parents=True, exist_ok=True)

def clean(v): return '' if pd.isna(v) else str(v).strip()
def sent(v):
    return {'negative':'Negative','neg':'Negative','neutral':'Neutral','neu':'Neutral','positive':'Positive','pos':'Positive'}.get(clean(v).lower())
def aspect(v):
    x=clean(v).lower()
    if not x or x=='null' or x in {'he','she','her','his','him','they','them','their','hers','this','that'}: return None
    x=re.sub(r'\s+',' ',x)
    mp={'prof.':'professor','prof':'professor','this course':'course','this class':'class','classes':'class','assignments':'assignment','lectures':'lecture','materials':'course materials','material':'course materials'}
    return mp.get(x,x)

def req(df, cols, name):
    m=[c for c in cols if c not in df.columns]
    if m: raise ValueError(f'{name} missing {m}; available={list(df.columns)}')

def main():
    print('='*78); print('PHASE 13 — SENTIMENT × ASPECT INTELLIGENCE'); print('='*78)
    ann=pd.read_csv(A); rev=pd.read_csv(R)
    req(ann,['review_id','aspect','opinion','category','sentiment'],'annotations_canonical.csv')
    req(rev,['review_id','text','domain','overall_sentiment'],'reviews_canonical.csv')
    ann['review_id']=ann['review_id'].astype(str); rev['review_id']=rev['review_id'].astype(str)
    meta=rev[['review_id','text','domain','overall_sentiment']].drop_duplicates('review_id')
    df=ann.merge(meta,on='review_id',how='left',validate='many_to_one',suffixes=('_annotation','_review'))
    # Handles both possible source schemas and prevents the old KeyError: domain.
    if 'domain' not in df.columns:
        for c in ('domain_review','domain_annotation'):
            if c in df.columns: df['domain']=df[c]; break
    if 'text' not in df.columns and 'text_review' in df.columns: df['text']=df['text_review']
    if 'overall_sentiment' not in df.columns and 'overall_sentiment_review' in df.columns: df['overall_sentiment']=df['overall_sentiment_review']
    req(df,['domain','text','overall_sentiment'],'merged Phase 13 dataframe')
    print(f'\nAnnotation rows: {len(df):,}')
    print(f'Unique annotated reviews: {df.review_id.nunique():,}')
    print(f'Domain missing after join: {df.domain.isna().sum():,}')
    df['aspect_original']=df.aspect.map(clean); df['aspect_normalized']=df.aspect.map(aspect)
    df['sentiment_norm']=df.sentiment.map(sent); df['overall_sentiment_norm']=df.overall_sentiment.map(sent)
    df.to_csv(OUT/'sentiment_aspect_annotations.csv',index=False)
    ad=df[df.aspect_normalized.notna()].copy(); cd=df[df.category.notna()].copy()
    def summary(g, keys, name):
        x=g.groupby(keys+['sentiment_norm'],dropna=False).agg(annotation_count=('review_id','size'),review_count=('review_id','nunique')).reset_index()
        x.to_csv(OUT/name,index=False); return x
    summary(ad,['aspect_normalized'],'aspect_sentiment_summary.csv')
    summary(cd,['category'],'category_sentiment_summary.csv')
    ad.groupby(['aspect_normalized','category','sentiment_norm']).agg(annotation_count=('review_id','size'),review_count=('review_id','nunique')).reset_index().to_csv(OUT/'aspect_category_matrix.csv',index=False)
    cd[cd.domain.notna()].groupby(['domain','category','sentiment_norm']).agg(annotation_count=('review_id','size'),review_count=('review_id','nunique')).reset_index().to_csv(OUT/'domain_category_sentiment.csv',index=False)
    neg=ad[ad.sentiment_norm.eq('Negative')]
    rec=neg.groupby('aspect_normalized').agg(negative_annotations=('review_id','size'),affected_reviews=('review_id','nunique')).reset_index()
    totals=ad.groupby('aspect_normalized').size().rename('total_annotations').reset_index(); rec=rec.merge(totals,on='aspect_normalized'); rec['negative_rate']=rec.negative_annotations/rec.total_annotations; rec.sort_values(['negative_rate','negative_annotations'],ascending=False).to_csv(OUT/'recurring_negative_aspects.csv',index=False)
    ev=[]
    for a,g in neg.groupby('aspect_normalized'):
        for _,r in g[['review_id','domain','category','opinion','text']].drop_duplicates('review_id').head(10).iterrows(): ev.append({'aspect_normalized':a,**r.to_dict()})
    pd.DataFrame(ev).to_csv(OUT/'negative_aspect_evidence.csv',index=False)
    df[df.sentiment_norm.notna()&df.overall_sentiment_norm.notna()].groupby(['overall_sentiment_norm','sentiment_norm']).agg(annotation_count=('review_id','size'),review_count=('review_id','nunique')).reset_index().to_csv(OUT/'overall_vs_aspect_sentiment.csv',index=False)
    hi=ad.groupby('aspect_normalized').agg(total_annotations=('review_id','size'),affected_reviews=('review_id','nunique'),negative_annotations=('sentiment_norm',lambda s:int((s=='Negative').sum()))).reset_index(); hi['negative_rate']=hi.negative_annotations/hi.total_annotations; hi['impact_signal']=hi.negative_annotations*hi.negative_rate; hi.sort_values('impact_signal',ascending=False).to_csv(OUT/'high_impact_concern_candidates.csv',index=False)
    s={'phase':'13','status':'completed','annotation_rows':len(df),'unique_reviews':int(df.review_id.nunique()),'unique_domains':int(df.domain.dropna().nunique()),'unique_categories':int(df.category.dropna().nunique()),'unique_normalized_aspects':int(df.aspect_normalized.dropna().nunique()),'missing_domain_rows':int(df.domain.isna().sum())}
    (OUT/'phase13_summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    (OUT/'PHASE_13_README.md').write_text('# Phase 13 — Sentiment × Aspect Intelligence\n\nFixed review metadata merge; domain is explicitly recovered and validated after merge. No retraining is performed.\n',encoding='utf-8')
    print('\nPHASE 13 COMPLETE'); print(f'Output: {OUT}')
    print('Generated:'); [print(' -',p.name) for p in sorted(OUT.iterdir())]

if __name__=='__main__': main()

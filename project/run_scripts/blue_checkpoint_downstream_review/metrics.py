"""Independent count-based metrics from stored predictions; no model imports."""
from collections import Counter
import math

TASKS=('sst2','mrpc','cola','rte','mmlu','nli')

def scores(gold,pred):
    if len(gold)!=len(pred) or not gold:raise ValueError('DENOMINATOR')
    classes=sorted(set(gold)|set(pred));n=len(gold);correct=sum(a==b for a,b in zip(gold,pred))
    per=[];g=Counter(gold);p=Counter(pred)
    for k in classes:
        tp=sum(a==b==k for a,b in zip(gold,pred));fp=p[k]-tp;fn=g[k]-tp
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        per.append(dict(label=k,support=g[k],predicted=p[k],tp=tp,fp=fp,fn=fn,f1=f1))
    den=(n*n-sum(v*v for v in g.values()))*(n*n-sum(v*v for v in p.values()))
    return dict(correct=correct,total=n,invalid=sum(v==-1 for v in pred),accuracy=correct/n,
        weighted_f1=sum(r['support']*r['f1'] for r in per)/n,
        mcc=(correct*n-sum(g[k]*p[k] for k in classes))/math.sqrt(den) if den else 0.,classes=per)

def labels_and_predictions(task,rows,data):
    if len(rows)!=len(data):raise ValueError('RAW_DATA_DENOMINATOR')
    gold=[];generation=[];alternative=[];original_generation=[];original_alternative=[];ties=0
    for row,datum in zip(rows,data,strict=True):
        for k in (('question',) if task=='mmlu' else ('sentence',) if task in ('sst2','cola') else ('sentence1','sentence2')):
            if row['sentence' if k=='question' else k]!=datum[k]:raise ValueError('RAW_DATA_ORDER_OR_TEXT')
        y=datum['answer'] if task=='mmlu' else (int(datum['label']=='entailment') if task=='nli' else datum['label'])
        if task=='mmlu':
            pp=[row['prob_'+k] for k in 'abcd'];maximum=max(pp)
            q=pp.index(maximum) if pp.count(maximum)==1 else -1;ties+=pp.count(maximum)>1
            literal='ABCD'[q] if q!=-1 else None
        else:
            positive,negative=('positive','negative') if task=='sst2' else ('true','false') if task=='nli' else ('yes','no')
            pp=[row['prob_'+positive],row['prob_'+negative]];q=int(pp[0]>pp[1]);ties+=pp[0]==pp[1]
            literal=(('positive','negative') if task=='sst2' else ('True','False') if task in ('rte','nli') else ('Yes','No'))[0 if q else 1]
        if not all(math.isfinite(x) and 0<=x<=1 for x in pp):raise ValueError('NONFINITE_PROBABILITY')
        if row['highest_probability_answer']!=literal:raise ValueError('ALTERNATIVE_LITERAL_PROBABILITY')
        a=row['answer']
        if type(a) is not int or a not in (-1,0,1,2,3) or (task!='mmlu' and a not in (-1,0,1)):raise ValueError('GENERATION_LABEL')
        if row['correct']!=(a==y) or row['correct_new']!=(q==y):raise ValueError('ORIGINAL_CORRECT_FLAG')
        original_generation.append(a);original_alternative.append(q)
        # RTE correction changes only scoring coordinates; invalid stays -1.
        if task=='rte':a={1:0,0:1}.get(a,-1);q={1:0,0:1}.get(q,-1)
        gold.append(y);generation.append(a);alternative.append(q)
    return dict(gold=gold,generation=generation,alternative=alternative,
        original_generation=original_generation,original_alternative=original_alternative,probability_ties=ties)

def transitions(gold,before,after):
    if not len(gold)==len(before)==len(after):raise ValueError('PAIR_DENOMINATOR')
    return dict(denominator=len(gold),W0_correct=sum(g==p for g,p in zip(gold,before)),
        endpoint_correct=sum(g==p for g,p in zip(gold,after)),
        lost=sum(b==g and a!=g for g,b,a in zip(gold,before,after)),
        gained=sum(b!=g and a==g for g,b,a in zip(gold,before,after)),
        still_correct=sum(b==g==a for g,b,a in zip(gold,before,after)),
        still_wrong=sum(b!=g and a!=g for g,b,a in zip(gold,before,after)))

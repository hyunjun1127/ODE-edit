"""Metadata-only panels; no evaluator output can enter selection."""
import math
from .records import digest

SALT='ODEEDIT-S06-SINGLE-LAYER-CUMULATIVE-RISK-ABC-SH1-V1'
ENTRIES={'Early':(10,1000),'Middle':(50,5000),'Late':(90,9000)}

def ranked(values,label,key=lambda x:x):
    return sorted(values,key=lambda x:(digest([SALT,label,key(x)]),str(key(x))))

def curve_rows(rows,panel):
    result=[r for r in rows if r['metric']=='RS' or
            (r['metric']=='PS' and r['panel']=='Current100') or
            (r['metric']=='NS' and r['prompt_index'] in panel['neighbors'][str(r['case_id'])])]
    assert len(result)==1100
    return result

def select(records,entry):
    batch,start=ENTRIES[entry]
    assert len(records)==10000 and len({r['case_id'] for r in records})==10000
    prior=list(range(100,start));past=[]
    for q in range(4):
        group=prior[len(prior)*q//4:len(prior)*(q+1)//4]
        past.extend(ranked(group,entry+'-past',lambda i:records[i]['case_id'])[:25])
    ids=dict(Current100=list(range(start,start+100)),Fixed100=list(range(100)),Past100=sorted(past))
    assert all(len(v)==100 for v in ids.values())
    assert len(set(sum(ids.values(),[])))==300
    neighbor={str(records[i]['case_id']):sorted(ranked(list(range(10)),'neighbor-'+str(records[i]['case_id']))[:2]) for i in sum(ids.values(),[])}
    generated=ranked(ids['Current100'],'generation',lambda i:records[i]['case_id'])[:20]
    return dict(entry=entry,batch=batch,panels=ids,neighbors=neighbor,generation=generated,salt=SALT,
                record_hashes={str(i):digest(records[i]) for i in sum(ids.values(),[])})

def writer_plan():
    result=[dict(entry='Middle',arm='N',alpha=None)]
    for arm in ['B','C']:
        result.extend(dict(entry='Middle',arm=arm,alpha=a) for a in [.02,.06,.2])
    for entry in ['Early','Late']:
        result.extend(dict(entry=entry,arm=arm,alpha=None if arm=='N' else 'TRAIN_SELECTED') for arm in ['N','B','C'])
    return result

def choose_alpha(candidate_trajectories,endpoint_receipts):
    """Select post-update W29..W32 objectives, with verified finite W32 receipt.

    W29..W31 objective/gradient evaluation is reused by the next update;
    W32 has the mandatory terminal objective observation.
    """
    scores={}
    for alpha,rows in candidate_trajectories.items():
        selected=[r['objective'] for r in rows if r['step'] in [29,30,31,32]]
        endpoint=endpoint_receipts.get(alpha,{})
        if (len(selected)==4 and all(math.isfinite(v) for v in selected)
                and endpoint.get('completed_steps')==32 and endpoint.get('finite') is True
                and endpoint.get('endpoint_sha256') and endpoint.get('endpoint_exists_verified') is True):
            scores[alpha]=sum(selected)/4
    if not scores:raise RuntimeError('NO_FINITE_TRAIN_CANDIDATE')
    return min(scores,key=lambda a:(scores[a],a)),scores

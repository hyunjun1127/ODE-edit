"""Independent CPU NLL reducer. No runtime/controller/evaluator imports."""
import csv, hashlib, json, math
from pathlib import Path
ROOT=Path('/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2')
LOCAL=ROOT/'completed-review-20260918-v1'
WT=LOCAL/'worktree'
REPORT=WT/'experiment-reports/servers/server4/sequential-local-z-allocation-seq1000-2026-09-17-v2/completed-review-20260918-v1'
ARMS=['N4','F48','G48','C4','C48','C45678']
MULT={'RS':1,'PS':2,'NS':10}
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def writejson(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
    p=REPORT/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def success(r,tag):return r['true_nll']<r['new_nll'] if tag=='NS' else r['new_nll']<r['true_nll']
def expected(records,tag):
    out=[]
    for r in records:
        rw=r['requested_rewrite'];ps=[rw['prompt'].format(rw['subject'])] if tag=='RS' else r['paraphrase_prompts' if tag=='PS' else 'neighborhood_prompts']
        assert len(ps)==MULT[tag]
        out.extend((r['case_id'],i,digest([r['case_id'],i,p,rw['target_new']['str'],rw['target_true']['str']])) for i,p in enumerate(ps))
    return out
def validate(doc,records):
    assert doc['requests']==len(records)
    assert doc['request_order']==digest([r['case_id'] for r in records])
    result={}
    for tag,m in MULT.items():
        v=doc['metrics'][tag];rs=v['rows'];ids=[(r['case_id'],r['prompt_index'],r['identity']) for r in rs]
        assert ids==expected(records,tag) and len(set(ids))==len(rs)
        assert all(math.isfinite(r[k]) for r in rs for k in ['new_nll','true_nll'])
        bits=[success(r,tag) for r in rs]
        assert bits==[r['success'] for r in rs]
        assert v['denominator']==len(rs) and v['numerator']==sum(bits)
        if 'rate' in v:assert v['rate']==sum(bits)/len(rs)
        result[tag]={'count':sum(bits),'denominator':len(rs),'percent':100*sum(bits)/len(rs),'ties':sum(r['new_nll']==r['true_nll'] for r in rs)}
    return result
def first_table():
    records=load('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')[:1000]
    rows=[];inputs=[];ref=None
    for arm in ARMS:
        out=ROOT/'arms'/arm/'attempt-v1/output';t=load(out/'terminal.json');assert t['status']=='COMPLETED_1000_REQUESTS' and t['requests']==1000 and t['batches']==10
        p=out/'B010/seen-full.json';d=load(p);v=validate(d,records)
        if ref is None:ref=v
        row={'arm':arm,'status':'TERMINAL_AND_NLL_VERIFIED_STATE_AUDIT_PENDING'}
        for tag,r in v.items():
            row.update({tag+'_'+k:val for k,val in r.items()});row[tag+'_delta_N4_pp']=r['percent']-ref[tag]['percent']
        rows.append(row);inputs.append({'arm':arm,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'request_order':d['request_order']})
    csvout('first-final-table.csv',rows);writejson(LOCAL/'first-table-inputs.json',inputs)
    print(json.dumps(rows,indent=2));print('FIRST_TABLE_SHA',sha(REPORT/'first-final-table.csv'))
if __name__=='__main__':first_table()

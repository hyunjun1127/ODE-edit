"""Afterany CPU-only independent count/ledger collector (never loads model)."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import hashlib

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checks(rows):
    result={}
    assert len({r['identity'] for r in rows})==len(rows)
    for kind in ('R','P','N'):
        group=[r for r in rows if r['kind']==kind]
        if not group:continue
        desired='true' if kind=='N' else 'new'
        for r in group:
            assert all(math.isfinite(r[k]) for k in ('new_nll','true_nll'))
            for t in ('new','true'):
                assert 0<=r[t+'_token_correct']<=r[t+'_token_count'] and r[t+'_token_count']>0
                assert r[t+'_strict']==(r[t+'_token_correct']==r[t+'_token_count'])
        success=sum((r['true_nll']<r['new_nll']) if kind=='N' else (r['new_nll']<r['true_nll']) for r in group)
        result[kind]={'n':success,'d':len(group),'new_nll':sum(r['new_nll'] for r in group)/len(group),
                      'true_nll':sum(r['true_nll'] for r in group)/len(group),
                      'desired_token_n':sum(r[desired+'_token_correct'] for r in group),
                      'desired_token_d':sum(r[desired+'_token_count'] for r in group),
                      'desired_strict_n':sum(r[desired+'_strict'] for r in group),
                      'desired_strict_d':len(group),
                      'desired_prompt_macro':sum(r[desired+'_token_correct']/r[desired+'_token_count'] for r in group)/len(group)}
    return result

def transitions(before,after,label):
    b={r['identity']:r for r in before};a={r['identity']:r for r in after};result=[]
    assert set(b)<=set(a)
    for kind in ('R','P','N'):
        for split in ('all','active','superseded'):
            keys=[k for k,r in b.items() if r['kind']==kind and (split=='all' or a[k]['active_at_endpoint']==(split=='active'))]
            def ok(r):return r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll']
            lost=sum(ok(b[k]) and not ok(a[k]) for k in keys);gained=sum(not ok(b[k]) and ok(a[k]) for k in keys)
            original=sum(ok(b[k]) for k in keys)
            result.append(dict(comparison=label,kind=kind,split=split,denominator=len(keys),
                               before_success=original,lost=lost,gained=gained,retained=original-lost))
    return result

def save_csv(path,rows):
    if not rows:return
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args()
    out=a.run/'output';dest=a.run/'collection';dest.mkdir(exist_ok=False)
    report={'status':'INCOMPLETE','postrun_independent_reviewer':'NOT_YET_USER_RECALL',
            'checkpoint_saved':False,'exact_resume':'NOT_AVAILABLE','science_synthesis':'GH_OWNED'}
    tables=[];retention=[];inputs=[]
    try:
        s=subprocess.run(['sacct','-X','-j',a.job,'--parsable2','--noheader','--format=JobID,User,State,ExitCode,ElapsedRaw,AllocTRES,MaxRSS'],text=True,capture_output=True)
        report['scheduler']={'returncode':s.returncode,'stdout':s.stdout,'stderr':s.stderr}
        ledger=json.loads((out/'ledger.json').read_text()) if (out/'ledger.json').exists() else []
        terminal=json.loads((out/'terminal.json').read_text()) if (out/'terminal.json').exists() else {'status':'MISSING_TERMINAL'}
        report['terminal']=terminal
        assert terminal.get('commits')==len(ledger),'DISK_RAM_COMMIT_AMBIGUITY'
        for item in ledger:
            assert not (out/f"B{item['batch']:03d}-failure.json").exists(),'FAILED_CANDIDATE_IN_LEDGER'
        report['commits']=len(ledger)
        report['calls']=sum(r['solver']['calls'] for r in ledger)
        report['history_appends']=sum(r['history_appends'] for r in ledger)
        report['links']=sum(ledger[i]['entry']==ledger[i-1]['post'] for i in range(1,len(ledger)))
        assert report['calls']<=1200 and all(r['solver']['calls']<=120 for r in ledger)
        assert all(r['commit'] and r['history_appends']==5 for r in ledger)
        assert report['links']==max(0,len(ledger)-1)
        observations={}
        for endpoint in [0]+[r['batch'] for r in ledger]:
            path=out/f'W{endpoint:02d}-observations.json'
            obs=json.loads(path.read_text());inputs.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
            observations[endpoint]=obs['rows']
            reduced=checks(obs['rows'])
            expected={'R':1000,'P':2000,'N':10000} if endpoint==0 else {'R':endpoint*100,'P':endpoint*200,'N':endpoint*1000 if endpoint in (5,10) else 1000}
            assert {k:v['d'] for k,v in reduced.items()}==expected,'ALL_SEEN_DENOMINATOR'
            for k,v in reduced.items():
                stored=obs['summary'][k]
                assert (v['n'],v['d'])==(stored['numerator'],stored['denominator'])
                for key,skey in [('new_nll','new_nll_mean'),('true_nll','true_nll_mean'),
                        ('desired_token_n','desired_token_correct'),('desired_token_d','desired_token_count'),
                        ('desired_strict_n','desired_strict_numerator'),('desired_strict_d','desired_strict_denominator'),
                        ('desired_prompt_macro','desired_prompt_macro')]:
                    assert abs(v[key]-stored[skey])<=1e-12,(key,skey)
                panel='W0_first1000' if endpoint==0 else 'current100_only' if k=='N' and endpoint not in (5,10) else 'all_seen'
                tables.append({'endpoint':endpoint,'panel':panel,'kind':k,**v})
            if endpoint:
                item=ledger[endpoint-1];assert sha(path)==item['observation']['sha256']
                assert obs['state']==item['post']
                current=checks([r for r in obs['rows'] if r['case_id'] in set(item['case_ids'])])
                assert {k:v['d'] for k,v in current.items()}=={'R':100,'P':200,'N':1000}
                for k,v in current.items():tables.append({'endpoint':endpoint,'panel':'current100','kind':k,**v})
        if len(ledger)==10 and terminal['status']=='TEN_BATCHES_COMMITTED_EVALUATED':
            assert {k:v['d'] for k,v in checks(observations[10]).items()}=={'R':1000,'P':2000,'N':10000}
            atwrite=[r for item in ledger for r in observations[item['batch']] if r['case_id'] in set(item['case_ids'])]
            retention+=transitions(atwrite,observations[10],'atwrite_to_W10')
            retention+=transitions(observations[5],observations[10],'W5_first500_to_W10_same500')
            retention+=transitions([r for r in observations[0] if r['kind']=='N'],observations[10],'W0_neighborhood_to_W10')
            report['status']='TEN_BATCH_CPU_REDUCTION_PASS'
        else:report['status']='PARTIAL_OR_TECHNICAL_FAILURE'
    except Exception as exc:
        import traceback
        report.update(status='COLLECTOR_VALIDATION_FAILED',error=str(exc),traceback=traceback.format_exc())
    save_csv(dest/'endpoint-metrics.csv',tables);save_csv(dest/'retention.csv',retention)
    (dest/'collection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (dest/'input-manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (dest/'factual-report-ko.md').write_text('# JLZ B100×10 자동 사실 수집\n\n'
        +'상태: `'+report['status']+'`. 이는 독립 reviewer의 사후 검토를 대신하지 않습니다.\n\n'
        +'commit 수: '+str(report.get('commits',0))+'; whole-batch oracle 수: '+str(report.get('calls',0))
        +'. checkpoint 미저장, exact resume 불가. 전체 원자료는 local에 보존합니다.\n\n'
        +'[지표](endpoint-metrics.csv), [전이](retention.csv), [receipt](collection.json).\n')
    manifest=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(dest.iterdir()) if p.is_file()]
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return 0 if report['status'] in ('TEN_BATCH_CPU_REDUCTION_PASS','PARTIAL_OR_TECHNICAL_FAILURE') else 2

if __name__=='__main__':raise SystemExit(main())

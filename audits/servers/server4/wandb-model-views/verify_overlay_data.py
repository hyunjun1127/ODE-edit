"""Verify existing and new remote curves share key/unit/scope, not runtime claims."""
import json
import math
from pathlib import Path
import wandb

def main():
    api=wandb.Api(timeout=30);root='wkdguswns2256/layer allocation/'
    ids=['price-59768-metrics','historical-memit-first2k','historical-memith-first2k',
         'historical-alphaedit-first2k','comparison-c5b6f44631f44f22','comparison-ab6e4d2618154482']
    coverage={}
    for rid in ids:
        run=api.run(root+rid);current=[];allseen=[]
        for row in run.scan_history():
            if row.get('edits') is None or row['edits']>2000:continue
            for prefix,steps in [('current/post',current),('all_seen/post',allseen)]:
                if not all(row.get(prefix+'/'+k+'/success_pct') is not None for k in 'RPN'):continue
                rates=[row[prefix+'/'+k+'/success_pct'] for k in 'RPN']
                for k,m in [('R',1),('P',2),('N',10)]:
                    den=row[prefix+'/'+k+'/count'];num=row[prefix+'/'+k+'/success_count']
                    assert den==m*(100 if prefix=='current/post' else row['edits']),(rid,prefix,k,row['edits'],den,row.get('progress/batch'))
                    assert math.isclose(row[prefix+'/'+k+'/success_pct'],100*num/den,abs_tol=1e-9)
                score=3/sum(1/r for r in rates) if all(rates) else 0.
                assert math.isclose(row[prefix+'/success_harmonic_pct'],score,abs_tol=1e-9)
                steps.append(int(row['edits']))
        coverage[rid]=dict(current_edits=sorted(set(current)),all_seen_edits=sorted(set(allseen)))
    common=set(coverage[ids[0]]['current_edits'])
    for rid in ids[1:5]:common&=set(coverage[rid]['current_edits'])
    assert {100,200,300,400}<=common
    local=Path('/data/janghj/ODE-edit/local/wandb-comparison-bridge')
    status=json.loads((local/'status.json').read_text());launch=json.loads((local/'launch.json').read_text())
    assert status['watch'] and not status['blocked']
    receipt=dict(status='LIVE_BRIDGE_AND_REMOTE_OVERLAY_VERIFIED',coverage=coverage,
        llama_common_current_edits=sorted(common),bridge_PID=launch['pid'],bridge_source=launch['source_commit'],
        interval_seconds=60,GPU=0,original_jobs_modified=False,extra_model_evaluations=0,
        runtime_identity_claim='Historical reference; metric scope/unit aligned, not identical runtime')
    with open('audits/servers/server4/wandb-model-views/live-overlay-verification.json','x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':main()

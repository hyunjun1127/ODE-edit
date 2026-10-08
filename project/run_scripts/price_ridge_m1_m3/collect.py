"""One CPU terminal/prefix reducer, not a retry or monitoring loop."""
import argparse
import json
from pathlib import Path
from .run import TASK
from .generation_schedule import generation_due


def collect(attempt):
    from project.run_scripts.jlz_interference_l1.cap_common import rows_from,validate_rows,verify,member
    c=json.loads((attempt/'config.json').read_text());result={}
    for cell,config in c['cells'].items():
        root=attempt/cell;previous=config['cold_W0_H0'];commits=[];issues=[];generation_endpoints=[]
        identities=json.loads(verify(config['observer_identity']).read_text())['rows']
        try:
            for b in range(1,21):
                folder=root/f'batch-{b:02d}'
                if not (folder/'commit.json').exists():break
                r=json.loads((folder/'commit.json').read_text())
                assert r['before']==previous and r['batch']==b and r['fit_count']==1
                assert r['history_appends']==len(config['arm_profiles'][cell]['eligible_layers'])
                selected=[i for p in config['packs'][:b] for i in p['ids']] if b in (5,10,15,20) else config['packs'][b-1]['ids']
                rows=rows_from(folder/'post',r['after'])
                assert validate_rows(rows,identities,selected,'W'+str(b))==r['post']
                assert r['RNG_before']==r['RNG_after']
                if commits:assert r['RNG_before']==commits[-1]['RNG_after']
                if generation_due(config,b):
                    from project.run_scripts.experiment_generation_eval.observer import read_observed
                    ref=json.loads((folder/'post/generation-reference.json').read_text())
                    g=read_observed(verify(ref['member']))
                    assert g['summary']==ref['summary']
                    assert len(g['rows'])==len(selected)
                    generation_endpoints.append(b)
                else:
                    skipped=json.loads((folder/'post/generation-skipped.json').read_text())
                    assert skipped==dict(schedule='W20_ONLY',batch=b,status='NOT_SCHEDULED',
                        new_generation_forwards=0,metric_values_omitted=True)
                    assert not (folder/'post/generation-reference.json').exists()
                commits.append(r);previous=r['after']
        except Exception as e:issues.append(dict(error_type=type(e).__name__,code=str(e)))
        terminal=json.loads((root/'terminal.json').read_text()) if (root/'terminal.json').exists() else None
        result[cell]=dict(verified_commits=len(commits),verified_edits=100*len(commits),
            verified_joins=max(0,len(commits)-1),verified_H=sum(x['history_appends'] for x in commits),
            terminal=terminal,issues=issues,last_post=None if not commits else commits[-1]['post'],
            generation_verified_endpoints=generation_endpoints,
            W20_complete=len(commits)==20 and not issues and terminal is not None and terminal['status']=='W20_COMPLETE')
    payload=dict(task=TASK,cells=result,blocked_cells=c['blocked_cells'],raw_reduced=True,
                 new_forward=0,new_W0=0,automatic_retry=False)
    with (attempt/'collection.json').open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    return {k:{f:v[f] for f in ('verified_commits','verified_edits','W20_complete')} for k,v in result.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    print(json.dumps(collect(p.parse_args().attempt)))

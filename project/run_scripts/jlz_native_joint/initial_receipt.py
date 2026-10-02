"""One-shot CPU verification of the authorized representative MAIN boundary."""
import argparse
import json
from pathlib import Path
from .common import member,write,require,sha


def load(path):
    return json.loads(path.read_text())


def inspect(root,arm):
    base=root/'output'/('main-'+arm)
    paths={name:base/rel for name,rel in {
        'commit':'B001/commit.json','writer':'B001/writer.json','complete':'B001/complete.json',
        'observer':'W001/summary.json','link':'initial-main-link.json','B2_entry':'B002/entry.json',
        'B2_input':'B002/input.json','B2_fit_entry':'B002/fit/entry-capture.json','W0_reuse':'W0-reuse.json'}.items()}
    missing=[str(p) for p in paths.values() if not p.exists()]
    if missing:return dict(arm=arm,status='NOT_OBSERVED',missing=missing)
    d={name:load(p) for name,p in paths.items()}
    config=load(root/'config.json');layers=config['profile']['eligible_layers']
    c=d['commit'];w=d['writer'];entry=d['B2_entry']
    require(c['candidate_count']==25 and c['Adam_updates']==24,'MAIN_B1_BUDGET')
    require(c['layers']==layers and w['layers']==layers,'MAIN_ALL_LAYERS')
    require(c['history_appends']==w['history_appends']==len(layers),'MAIN_ALL_HISTORY_APPEND')
    require(c['after']==w['after']==entry['state']==d['observer']['state']==d['link']['B2_own_entry_state'],'MAIN_B1_OBSERVER_B2_STATE')
    require(entry['own_entry'] and d['observer']['no_mutation'] and not d['observer']['optimizer_feedback'],'MAIN_OBSERVER_OR_ENTRY')
    require(d['B2_input']['identity']==d['B2_fit_entry']['input_identity']==d['link']['B2_input_identity'],'B2_ACTUAL_CAPTURE_IDENTITY')
    require(entry['ids']==d['B2_input']['ids']==d['link']['B2_ids'],'B2_ID_ORDER')
    require(d['B2_fit_entry']['candidate1_differentiable'] and d['B2_fit_entry']['history_appends']==0,'B2_CAPTURE_BEFORE_WRITE')
    for key,path in [('commit',paths['commit']),('observer',paths['observer'])]:
        require(d['complete'][key]['sha256']==sha(path),'B1_COMPLETE_MEMBER_HASH')
    require(d['W0_reuse']['cold_W0_H0_exact'],'MAIN_COLD_INITIAL')
    return dict(status='ACTUAL_REPRESENTATIVE_MAIN_INITIAL_CONFIRMED',arm=arm,
        main_B1_commit=True,all_layer_history_append=len(layers),observer_restored=True,
        B2_own_entry_actual_capture=True,evidence={k:member(v) for k,v in paths.items()},
        scope='Representative main B1→B2 only; not two-arm/all20 completion',
        monitoring_active=False,automatic_resume=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    results=[inspect(a.attempt,arm) for arm in ['JLZ_A','JLZ_B']]
    confirmed=[r for r in results if r['status']=='ACTUAL_REPRESENTATIVE_MAIN_INITIAL_CONFIRMED']
    if not confirmed:
        print(json.dumps(results));return
    result=dict(status='INITIAL_GATE_PASS_MONITORING_STOPPED',representative=confirmed[0],
        other_arm_observed=results[1] if confirmed[0]['arm']=='JLZ_A' else results[0],
        submission=member(a.attempt/'submission.json'),execution_lock=member(a.attempt/'execution.lock.json'),
        monitoring_active=False,automatic_resume=False,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    write(a.attempt/'initial-handoff.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()

"""One CPU reduction of the first available representative MAIN B1->B2.

No scheduler, sleeps, recurring monitor, extra model call or terminal collection.
Called by the owner only before the authorized initial monitoring boundary.
"""
import argparse
import datetime
import json
from pathlib import Path
from .common import write,sha,member,require
from .collect import extract_rows,validate_rows,reduce_rows,validate_ledger

def review(root):
    root=Path(root);sub=json.loads((root/'submission.json').read_text())
    config=json.loads((root/'config.json').read_text());configsha=sha(root/'config.json');locksha=sha(root/'execution.lock.json')
    for arm in ('A','B'):
        folder=root/('main-'+arm);path=folder/'INITIAL_VALID.json'
        if not path.exists():continue
        initial=json.loads(path.read_text());firstpath=folder/'B001-committed.json'
        first=json.loads(firstpath.read_text());entry=json.loads((folder/'B002-entry.json').read_text())
        observation=json.loads((folder/'W01-observations.json').read_text())
        require(initial['phase']=='main' and initial['arm']==arm and initial['main_representative'] is True,'REPRESENTATIVE_MAIN_NOT_PILOT')
        require(initial['B1']['sha256']==sha(firstpath),'INITIAL_B1_RECEIPT')
        require(first['config_sha256']==entry['config_sha256']==configsha and first['execution_lock_sha256']==entry['execution_lock_sha256']==locksha,'INITIAL_SOURCE_CONFIG')
        require(first['post']==entry['entry']==initial['B2_entry']==observation['state'],'INITIAL_W_H_LINK')
        require(first['auxiliary_post']==entry['auxiliary_entry']==initial['B2_auxiliary'],'INITIAL_RNG_CONTEXT_LINK')
        require(first['commit_gate']['weight_bitwise'] and first['commit'] and first['history_appends']==5,'INITIAL_COMMIT')
        require(observation['no_mutation'] and not observation['optimizer_feedback'],'INITIAL_OBSERVER')
        records=json.loads(Path(config['stream']).read_text());ids=[r['case_id'] for r in records[:2000]]
        counts=validate_ledger([first],ids,stage='main',schedule=config['schedules']['main'])
        rows=extract_rows(observation);validate_rows(rows,{r['case_id']:r for r in records},{k:ids[:100] for k in ('R','P','N')})
        metrics=reduce_rows(rows)
        result=dict(state='INITIAL_GATE_PASS_MONITORING_STOPPED',
           recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           representative='main-'+arm,job=sub['jobs']['main-'+arm],job_ids=sub['jobs'],
           actual_gpu_validation='REPRESENTATIVE_MAIN_B1_FINAL_COMMIT_5_HISTORY_OBSERVER_RESTORE_B2_OWN_ENTRY',
           scope='one representative; NOT both main chains complete; pilot/shared evidence separately stored',
           initial_receipt=member(path),B1_receipt=member(firstpath),B2_entry_receipt=member(folder/'B002-entry.json'),
           observer_receipt=member(folder/'W01-observations.json'),independent_CPU_metrics=metrics,
           budget_and_history=counts,checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
           numerical_certification='NOT_ESTABLISHED',monitoring_active=False,automatic_resume=False,
           already_registered_runner_collector_continue=True,new_scientific_submission=False)
        write(root/'handoff.json',result)
        return result
    return dict(state='INITIAL_NOT_OBSERVED',monitoring_boundary_reached=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    print(json.dumps(review(a.run),ensure_ascii=False))

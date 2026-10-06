"""Compact post-registration receipts only; no scheduler/result polling."""
import json
from pathlib import Path
from project.run_scripts.jlz_interference_l1.cap_common import ROOT,member,write

def main():
    local=Path('/data/janghj/ODE-edit/local');run='logging-repair-20261007'
    evidence=local/'jlz-price-cap-base-repair-2k/logging-repair-evidence'
    cancellation=json.loads((evidence/'cancellation.json').read_text())
    regression=json.loads((evidence/'logger-regression.json').read_text())
    old_cost={row[0]:{'state':row[3],'elapsed':row[5],'allocation':row[6]} for row in regression['previous_accounting']}
    entries=[]
    for task in ['jlz-price-cap-base-repair-2k','jlz-price-alpha-writer-2k']:
        attempt=local/task/run;s=json.loads((attempt/'submission.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
        audit=ROOT/'audits/servers/server4'/task/run
        compact=dict(task=task,run_instance=run,status='REPLACEMENT_SUBMITTED_RELEASED_PENDING',source=s['source'],
            config_sha256=lock['config_sha256'],archive=lock['archive'],lock=member(attempt/'execution.lock.json'),
            jobs=s['jobs'],mapping=s['mapping'],held=member(attempt/'held-inspection.json'),
            parent_submission=s['parent_submission'],initial_snapshot=s['bounded_initial_snapshot'],
            combined_GPU_cap=2,scientific_changes=[],logging_regression_passed=5,actual_repaired_B1='NOT_OBSERVED',
            repaired_WB_startup='NOT_OBSERVED',W20='NOT_OBSERVED',monitoring_active=False,automatic_resume=False,automatic_retry=False,
            noCP=True,exact_resume='NOT_AVAILABLE',old_raw_preserved=True,
            old_jobs_accounting=old_cost,old_cancellation=cancellation,
            broadcast='NO_BROADCAST_NOT_REQUIRED',reviewer='owner replay/static/source audit; no independent reviewer')
        write(audit/'submission.json',compact)
        write(ROOT/'runs'/task/run/'submission.json',compact)
        status=dict(task_id=task,status=compact['status'],source=s['source'],jobs=s['jobs'],
            current_attempt=str(attempt),previous_attempt=str(local/task/'attempt'),
            submission=str(Path('runs')/task/run/'submission.json'),monitoring_active=False,automatic_resume=False,
            actual_repaired_B1='NOT_OBSERVED',W20='NOT_OBSERVED',old_evidence_keep=True,
            user_authority='현재 사용자 repair/전부 재제출 및 Alpha6 교체 승인',combined_server4_GPU_cap=2)
        # These are latest task-status pointers, not immutable production/raw files.
        for p in [ROOT/'tasks/status'/task/'server4.json',ROOT/'messages/server-heads/server4'/(task+'.json')]:
            p.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        entries.append({'task':task,'source':s['source'],'jobs':s['jobs'],'config_sha256':lock['config_sha256']})
    print(json.dumps(entries))

if __name__=='__main__':main()

"""Publish compact existing submission facts only; no scheduler or SDK calls."""
import csv
import json
from pathlib import Path
from .reconcile import ROOT, LOCAL, MEMIT, ALPHA, MEMIT_NEW, ALPHA_NEW, OLD_SOURCE, AUTHORITY
from project.run_scripts.jlz_interference_l1 import member, sha, require, verify, write

REPORT=ROOT/'experiment-reports/servers/server4/price-model-runs-tracking'
AUDIT=ROOT/'audits/servers/server4/price-model-runs-tracking'

def publish():
    cancel=json.loads((LOCAL/'cancellation.json').read_text())
    require(cancel['status']=='EXACT_NEVER_STARTED_PENDING_CANCELLED','CANCEL_RECEIPT')
    attempts=[('llama3','memit',MEMIT_NEW),('llama3','alphaedit',ALPHA_NEW),
        ('gptj',None,Path('/data/janghj/ODE-edit/local/jlz-price-gptj-2k/submission'))]
    held={};bindings=[];ledger=[]
    oldlock=json.loads((MEMIT/'execution.lock.json').read_text())
    require(sha(MEMIT/'config.json')==oldlock['config_sha256'] and oldlock['source_commit']==OLD_SOURCE,'KEEP_ORIGINAL_CONFIG_SOURCE')
    legacy=json.loads((MEMIT/'LLAMA_CAP075/tracking/receipt.json').read_text())
    ledger.append(dict(model='llama3',writer='memit',arm='CAP075',kept_running_or_completed_job='60001',
        cancelled_pending_job='',new_job_id='',dependency='',execution_source_commit=OLD_SOURCE,
        config_sha=oldlock['config_sha256'],WandB_run_identity=legacy['run_id'],
        WandB_validation_stage='UNCHANGED_LEGACY_RUN; SDK_ASYNC_NOT_REMOTE_ACK receipt is not remote verification',
        initial_state='RUNNING at bounded submission snapshot',not_submitted_reason='KEEP; no duplicate'))
    for model,writer,attempt in attempts:
        sub=json.loads((attempt/'submission.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
        c=json.loads((attempt/'config.json').read_text());h=json.loads((attempt/'held-inspection.json').read_text())
        require(sub['status']=='RELEASED' and sub['source']==lock['source_commit'] and sha(attempt/'config.json')==lock['config_sha256'],
            'ACTUAL_SUBMISSION_SOURCE_BINDING')
        for row in lock['launchers']:verify(row)
        verify(lock['archive']);held[str(attempt)]=h
        bindings.append(dict(model=model,writer=writer or 'memit+alphaedit',attempt=str(attempt),source=sub['source'],
            config=member(attempt/'config.json'),lock=member(attempt/'execution.lock.json'),archive=lock['archive'],
            submission=member(attempt/'submission.json'),held=member(attempt/'held-inspection.json'),
            jobs=sub['jobs'],dependencies={k:v['dependency'] for k,v in sub['mapping'].items()},
            snapshot=sub['bounded_initial_snapshot'],noCP=lock['noCP']))
        for cell,job in sub['jobs'].items():
            if cell=='collector':continue
            arm=cell.rsplit('_',1)[1];actual_writer=writer or ('memit' if cell.startswith('MEMIT_') else 'alphaedit')
            old={'LLAMA_CAP100':'60002','LLAMA_FREE100':'60003','LLAMA_AE_CAP075':'60011',
                'LLAMA_AE_CAP100':'60012','LLAMA_AE_FREE100':'60013'}.get(cell,'')
            ledger.append(dict(model=model,writer=actual_writer,arm=arm,kept_running_or_completed_job='',
                cancelled_pending_job=old,new_job_id=job,dependency=sub['mapping'][cell]['dependency'] or '',
                execution_source_commit=sub['source'],config_sha=lock['config_sha256'],WandB_run_identity='NOT_OBSERVED_BEFORE_JOB_STARTUP',
                WandB_validation_stage='CPU_RAW_AND_FAKE_SDK_VERIFIED; actual online startup/readback NOT_OBSERVED',
                initial_state='PENDING at bounded released snapshot',not_submitted_reason=''))
    require(len(ledger)==12 and len({(r['model'],r['writer'],r['arm']) for r in ledger})==12,'TWELVE_CELL_LEDGER_NO_DUPLICATE')
    require({r['new_job_id'] for r in ledger if r['new_job_id']}=={'60102','60103','60105','60106','60107','60112','60113','60114','60115','60116','60117'},'ACTUAL_EXPECTED_NEW_IDS')
    REPORT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'cell-ledger.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    small=[]
    for name in ('logger-cpu.json','source-final.json','gptj-final-static.json','gptj-existing-raw-review.json','hold-complete.json','cancellation.json','submission.json','held-inspection.json'):
        small.append(member(LOCAL/name))
    receipt=dict(schema=1,authority=AUTHORITY,stage='RELEASED_BOUNDED_HANDOFF',protected_job='60001',
        protected_job_mutations=0,original_source=OLD_SOURCE,executed_source=bindings[0]['source'],
        new_GPU_jobs=11,kept_GPU_jobs=1,new_CPU_collectors=3,server4_target_cells=12,
        target18_includes_server1_separate6=True,bindings=bindings,
        cancellation=dict(receipt=member(LOCAL/'cancellation.json'),jobs=[x['before']['job'] for x in cancel['cancelled']],
            all_never_started_terminal_unallocated=True,order='collector first; reverse topological exactGPU successors'),
        combined_cap=3,cap_proof='Llama retained60001+five successors antichain1; GPTJ native input predecessor thentwo lanes antichain2; combined<=3',
        GPTJ_admission_prior_width=held[str(attempts[-1][2])]['predecessor']['maximum_concurrency'],
        combined_output_reserve_bytes=36590583808,free_bytes_at_final_config=41496006656,
        storage_reservation=False,storage='fresh admission estimate, not exclusive future guarantee; runtime batchboundary guards retained',
        new_GPU_qualification='NOT_OBSERVED',new_WandB_online_identity='NOT_OBSERVED',new_W20='NOT_OBSERVED',
        monitoring_active=False,automatic_resume=False,automatic_retry=False,Qwen='STOP_UNCHANGED',
        local_receipts=small,broadcast='NO_BROADCAST_NOT_REQUIRED; raw/credentials/spool original localKEEP')
    write(AUDIT/'submission-receipt.json',receipt)
    runpath=ROOT/'runs/price-model-runs-tracking/server4/submission.json';write(runpath,receipt)
    manifest=dict(schema=1,execution_source=bindings[0]['source'],analysis_source='publish_reconciliation.py after freeze; no execution source mutation',
        files=[member(REPORT/'cell-ledger.csv'),member(AUDIT/'submission-receipt.json'),member(runpath)],local_receipts=small)
    write(AUDIT/'artifact-manifest.json',manifest)
    return dict(stage=receipt['stage'],new_GPU_jobs=11,kept_GPU_job='60001',collectors=[60104,60108,60118],
        report=str(REPORT/'report-ko.md'),scheduler_queries=0,WandB_calls=0)

if __name__=='__main__':print(json.dumps(publish()))

"""Seal the completed CPU report update; never invokes Slurm or a model."""
import csv
from pathlib import Path
from .common import read, record, save, sha
from .review_b010 import PUB
from .review_final_eval import ROOT, SOURCE, LOCK

def run():
    audit=Path('audits/servers/server2/joint-multilayer-bs1-20260929-v1/b010-review-20260930-v1')
    dest=audit/'final-eval-update-r1';dest.mkdir(exist_ok=False)
    prior=read(audit/'artifact-manifest.json')
    unchanged=[]
    for r in prior['outputs']:
        if r['path'].endswith('/report-ko.md'):continue
        assert sha(r['path'])==r['sha256'];unchanged.append(r['path'])
    out=PUB/'final-eval-update-r1'
    metrics=list(csv.DictReader((out/'metrics.csv').open()))
    assert len(metrics)==9
    report=(PUB/'report-ko.md').read_text()
    for text in ('744/1000 (74.4%)','785/1000 (78.5%)','786/1000 (78.6%)','507 GPU-seconds'):
        assert text in report
    scheduler=dict(command='sacct -X -j 55331 --array --format=JobID,JobName%32,State,ExitCode,ElapsedRaw,AllocTRES%70 -n -P',
        observation_date='2026-09-30 Asia/Seoul',source='owner tool output during explicit JOINT_CUM completion recall',
        rows=[dict(job=f'55331_{i}',name='odeedit_joint_finaleval_s2',state='COMPLETED',exit_code='0:0',
            elapsed_gpu_seconds=n,gres_gpu=1,cpu=8,mem='59G') for i,n in enumerate((173,168,166))],
        squeue_exact_array_empty=True,unrelated_scheduler_queries=0)
    save(dest/'terminal-scheduler-receipt.json',scheduler)
    save(dest/'owner-review.json',dict(status='CPU_REPORT_UPDATE_VERIFIED',independent_red_agent=False,
        cpu_tests='test_review_b010: 6 passed',reducer='Completed terminal raw reduction: 3 arms, 7800 rows, 300 request files',
        retained_original_artifacts=unchanged,original_manifest=record(audit/'artifact-manifest.json'),
        report_update_authority='User: include odeedit_joint_finaleval_s2; wait for JOINT_CUM and update specified report',
        all_three_completed=True,prior_rp_exact=True,paired_identity=True,ties_failure=True,
        new_model_forward=0,new_job_submit=0,new_cancel=0,source_raw_unchanged=True,
        limits=['No W0/parent NS1000 baseline','Nonselected full byte hash not claimed','No independent red agent',
                'Different arm endpoint comparison is not temporal recovery','Original CSV CRLF preserved'],
        scientific_promotion=False,NO_BROADCAST_NOT_REQUIRED=True))
    save(dest/'artifact-manifest.json',dict(previous_report=prior['report'],report=record(PUB/'report-ko.md'),
        execution_source=SOURCE,execution_lock_sha256=LOCK,analysis_source=[record(Path(__file__)),record(Path(__file__).with_name('review_final_eval.py'))],
        outputs=[record(p) for p in sorted(out.iterdir())],
        terminal_raw_receipts=[record(ROOT/'output'/a/'terminal.json') for a in ('NATIVE','JOINT_STEP','JOINT_CUM')],
        audit_members=[record(p) for p in sorted(dest.iterdir())],publication_source='Containing Git commit, separate from frozen execution source',
        NO_BROADCAST_NOT_REQUIRED=True))
    print(record(PUB/'report-ko.md'))

if __name__=='__main__':run()

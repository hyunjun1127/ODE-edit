"""Seal compact review outputs; no science/runtime or scheduler access."""
import argparse
import subprocess
from .review import PACKAGE, REPO, sha, read, write_json

INSTRUCTION = 'ODEEDIT-GH-SH4-ALL-JOBS-DETAILED-REVIEW-20260924-R1'
JOBS = [50974,50983,51055,51056,51057,51058,51260,52527,52528,52529,
        52530,52563,52564,52565,52566,52575,52576]


def member(path):
    return dict(path=str(path.relative_to(REPO)),bytes=path.stat().st_size,
                sha256=sha(path))


def run(source):
    assert len(source)==40
    code=REPO/'project/run_scripts/server4_completed_jobs_review_20260924'
    members=[member(p) for p in sorted(code.glob('*.py'))]
    for m in members:
        prior=subprocess.check_output(['git','show',source+':'+m['path']],cwd=REPO)
        assert prior==(REPO/m['path']).read_bytes(),('ANALYSIS_SOURCE_CHANGED',m['path'])
    manifest=dict(instruction=INSTRUCTION,analysis_source_commit=source,
        authority_main='fbeed9212ff14e4d7726ac17ec411a091a18f2b6',
        analysis_source_members=members,
        frozen_execution_sources=dict(alpha_r1='a95876f8e5c4cf59df9cd9d7f824d1ac99f8bc77',
          alpha_r3='f9fbd56f31b0c520763ec9026e660a76cb3074ff',
          alpha_r4='a21cffa08d4cf86ed258acabdb786fba778f4dfb',
          SLMF='5f79085629b10b2bb8bdee88d017e18a46bb4c74',
          EN_adaptive='b6e86234640ca127546aee094f2a67bbe684a490'),
        report=member(PACKAGE/'report-ko.md'),
        accounting=member(PACKAGE/'accounting-receipt.json'),
        original_input_evidence=member(PACKAGE/'authority-binding.json'),
        output_fullSHA_inventory=member(PACKAGE/'output-inventory.csv'),
        input_rehash='PRIOR_INPUT_FULL_SHA_RECEIPTS_PLUS_CURRENT_BINDING; no model/12CP full rehash',
        output_rehash='6465 terminal output paths newly hashed; 6207 unique inodes',
        reused_history=member(PACKAGE/'historical-report-index.csv'),
        runtime_changes=0,model_calls=0,Slurm_mutations=0,remote_payload_transfers=0,
        publication_commit='RECORDED_SEPARATELY_AFTER_NONFORCE_PUSH; not self-referential')
    write_json(PACKAGE/'analysis-manifest.json',manifest)
    root=dict(instruction=INSTRUCTION,analysis_source_commit=source,
        member_scope='all compact package files except rooted-receipt.json; no self-hash',
        members=[member(p) for p in sorted(PACKAGE.iterdir())
                 if p.is_file() and p.name!='rooted-receipt.json'],
        report_sha256=sha(PACKAGE/'report-ko.md'),
        analysis_manifest_sha256=sha(PACKAGE/'analysis-manifest.json'),
        validation_sha256=sha(PACKAGE/'validation.json'),
        validation_status=read(PACKAGE/'validation.json')['status'],
        original_data_mutations=0,independent_agent_red=False,
        limitations=['full_numerical_equivalence=NOT_ESTABLISHED',
          'GPU_continuation=NOT_TESTED','full_W_M_reconstruction=NOT_PERFORMED',
          'HTML_render=NOT_RUN_RENDERER_NOT_INSTALLED','pure_IO_time=NOT_SEPARATED'])
    write_json(PACKAGE/'rooted-receipt.json',root)
    receipt=dict(instruction=INSTRUCTION,status='TASK_COMPLETE_STOP',
        monitoring_active=False,automatic_resume=False,review_GPU_allocations=0,
        snapshot_utc='2026-09-23T15:52:05Z',snapshot_project_queue_empty=True,
        exact_parent_jobs=JOBS,report_path=str((PACKAGE/'report-ko.md').relative_to(REPO)),
        report_sha256=sha(PACKAGE/'report-ko.md'),
        analysis_source_commit=source,
        analysis_manifest_sha256=sha(PACKAGE/'analysis-manifest.json'),
        rooted_receipt_sha256=sha(PACKAGE/'rooted-receipt.json'),
        alpha_approved_families_completed=94,followups_not_submitted=7,
        alpha_parent_allocated_GPU_sec=50605,recent_parent_allocated_GPU_sec=61565,
        max_observed_concurrent_GPU=2,allocation_is_utilization=False,
        observed_writer_endpoints=24,observed_component_endpoints=136,
        independent_metric_rows=1608,historical_reports_reused=97,
        numerical_comparisons_policy='OBSERVER_ONLY_USER_DIRECTED; not parity PASS',
        no_broadcast='NO_BROADCAST_NOT_REQUIRED',publication='COMPLETED_BY_NONFORCE_MAIN_COMMIT_CONTAINING_THIS_RECORD')
    for relative in [
        'tasks/status/server4-jobs-review-20260924/server4.json',
        'runs/odeedit_server4_jobs_review_20260924/receipt.json',
        'audits/servers/server4/2026-09-24-completed-jobs-review/completion.json']:
        write_json(REPO/relative,receipt)
    write_json(REPO/'audits/servers/server4/2026-09-24-completed-jobs-review/validation.json',
               read(PACKAGE/'validation.json'))
    message=REPO/'messages/server-heads/server4/2026-09-24-completed-jobs-review.md'
    message.parent.mkdir(parents=True,exist_ok=True)
    message.write_text(f'''# Server4 job 상세 리뷰 완료

ACK nonce: `{INSTRUCTION}`.

CPU-only 리뷰를 마쳤다. Snapshot 2026-09-24 00:52:05 KST에서 현재 queue0,
최근 exact parent17개 및 과거 게시보고97개를 대조했다. Alpha94개 승인 family가
완료 산출물에 대응하며 후속7개는 FOLLOWUP_NOT_SUBMITTED다.
신규 model/GPU/evaluator/Slurm write/repair/삭제/원격전송0.

보고서: `{receipt['report_path']}`

- Report SHA256: `{receipt['report_sha256']}`
- Analysis source: `{source}`
- Analysis manifest SHA256: `{receipt['analysis_manifest_sha256']}`
- Rooted receipt SHA256: `{receipt['rooted_receipt_sha256']}`
- Exact jobs: {', '.join(map(str,JOBS))}.
- Alpha GPU allocation50,605초(14.056944GPUh), 최근 총61,565초; 최대동시2.

Native Current R/P/N: W50 100/190/620, W70 100/190/547,
W80 100/180/550, W90 100/186/523 (분모100/200/1000).
H56의 NS 차이는 각 entry 0/−1/0/+4개이며 PS/RS는 동일하다.
TF·joint·문항 lost/gained·geometry·KR·history·cost는 보고서/CSV에 분리했다.
최신 수치차 observer-only 정책을 parity PASS로 바꾸지 않았다.
Full numerical equivalence NOT_ESTABLISHED, GPU continuation NOT_TESTED,
HTML renderer 미설치 NOT_RUN. Owner audit+independent reducer이며 별도 agent red0.

원 runtime/raw/input12CP/실패/waiver/타task 유지.
Own-scope nonforce publication 외 추가 작업 없음.
TASK_COMPLETE_STOP; monitoring_active=false; automatic_resume=false.
''')
    # Verify rooted package without reopening large original outputs.
    for m in root['members']:
        path=REPO/m['path']
        assert path.stat().st_size==m['bytes'] and sha(path)==m['sha256']
    print(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis-source',required=True)
    run(p.parse_args().analysis_source)

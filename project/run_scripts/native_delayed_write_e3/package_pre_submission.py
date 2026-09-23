"""Compact pre-submission handoff only; copies no raw/prompt/tensor payload."""
from pathlib import Path
import datetime
import subprocess
from .common import ROOT, PANELS, INSTRUCTION, read, save, sha, file_record


def main():
    repo=Path(__file__).resolve().parents[3]
    report=repo/'experiment-reports/servers/server4/native-delayed-write-e3-20260924-v1'
    audit=repo/'audits/servers/server4/native-delayed-write-e3-20260924-v1'
    panels=read(PANELS/'panel-manifest.json')
    preflight=read(ROOT/'receipts/preflight-r1.json')
    lock=ROOT/'attempt-v1/execution.lock.json'
    assert not (ROOT/'attempt-v1/submission-events.jsonl').exists()
    inputs=[file_record(p) for p in sorted((ROOT/'inputs/design').rglob('*')) if p.is_file()]
    save(audit/'full-read-and-inputs.json',dict(instruction_id=INSTRUCTION,design_members=inputs,
        receiver_verified_files=15,receiver_bytes=245092,full_read_all_15=True,
        original243cells='HISTORICAL_NOT_SUBMISSION_ALLOWLIST',new_literature_PDF_read=0,
        protocol_sha256='4209c7d09fb06b81d9f0bfb2b0c86076aa884099bca8be8d9d3e14937f8ffae4',
        protocol_full_read='EXACT_PRIOR_RECEIPT_REBOUND + transfer clauses reread',
        actual_session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',root='/data/janghj/ODE-edit',
        dedicated_worktree=str(repo),baseline_source='read-only original BASE/native/evaluator; new backend separate'))
    compact={k:v for k,v in preflight.items() if k not in ('test_stdout','test_stderr')}
    compact['test_log_local_only']=str(ROOT/'receipts/preflight-r1.json')
    compact['test_result_summary']='9 passed; full CPU log local only'
    save(report/'preflight.json',compact)
    save(report/'panel-summary.json',{k:v for k,v in panels.items() if k not in ('selected_N_case_ids',)})
    memit=read(ROOT/'receipts/memit-transfer.json')
    save(repo/'transfers/verifications/2026-09-24-native-delayed-write-e3-sh4/memit-received.json',memit)
    status=dict(instruction_id=INSTRUCTION,status='BLOCKED_STORAGE_NOT_SUBMITTED',job_ids=[],sbatch_calls=0,
        GPU_model_calls=0,actual_gates={x:'NOT_RUN' for x in read(lock)['allowed_stages']},
        current_scope='existing BASE endpoint E0 -> E1 -> E3, fixed25endpoints/12pairs',
        execution_source='3ebe0b07078940c2d46f9ea2226ccc20c0446162',execution_tree='7dece41be8ed2666d05a96cdb8432828c88fb5c0',
        archive=file_record(ROOT/'source-r1.tar'),lock=file_record(lock),
        submission_admission_error="AssertionError: ('BLOCKED_STORAGE', 4236369920)",
        free_bytes_at_failed_admission=4236369920,required_output_temp_safety_bytes=4*2**30,
        deficit_bytes=4*2**30-4236369920,reserved_or_exclusive_space=False,
        exact_receiver_inputs=15,memit_checkpoint_receiver_files=12,memit_receiver_bytes=memit['bytes'],
        checkpoint_saved=False,existing_checkpoint_inputs_preserved=True,exact_new_resume='NOT_AVAILABLE',
        project_gpu_cap=2,task_gpu_cap=1,other_job_mutations=0,deletions=0,
        monitoring_active=False,automatic_resume=False,stage_outcome='NO_SCIENTIFIC_RESULT',
        NO_BROADCAST_NOT_REQUIRED=True,recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(report/'terminal.json',status)
    save(repo/'tasks/status/native-delayed-write-e3-20260924-v1/server4.json',status)
    save(repo/'runs/odeedit_native_delayed_write_e3_s4_20260924/pre-submission.json',status)
    save(audit/'owner-review.json',dict(independent_agent_review=False,CPU_tests=9,CPU_tests_status='PASS',
        actual_Llama_validation='NOT_RUN',source_byte_freeze=True,raw_free_Git=True,production_native_modified=False,
        allowed_science_enumeration='25 endpoint logical / 12 fixed pairs / no new fitting or writes',
        required_remaining=['storage admission','held registration/inspection/release','G10 actual endpoint checks','E1/E3 actual outputs','CPU terminal reducer/full postrun audit'],
        limitations=['BaseEval external entity aliases unavailable','new runtime GPU semantics not established','memory/wall are estimates']))
    members=[file_record(p) for root in (report,audit) for p in sorted(root.rglob('*')) if p.is_file()]
    save(report/'handoff-manifest.json',dict(instruction_id=INSTRUCTION,members=members,large_inputs_local_only=True,source_archive_sha256=sha(ROOT/'source-r1.tar')))
    print(status['status'],status['job_ids'],status['deficit_bytes'])


if __name__=='__main__':main()

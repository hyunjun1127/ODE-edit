"""이미 한 번 조회한 accounting과 exact local 실패 자료만 CPU로 결속한다."""
import argparse,csv,hashlib,json
from pathlib import Path


def member(path):
    path=Path(path);b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root;diag=root/'failure-diagnosis-r1'
    lock=json.loads((root/'control/execution.lock.json').read_text());submission=json.loads((root/'control/submission.json').read_text())
    columns='JobIDRaw JobID JobName User State ExitCode DerivedExitCode Start End ElapsedRaw AllocTRES NodeList ReqCPUS ReqMem MaxRSS TotalCPU Reason'.split()
    accounting=[dict(zip(columns,row,strict=True)) for row in csv.reader((diag/'accounting-once.txt').read_text().splitlines(),delimiter='|')]
    parents=[x for x in accounting if '.' not in x['JobID']]
    assert [x['JobID'] for x in parents]==['52527','52528','52529','52530']
    names={str(v):'odeedit_alpha_key_'+k+'_s4' for k,v in submission['jobs'].items()}
    for row in parents:
        assert row['User']=='janghj' and row['JobName']==names[row['JobID']]
        row['allocated_gpu_seconds']=int(row['ElapsedRaw'])*(int(next((s.split('=')[1] for s in row['AllocTRES'].split(',') if s.startswith('gres/gpu=')),'0')))
    assert sum(x['allocated_gpu_seconds'] for x in parents)==44
    files=[root/'control'/x for x in ('execution.lock.json','submission.json','held-inspection.json','submission-events.jsonl','gate.sh','geometry.sh','writers.sh','reduce.sh')]
    out=root/'execution/attempt-r1'
    artifacts=[p for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink()]
    assert {str(p.relative_to(out)) for p in artifacts}=={'gate-failure.json','gate/checkpoint-content-reuse.json','gate/prefix-g1/G1-first-failure.json','raw-inventory.json'}
    files+=artifacts
    files+=[root/'logs'/x for x in ('gate-52527.err','gate-52527.out','reduce-52530.err','reduce-52530.out')]
    files+=[diag/'accounting-once.txt',diag/'bos-cpu-reproduction.json']
    files+=[root/'receipts'/x for x in ('checkpoint-transfer-r1.json','native-input-binding-r2.json','cpu-final-r3.json','preflight-r1/checkpoint-content/checkpoint-content.json')]
    auto=Path(lock['publication_repo'])/'experiment-reports/servers/server4/alpha-key-causal-20260923-r1/generated-r1'
    auto_members=[member(p) for p in sorted(auto.iterdir()) if p.is_file()]
    sources=[]
    for m in lock['execution_source_members']:
        got=member(Path(lock['repo'])/m['relative_path']);assert (got['bytes'],got['sha256'])==(m['bytes'],m['sha256'])
        sources.append(got)
    # Stat reuse only: no 63GB checkpoint content hash scan or tensor reload.
    prior=json.loads((root/'receipts/preflight-r1/checkpoint-content/checkpoint-content.json').read_text())
    cp=[]
    for row in prior['rows']:
        s=Path(row['path']).stat();old=row['current_stat']
        same=(s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size)==(old['device'],old['inode'],old['mtime_ns'],old['size'])
        assert same
        cp.append(dict(batch=row['batch'],path=row['path'],bytes=s.st_size,prior_file_sha256=row['file_sha256'],same_stat_as_prior_full_content_verification=same,current_full_rehash=False))
    failure=json.loads((out/'gate-failure.json').read_text());g1=json.loads((out/'gate/prefix-g1/G1-first-failure.json').read_text())
    assert failure['job_id']=='52527' and failure['message']=='WRITER_UNEXPECTED_BOS: 0'
    assert not g1['comparisons'] and not g1['timings']
    assert not (out/'gate/READY.json').exists() and not (out/'gate/prefix-g1/G1-restore-failure.json').exists()
    reproduced=json.loads((diag/'bos-cpu-reproduction.json').read_text())
    assert reproduced['failure']['message']==failure['message'] and reproduced['actual_sequences']==12
    value=dict(status='DIAGNOSIS_COMPLETE_NOT_REPAIRED',nonce='ODEEDIT-GH-SH4-ALPHA-KEY-GATE-FAILURE-CHECK-20260923-R1',
       execution_source=lock['source_commit'],execution_tree=lock['source_tree'],execution_lock_sha256=member(root/'control/execution.lock.json')['sha256'],
       archive=dict(**lock['archive'],verification='PRIOR_FULL_SHA_AND_CURRENT_SIZE; NOT_REHASHED_THIS_DIAGNOSIS',current_bytes=Path(lock['archive']['path']).stat().st_size),
       accounting=parents,step_accounting=accounting,allocated_gpu_seconds=44,allocated_gpu_hours=44/3600,
       failure_elapsed_seconds=failure['elapsed_seconds'],g1_pre_cleanup_seconds=g1['wall_seconds'],
       first_cause=dict(type=failure['exception_type'],message=failure['message'],source='technical.py:200',category='TOKENIZATION_ASSERTION_ASSUMPTION_ERROR',
           CPU_reproduced=True,original_current_token_pack_exact=reproduced['original_current_token_pack_exact'],
           note='add_bos_token=False attribute does not alter this generic fast tokenizer backend; historical full-run tokens not independently observed'),
       cleanup=dict(separate_failure_record=False,independent_restore_receipt=False,evidence='finally restore/hash/RNG check returned before rethrow at line290; source-flow evidence only'),
       stage=dict(G0='SOURCE_INPUT_MODEL_W50_REACHED; FULL_ACTUAL_G0_NOT_SEALED',G1='FAILED_BEFORE_FIRST_FORWARD_AT_TOKEN_FIXTURE',G2='NOT_RUN',G3='REGISTERED_GRAPH_ONLY; AFTEROK_PREVENTED_SCIENCE',
                  native100_requests_completed=0,SHAM_completed=False,completed_contrast_families=0,expected_contrast_families=94,new_history_appends=0,
                  key_forward_comparisons=0,timestamp_bank_created=False),
       later_jobs=dict(afterok_science_started=False,geometry_writers='CANCELLED_START_NONE_ALLOCATION_NONE',cancellation_actor='NOT_RECORDED_IN_ACCOUNTING; kill-on-invalid-dep=yes was submitted',
                       reducer='COMPLETED_AFTERANY_GPU0; report correctly TECHNICAL_FAILED_OR_PARTIAL'),
       source_current_hashes=sources,input_checkpoint_stat_reuse=cp,evidence=[member(x) for x in files],automatic_reducer_artifacts=auto_members,
       minimal_proposal='Keep original native tokenizer/model/science unchanged; replace no-BOS assumption with exact original native token/backend/mask/lookup binding and actual-tokenizer CPU regression; no implementation this turn',
       restart_scope_if_authorized='New immutable gate attempt from approved W50/input CP; G1/timestamp unfinished. Native100 fit has not run, so no valid new targets/SHAM/science to reuse. No exact new-process resume checkpoint.',
       diagnosis_only=dict(new_GPU=0,model_loads=0,forwards=0,slurm_writes=0,runtime_edits=0,threshold_edits=0,checkpoint_deletes=0,independent_agent_red=False),
       scratch_harness_notes=['CPU diagnostic first attempt lacked native import cwd globals.yml; corrected only diagnostic script',
                              'CPU diagnostic second attempt report parser assumed postprocessor single field; corrected Sequence rendering only'],
       monitoring_active=False,automatic_resume=False,terminal='STOP_AWAITING_USER')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=value['status'],source_members=len(sources),input_CP_stat_unchanged=len(cp),raw_members=len(artifacts),
                         source_evidence_files=len(files),reducer_artifacts=len(auto_members),allocated_GPU_seconds=44),ensure_ascii=False))


if __name__=='__main__':main()

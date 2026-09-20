"""Build small exact allowlist handoff only; no network, no raw edits."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

BASE = Path('/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/server4-migration-r1')
ATTEMPT = BASE/'attempt-v1'
ROOT = Path(__file__).resolve().parents[5]
AUDIT = Path(__file__).resolve().parent
REPORT = ROOT/'experiment-reports/servers/server4/en-adaptive-nullspace-2026-09-20-v1/failure-handoff-r1'
DEST = '/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/server3-repair-r1/handoff-from-server4'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def save(p, data):
    with p.open('x') as f: json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    assert ROOT.name == 'worktree'
    inventory = json.loads((AUDIT/'raw-inventory.json').read_text())
    for row in inventory:
        p=Path(row['path'])
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], str(p)
    files = ['execution.lock.json','submission.json','held-inspection.json','release.json',
             'execution-inputs/manifest.json','execution-inputs/evaluator-map.json',
             'execution-inputs/source-allowlist.json','execution-inputs/asset-inventory.json',
             'output/technical-failure.json','output/runtime-load.json','output/T0-result.json',
             'output/T0/t0-final-state.json','output/T0/t0-observations.json',
             'output/T0/z-source-comparison.json','logs/51260.err']
    files += ['source/project/run_scripts/en_adaptive_nullspace/'+x+'.py' for x in
              ('runner','controller','geometry','objective','runtime','native','current','selector','technical','metrics','test_server4')]
    members=[]
    bundle=BASE/'failure-handoff-r1'/'small-handoff.tar.gz'
    with bundle.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for name in files:
                    p=ATTEMPT/name
                    data=p.read_bytes()
                    assert len(data)<2*1024**2, name
                    row=dict(source=str(p),destination=DEST+'/'+name,member=name,
                             bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),owner='SH4',
                             operation='OPTIONAL_SH3_SOLE_PULL_SOURCE_KEEP')
                    members.append(row)
                    info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;info.mtime=0
                    tar.addfile(info,io.BytesIO(data))
    manifest=dict(schema='small-failure-handoff-v1',job_id='51260',
        instruction='ODEEDIT-S06-EN-ADAPT-B300-FAILURE-HANDOFF-SH4-V1',
        execution_commit='b6e86234640ca127546aee094f2a67bbe684a490',
        execution_tree='5eacfc956214ffbd0cfccbeb56c068a888b88990',
        analysis_pre_recall='1bb93e1d45d9144faf8af0aa9574a6d2c0dab438',
        archive_sha256='eee7e31a072fcf54d4390f5d6679ef97b75f848208b3e9996b452f1ec81e9867',
        lock_sha256='cfc4e2150cf57f71d0fea838d2302d84626402b8ad9e8941690da3220bd671ca',
        package=dict(source=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),
            destination=DEST+'/small-handoff.tar.gz'),
        files=members,reference_transfer_bytes=0,new_history_teacher_count=0,
        source_git_preferred=True,transfer_performed=False,source_keep=True,
        excluded=['full stdout','model','teacher','keys','current prompt-bearing raw','full spectrum','weights','partial controller raw'],
        fresh_restart='W0_ZERO_M4_B1_SHARED_NATIVE_REQUIRED',exact_resume='NOT_AVAILABLE')
    save(REPORT/'handoff-manifest.json',manifest)
    account=dict(job_id='51260',job_name='odeedit_en_adapt_B300_s4',owner='janghj',node='server4',
                 state='FAILED',exit_code='1:0',elapsed_seconds=6218,gpus=1,allocated_gpu_seconds=6218,
                 start_scheduler='2026-09-20T18:36:31',end_scheduler='2026-09-20T20:20:09',
                 batch_maxrss_kib=55899724,extern_added=False,batch_added=False,
                 observation='One exact sacct read on user recall; transcription of returned fields',
                 command='sacct -j 51260 --starttime 2026-09-20 -P --format=JobIDRaw,JobName%40,User,State,ExitCode,ElapsedRaw,Start,End,NodeList,AllocTRES,MaxRSS')
    save(AUDIT/'accounting.json',account)
    assert len(inventory)==17
    save(AUDIT/'checks.json',dict(raw_identity_unchanged='PASS',raw_members=len(inventory),
        partial_controller='INVALID_PARTIAL_PRESERVED',gpu_calls=0,model_calls=0,scheduler_mutations=0,
        failure_fixture='REPRODUCED',nan_inf_tensor_negative='PASS',production_edits=0,
        method_threshold_changes=0,independent_agent_review=False,source_git_only=True,
        renderer='NOT_RUN',original_archive_hash_checked=True,full_teacher_rehash=False))
    print(json.dumps(manifest['package'],indent=2))


if __name__ == '__main__': main()

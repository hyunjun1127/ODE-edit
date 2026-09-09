"""Create-once execution/metric/source locks; no scheduler or model calls."""
import argparse
import json
from pathlib import Path
import subprocess
from verify_ready import file_sha, write_once
from evaluator_adapter import TASKS
from rte_scoring import VERSION, AUTHORITY


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no']).strip()
    a.attempt.mkdir(mode=0o700,parents=False,exist_ok=False)
    (a.attempt/'logs').mkdir(mode=0o700)
    imports=Path('/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1')
    preedit_lock=Path('/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/execution.lock.json')
    # Read-only immutable asset/dependency seal, not PRE_EDIT job or results.
    assert file_sha(preedit_lock)=='7ea4991018fb18b1cb4dc520acf3e6adb386ad39112440fa2f57af9e33bdf167'
    baseline=json.loads(preedit_lock.read_text())
    snapshot=baseline['snapshot']
    deps=baseline['dependencies']
    paths=[x for x in baseline['members'] if x['path'].startswith(snapshot+'/') or x['path'].startswith(deps+'/')]
    assert paths and any(x['path'].endswith('model.safetensors.index.json') for x in paths)
    members={x['path']:x for x in paths}
    for x in paths:
        path=Path(x['path'])
        assert path.stat().st_size==x['bytes'] and file_sha(path)==x['sha256'],path
    for directory in (imports/'source', imports/'evaluator-source', imports/'evaluator-import-supplement',Path(__file__).parent):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                members[str(path)]=dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))
    audit=Path('/mnt/raid5/janghj/ODE-edit/local/state/downstream-dataset-20260909-v1/source-audit.json')
    receipt=Path('/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/rte-corrected-payload-verification-v1.json')
    verify=json.loads(receipt.read_text())
    assert verify['payload_file_verification']=='PASS' and verify['tests']=='PASS'
    for path in (audit,receipt,imports/'transfer-ready.json'):
        members[str(path)]=dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))
    lock=dict(instruction_id='ODEEDIT-S06-BLUE-CHECKPOINT-DOWNSTREAM-S2-S4-V1',
        source_head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        source_tree=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD^{tree}'],text=True).strip(),
        source_root=str(root),imports=str(imports),snapshot=snapshot,revision=baseline['revision'],
        dependencies=deps,dataset_root='/mnt/raid5/janghj/EasyEdit/glue_eval/dataset',dataset_audit=str(audit),
        seed=20260907,model_dtype='torch.float32',backend='eager',generation='source_exact_greedy_5',
        fewshot=0,eval_slice=[10,110],tasks=list(TASKS),checkpoint_count=72,W0_count=1,
        expected_task_evaluations=438,expected_examples=43800,
        mapping=VERSION,mapping_authority=AUTHORITY,mmlu_parser='SOURCE_EXACT',
        generation_and_alternative_metrics='SEPARATE_WEIGHTED_F1',official_full_benchmark=False,
        import_policy='DIRECT_SIX_CLASSES; unused GLUEEval/util wrapper not imported; original 13+2 sealed',
        resource=dict(gpus=1,cpus=8,mem_mib=60416,project_cap=2,walltime='48:00:00'),
        monitoring='FIRST_VALID_W0_AND_CP_THEN_PAUSED_AWAITING_GH',
        preedit_job_access=0,edit=0,backward=0,history_apply=0,scientific_promotion=False,
        members=list(members.values()))
    write_once(a.attempt/'execution.lock.json',lock)
    print(json.dumps(dict(status='PREPARED_NOT_SUBMITTED',source_head=lock['source_head'],
                         lock=str(a.attempt/'execution.lock.json'),sha256=file_sha(a.attempt/'execution.lock.json'),
                         members=len(members),checkpoints=72,tasks=6)))


if __name__=='__main__':
    main()

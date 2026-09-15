"""Create-once execution closure and input lock, never submits or loads a model."""
import argparse
import json
import os
from pathlib import Path
import shutil
from .preflight import TASK_ROOT, BASELINE_ROOT, file_identity, git, digest, CHECKPOINTS, DATA_ROOT


def write_once(path, value):
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)
        f.write('\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--preparation',type=Path,required=True)
    a=ap.parse_args();repo=a.repo.resolve();task=TASK_ROOT
    pre=json.loads(a.preparation.read_bytes())
    assert pre['storage']['tensor_checkpoint_storage'] is False
    assert pre['storage']['status']=='DISK_ARITHMETIC_ONLY_PASS'
    assert not git(repo,'status','--porcelain'), 'SOURCE_MUST_BE_COMMITTED_CLEAN'
    source=task/'source';source.mkdir(mode=0o700,exist_ok=False)
    package=repo/'project/run_scripts/cake_native_lifelong'
    shutil.copytree(package,source/'cake_native_lifelong',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    loader=source/'fixed_counterfact.py';shutil.copyfile(repo/'scripts/fixed_counterfact.py',loader)
    baseline=json.loads((BASELINE_ROOT/'execution.lock.json').read_bytes())
    helper=Path(baseline['source_root'])
    helper_paths=[
        'project/run_scripts/blue_alphaedit_sequential_comparison/__init__.py',
        'project/run_scripts/blue_alphaedit_sequential_comparison/evaluation.py',
        'project/run_scripts/blue_alphaedit_sequential_comparison/integrity.py',
        'project/run_scripts/alphaedit_strength_neutral_barrier/contracts.py',
        'project/run_scripts/alphaedit_strength_neutral_barrier/evaluator.py',
        'project/run_scripts/ordered_response_barrier_ode/__init__.py',
        'project/run_scripts/ordered_response_barrier_ode/counterfact_locality_evaluator.py',
    ]
    members=[file_identity(p) for p in sorted(source.rglob('*')) if p.is_file()]
    cake=task/'execution/CAKE'
    members += [file_identity(cake/p) for p in git(cake,'ls-files').splitlines()]
    members += [file_identity(helper/p) for p in helper_paths]
    deps='/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2/'
    # Preserve old sealed dependency bytes; do not replace the shared environment.
    members += [x for x in baseline['members'] if x['path'].startswith(deps)]
    members += [pre['contexts'] if 'contexts' in pre else pre['context'],pre['original_hparams']]
    members += pre['asset_references']
    snapshot=Path(baseline['snapshot'])
    for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json','generation_config.json']:
        p=snapshot/name
        if p.is_file():members.append(file_identity(p))
    # Full model shards were previously sealed; preserve references and readability.
    index=json.loads((snapshot/'model.safetensors.index.json').read_bytes())
    shards=[]
    for rel in sorted(set(index['weight_map'].values())):
        p=snapshot/rel;assert p.is_file() and os.access(p,os.R_OK)
        expected=next((x for x in baseline['members'] if x['path']==str(p)),None)
        shards.append(dict(path=str(p),bytes=p.stat().st_size,previous_lock_identity=expected,
                           content_verification='PRIOR_MODEL_CLOSURE_REUSE_CURRENT_READABILITY_NO_REDUNDANT_FULL_REHASH'))
    dedup={x['path']:dict(path=x['path'],bytes=x['bytes'],sha256=x['sha256']) for x in members}
    for x in dedup.values():
        assert Path(x['path']).stat().st_size==x['bytes']
    projector=next(x for x in pre['asset_references'] if x['path']==baseline['projector'])
    context=pre['context']
    lock=dict(instruction_id=pre['instruction_id'],method='CAKE_NATIVE',policies=['CAKE_NATIVE'],batches=100,batch_size=100,
        checkpoint_storage='DISABLED_USER_DIRECTED_NO_W_M_TENSORS',storage_user_override=pre['user_override'],
        exact_restart_available=False,output=str(task/'output/main'),source_root=str(source),helper_root=str(helper),
        source=dict(head=git(repo,'rev-parse','HEAD'),tree=git(repo,'rev-parse','HEAD^{tree}'),branch=git(repo,'branch','--show-current')),
        upstream=dict(head=pre['upstream_head'],tree=pre['upstream_tree'],original_main=pre['original_main'],execution_main=pre['execution_main'],patch='remove unused notebooks.util import only'),
        cake_root=str(cake),hparams_path=str(cake/'hparams/Cake/Llama3-8B.json'),hparams=pre['hparams'],
        evaluation_batches=CHECKPOINTS,fixed_loader=str(loader),dataset_root=str(DATA_ROOT),
        sample_root=pre['dataset']['ordered_root'],case_order_sha256=pre['dataset']['case_order_sha256'],
        sample_batches=pre['batches'],snapshot=baseline['snapshot'],revision=baseline['revision'],seed=baseline['seed'],
        torch_version='2.9.1+cu128',projector=projector,projector_mapping=pre['projector_mapping'],
        projector_tensor_sha256='f5c2582d872d76607bf5d65df4ae983af180f028b6133e39d7b8ff013f2ac510',
        contexts=context,context_digest=digest(json.loads(Path(context['path']).read_bytes())),
        stats_references=baseline['stats_members'],model_shard_references=shards,members=list(dedup.values()),
        raw_reserve_bytes=16*1024**3,resource=pre['resource_plan'],scientific_promotion=False,
        W0_reference='sealed42673; no W0 evaluation',monitoring_policy='PENDING_OR_FIRST_NATIVE_BATCH_THEN_PAUSE')
    write_once(task/'execution.lock.json',lock)
    print(json.dumps(file_identity(task/'execution.lock.json')))


if __name__=='__main__':main()

"""SH2 local asset/storage binding for two author-hparam history chains (CPU only)."""
import argparse
import json
import shutil
from pathlib import Path

from official.experiments.prepare import digest, write_new, file_sha
from official.runners.server1.assets import member, validate_c0
from official.runners.server1.fe_history_prepare import runtime

INSTRUCTION = 'USER-SH-FE-AUTHOR-HPARAMS-2K-CF-ZSRE-20261010-R1'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local/fe-author-hparams-2k-20261010')
OLD = Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1')
CONTEXT = Path('/mnt/raid5/janghj/ODE-edit/local/native-context-import/server3-20261010/job-62101/qwen25')
CONTEXT_SHA = '5c01bc1a91c0890af2897f18b7a5f95de011badc1eca5e27f44199acc0cb6515'

def read(path):
    return json.loads(Path(path).read_text())

def storage_plan(free_bytes, existing_growth_bytes=48 * 2**30):
    # Latest-only checkpoint implementation: W0 is the initial latest, not a
    # permanently retained third copy. Chain 1 W20 + chain 2 latest + atomic temp.
    tensor_bytes = 5 * 4 * (3584 * 18944 + 18944 * 18944)
    per_checkpoint = tensor_bytes + 64 * 2**20
    peak_payloads = 3
    raw_and_logs = 8 * 2**30
    reserve = 32 * 2**30
    required = peak_payloads * per_checkpoint + raw_and_logs + existing_growth_bytes + reserve
    return dict(free_bytes=free_bytes, selected_weights_history_bytes=tensor_bytes,
                checkpoint_budget_bytes=per_checkpoint, concurrent_payloads=peak_payloads,
                two_serial_chain_checkpoint_peak_bytes=peak_payloads * per_checkpoint,
                existing_jobs_growth_reserve_bytes=existing_growth_bytes,
                raw_logs_reserve_bytes=raw_and_logs, filesystem_reserve_bytes=reserve,
                required_free_bytes=required, sufficient=free_bytes >= required,
                deletes_existing_files=False, W20_future_consumer_pending=True)

def prepare_assets(output):
    output = Path(output).absolute()
    if not output.is_relative_to(LOCAL) or output.exists():
        raise ValueError('NEW_OWN_PREPARATION_DIRECTORY_REQUIRED')
    old = read(OLD / 'assets.json'); proof = read(OLD / 'asset-preflight.json')
    snap = Path(old['model_snapshot']); hashes = proof['asset_identity']['model_files_sha256']
    files = [member(snap / name, allow_symlink=True, expected_sha=sha)
             for name, sha in sorted(hashes.items())]
    c0 = {}
    for layer in range(4, 9):
        row = old['covariance'][str(layer)]
        c0[str(layer)] = dict(member=member(row['path'], expected_sha=row['sha256'],
                                           expected_bytes=row['bytes']),
                             validation=validate_c0(row['path'], 18944),
                             module=f'model.layers.{layer}.mlp.down_proj')
    context = member(CONTEXT / 'contexts.json', expected_sha='caf43aaf6e04f8b894f49051cbca4312e51b63ac42466be4edd82a70d7589dea')
    context_value = read(context['path'])
    if digest(context_value) != CONTEXT_SHA:
        raise ValueError('CONTEXT_CANONICAL_IDENTITY')
    assets = dict(model=dict(snapshot=str(snap), identity=proof['model_identity'],
                            config=read(snap / 'config.json'), members=files),
                  C0=c0, tokenizer_sha256=digest([m for m in files if not m['path'].endswith(('.safetensors','.bin'))]),
                  runtime=runtime())
    assets['assets_sha256'] = digest(assets)
    write_new(output / 'assets.json', assets)
    streams = {dataset: member(OLD / 'streams' / f'{dataset}-stream.json') for dataset in ('cf','zsre')}
    for value in streams.values():
        records = read(value['path'])
        if isinstance(records, dict): records = records['records']
        assert len(records) == 2000
    receipt = dict(instruction_id=INSTRUCTION, assets=member(output/'assets.json'), streams=streams,
                   context=context, context_ready=member(CONTEXT/'READY.json'),
                   context_canonical_sha256=CONTEXT_SHA, storage=storage_plan(shutil.disk_usage(LOCAL).free),
                   model_forward=0, GPU_qualification='NOT_RUN_USER_DISABLED', downloads=0,
                   asset_recomputation=0, asset_copies=0)
    write_new(output / 'asset-preparation.json', receipt)
    return receipt

def prepare_configs(asset_preparation, output, context_binding):
    from official.baselines.fe_author_profile import TASK, METHOD, PROFILE, profile, resolve
    from official.runners.fe_author_history import validate_config, tracking_values, load_native_context
    from official.runners.server1.common import member as compact_member, verify
    from official.tracking.schema import config as tracking_config
    from official.evaluation.zsre_query_parity import compare_queries
    from transformers import AutoTokenizer
    output = Path(output).absolute()
    assert output.is_relative_to(LOCAL) and not output.exists()
    prep = read(asset_preparation); assets = read(verify(prep['assets']))
    records = {d:read(verify(m)) for d,m in prep['streams'].items()}
    records = {d:r['records'] if isinstance(r,dict) else r for d,r in records.items()}
    tok = AutoTokenizer.from_pretrained(assets['model']['snapshot'], local_files_only=True)
    tok.pad_token=tok.eos_token; tok.padding_side='right'
    proof=compare_queries(tok,records['zsre'],model_family='qwen25')
    assert proof['requests']==2000 and proof['input_mismatches']==proof['target_mismatches']==0
    pp=output/'zsre-query-parity.json'; write_new(pp,proof)
    configs=[]
    hp=resolve('qwen25',device=0)
    assert (hp.clamp_norm_factor,hp.v_num_grad_steps,hp.v_lr,hp.v_weight_decay,hp.v_loss_layer)==(1,35,.5,.001,27)
    for dataset in ('cf','zsre'):
        rows=records[dataset]
        assert len(rows)==2000 and [r['occurrence_index'] for r in rows]==list(range(1,2001))
        c=dict(schema='official-fe-author-history-v1',instruction_id=INSTRUCTION,task_id=TASK,
            server='server2',model='qwen25',method=METHOD,dataset=dataset,
            arm=dataset+'-MEMIT_FE_HISTORY_FE_AUTHOR_HPARAMS',hparams=profile('qwen25')['hparams'],
            author_profile_sha256=file_sha(PROFILE),assets=compact_member(prep['assets']['path']),
            stream_member=compact_member(prep['streams'][dataset]['path']),stream_sha256=digest(rows),
            ordered_case_ids_sha256=digest([r['case_id'] for r in rows]),batch_size=100,batches=20,
            requests=2000,edit_seed=0,milestones=[5,10,15,20],qualification='NOT_RUN_USER_DISABLED',
            storage_min_free_bytes=32*2**30+prep['storage']['checkpoint_budget_bytes'],
            runtime=runtime(),tracking_env_file=read(OLD/'assets.json')['wandb_env'],
            output=str(output/'runs'/dataset),checkpoint_policy='W0_THEN_LATEST_EACH_BATCH_KEEP_W20',
            future_consumer_pending=dataset=='cf',author_only_changes=['clamp_norm_factor','v_num_grad_steps'])
        c.update(context_binding)
        if dataset=='cf': c['generation_schedule']='DEFERRED_CHECKPOINT_EVALUATION'
        else: c['zsre_query_parity']=compact_member(pp)
        c['config_sha256']=digest(c); validate_config(c)
        assert digest(load_native_context(c,assets,tok)) == CONTEXT_SHA
        tracking_config(tracking_values(c,{'source_commit':'a'*40}))
        path=output/'configs'/f'{dataset}.json';write_new(path,c);configs.append(compact_member(path))
    result=dict(instruction=INSTRUCTION,server='server2',model='qwen25',datasets=['cf','zsre'],configs=configs,
        asset_preparation=compact_member(asset_preparation),author_profile=compact_member(PROFILE),
        zsre_query_parity=compact_member(pp),GPU_qualification='NOT_RUN_USER_DISABLED')
    write_new(output/'preparation.json',result);return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--assets');p.add_argument('--context-binding')
    args=p.parse_args()
    result=(prepare_configs(args.assets,args.output,read(args.context_binding)) if args.assets else prepare_assets(args.output))
    print(json.dumps(result, indent=2))

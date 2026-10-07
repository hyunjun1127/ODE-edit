"""CPU/source-only, read-only assets and native JSON constructor binding."""
import argparse
import csv
import importlib
import shutil
import subprocess
from dataclasses import asdict
from .common import *
from project.run_scripts.experiment_tracking.schema import load_env
from project.run_scripts.base_model_eval.gpt2xl_server1_prepare import runtime
from scripts.fixed_counterfact import load_prefix

PARENT=Path('/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/attempt-checkpoint-repair-r1/config.json')

def prepare(out,attempt):
    contract=authority();require(not out.exists() and not attempt.exists(),'CREATE_ONCE_PREPARATION')
    require(not list(LOCAL.glob('*/submission.json')),'NONCE_NOT_REGISTERED')
    old=read(PARENT);m=old['models']['MEMIT'];ready=read(verify(old['input_reuse_ready']))
    records=load_prefix(Path(old['stream']).parent,2000);chunks=list(batches(records))
    require(len({r['case_id'] for r in records})==2000,'OCCURRENCE_ID_UNIQUENESS_FOR_EXISTING_EVALUATOR')
    require(sha(old['stream'])==contract['scope']['dataset_sha256'],'DATASET_SHA')
    schedule=ROOT/contract['scope']['schedule']
    require(sha(schedule)==contract['scope']['schedule_sha256'],'SCHEDULE_SHA')
    with schedule.open(newline='') as stream:scheduled=list(csv.DictReader(stream))
    require([int(r['case_id']) for r in scheduled]==[r['case_id'] for r in records]
        and [int(r['stream_index0']) for r in scheduled]==list(range(2000)),'SCHEDULE_OCCURRENCES')
    assets=[r for r in old['assets'] if Path(r['path']).suffix!='.pt']
    for row in assets:stat_seal(row)
    require(ready['ordered_ids_sha256']==ORDERED_SHA and ready['model_asset_identity']==m['model_asset_identity'],'INPUT_REFERENCE_IDENTITY')
    verify(ready['observer_identity'])
    model=Path(m['model']);revision=subprocess.check_output(['git','-C',str(model),'rev-parse','HEAD'],text=True).strip()
    require(revision==contract['scope']['revision'],'MODEL_REVISION')
    require(read(model/'config.json')['n_embd']==1600 and read(model/'config.json')['n_layer']==48,'GPT2_CONFIG')
    for row in m['evaluator_sources']:verify(row)
    reference={Path(r['path']).name:r for r in m['evaluator_sources']}
    for name in ('project.run_scripts.jlz_price_gpt2xl.scores','project.run_scripts.jlz_realization.inputs','project.run_scripts.jlz_pilot.prompts'):
        row=member(importlib.import_module(name).__file__)
        require(row['sha256']==reference[Path(row['path']).name]['sha256'],'EVALUATOR_EXACT_REFERENCE')
    load_env(old['tracking']['env_file'])
    from .native import adopt_native,load_native,parse_hparams,closure
    bundle=adopt_native(out/'native');parsed={};imports={}
    for arm in ARMS:
        b=load_native(bundle[arm],arm);parsed[arm]=asdict(parse_hparams(b));imports[arm]=closure(b)
    stats={}
    for layer in LAYERS:
        p=Path(old['stats_root'])/'gpt2-xl/wikipedia_stats'/f'transformer.h.{layer}.mlp.c_proj_float32_mom2_100000.npz'
        stats[str(layer)]=next(r for r in old['assets'] if Path(r['path'])==p)
    require(shutil.disk_usage(LOCAL.parent).free>8*1024**3,'STORAGE_RESERVE')
    c=dict(schema=1,instruction_id=NONCE,task_id=TASK,attempt=str(attempt),seed=20261002,
        model=m['model'],model_revision=revision,model_asset_identity=m['model_asset_identity'],
        assets=assets,stream=old['stream'],ordered_ids_sha256=ORDERED_SHA,
        cold_W0_H0=m['cold_W0_H0'],cold_W=m['cold_W0_H0']['W'],runtime=runtime(),
        observer_identity=ready['observer_identity'],evaluator_sources=m['evaluator_sources'],
        packs=[dict(batch=n,ids=[r['case_id'] for r in current],identity=None) for n,current,_ in chunks],
        native_bundle=bundle,native=dict(stats=stats,
            effective_hparams=parsed,closure=imports,context_reuse=None),
        W0_reuse=None,context_policy='OWN_NATIVE_COLD_GENERATOR_ONCE; no EasyEdit context transplant',
        authority=member(ROOT/ENVELOPE),contract=member(ROOT/CONTRACT),
        tracking=dict(env_file=old['tracking']['env_file']),run_instance=dict(attempt=attempt.name),
        resources=dict(project_cap=2,task_cap=2,gpu=1,cpu=8,host_mib=65536,hard_host_mib=183296,
            wall='2-00:00:00',collector_cpu=8,collector_host_mib=24576,collector_wall='04:00:00',
            reserve_bytes=8*1024**3,ETA='NOT_MEASURED; wall is request only'),
        noCP=True,exact_resume='NOT_AVAILABLE',z_disk_cache=False,P_required=False,H_required=False,
        repair=dict(name='PRUNE_TERMINAL_BASE_FIX',native_upstream_bitwise=False,
            formula='saved_coldW0+compressed(W20_dense-W0)',terminal_transforms=1),
        input_source=member(PARENT),schedule=member(schedule),
        broadcast='NO_BROADCAST_NOT_REQUIRED; same-host protected assets/raw KEEP')
    write(out/'config.json',c)
    write(out/'preparation.json',dict(status='CPU_METADATA_BOUND_NOT_GPU_PASS',config=member(out/'config.json'),
        native_imports=imports,context='OWN_NATIVE_NOT_YET_GENERATED',model_loads=0,GPU=False,
        C0_P_recomputed=False,large_payload_validation='prior exact SHA + unchanged size/inode/mtime'))
    return out/'config.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args();print(prepare(a.out.resolve(),a.attempt.resolve()))
if __name__=='__main__':main()

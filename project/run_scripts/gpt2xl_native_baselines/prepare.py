"""CPU metadata binding; no native apply, model load, covariance or P rebuild."""
import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from .common import *
from project.run_scripts.experiment_tracking.schema import load_env
from scripts.fixed_counterfact import load_prefix

PARENT = Path('/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/attempt-checkpoint-repair-r1')

def stat_seal(row):
    s=Path(row['path']).stat()
    require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']), 'ASSET_STAT_CHANGED')
    return row

def prepare(out,attempt):
    authority();require(not out.exists() and not attempt.exists(),'CREATE_ONCE_PREPARATION')
    old=json.loads((PARENT/'config.json').read_text());m=old['models']['MEMIT']
    ready=json.loads(verify(old['input_reuse_ready']).read_text())
    records=load_prefix(Path(old['stream']).parent,2000);list(batches(records))
    require(ready['reference_fits']==ready['updates']==ready['history_appends']==0 and ready['RNG_restored'], 'COLD_NATIVE_CONTEXT_PROVENANCE')
    for key in ('contexts_member','native_input_alignment','native_full_input_binding','observer_identity'):
        verify(ready[key])
    require(ready['model_asset_identity']==m['model_asset_identity'] and ready['ordered_ids_sha256']==ORDERED_SHA,'INPUT_READY_IDENTITY')
    for row in old['assets']:stat_seal(row)
    stats_dir=Path(old['stats_root'])
    for layer in LAYERS:
        path=stats_dir/'gpt2-xl/wikipedia_stats'/f'transformer.h.{layer}.mlp.c_proj_float32_mom2_100000.npz'
        sealed=next((row for row in old['assets'] if Path(row['path'])==path),None)
        require(sealed is not None and path.is_file(),'NATIVE_C0_EXACT_CACHE_PATH')
    for row in old['runtime']['source_members']+m['context_sources']+m['evaluator_sources']:verify(row)
    # Existing sealed large payload hashes plus unchanged inode/mtime/size; no second 6GB model hash/load.
    from .native import load_native,closure as native_closure
    native=Path(old['native_root']);native_source=native/'easyeditor'
    modules=load_native(native_source)
    closure=native_closure(modules)
    parsed={};yaml={}
    for writer in ('memit','alphaedit'):
        module=modules.memit if writer=='memit' else modules.alphaedit
        cls=module.MEMITHyperParams if writer=='memit' else module.AlphaEditHyperParams
        path=native/'hparams'/('MEMIT' if writer=='memit' else 'AlphaEdit')/'gpt2-xl.yaml'
        yaml[writer]=member(path)
        require(yaml[writer]['sha256']==authority()['native']['yaml_sha'][0 if writer=='memit' else 1],'NATIVE_YAML_AUTHORITY')
        hp=cls.from_hparams(str(path));parsed[writer]=asdict(hp)
        require(hp.layers==list(LAYERS) and (hp.v_lr,hp.v_num_grad_steps,hp.v_loss_layer,hp.v_weight_decay,hp.clamp_norm_factor,hp.kl_factor)==(.5,20,47,.5,.75,.0625),'NATIVE_PARSED_FIELDS')
        require(hp.mom2_update_weight==20000 if writer=='memit' else hp.L2==10 and hp.nullspace_threshold==.02,'NATIVE_COEFFICIENT')
    load_env(old['tracking']['env_file']) # Safe config only, no online smoke or credential serialization.
    reuse=m['W0_reuse']
    for row in reuse['chunks']+[reuse['runtime'],reuse['summary']]:verify(row)
    runtime=json.loads(verify(reuse['runtime']).read_text())
    require(runtime['cold_W0_H0']['W']==m['cold_W0_H0']['W'],'W0_WEIGHT_SEAL')
    require(shutil.disk_usage(Path('/mnt/raid5/janghj/ODE-edit/local')).free>8*1024**3,'DISK_RESERVE')
    diff=subprocess.run(['git','diff','--','easyeditor/models/memit','easyeditor/models/alphaedit','easyeditor/models/rome','easyeditor/util'],cwd=native,text=True,capture_output=True,check=True).stdout
    c=dict(instruction_id=NONCE,task_id=TASK,attempt=str(attempt),seed=20261002,
        stream=old['stream'],model=m['model'],model_revision='15ea56dee5df4983c59b2538573817e1667135e2',
        assets=old['assets'],runtime=old['runtime'],ordered_ids_sha256=ORDERED_SHA,
        model_asset_identity=m['model_asset_identity'],cold_W=m['cold_W0_H0']['W'],cold_W0_H0=m['cold_W0_H0'],
        W0_reuse=reuse,observer_identity=ready['observer_identity'],packs=ready['packs'],
        contexts=ready['contexts'],observation_identity=ready['observation_identity'],
        evaluator_sources=m['evaluator_sources'],input_reuse_ready=old['input_reuse_ready'],
        authority=member(ROOT/ENVELOPE),tracking=dict(env_file=old['tracking']['env_file']),
        run_instance=dict(attempt='native-r1'),
        resources=dict(project_cap=2,task_cap=2,gpu=1,cpu=8,host_mib=65536,hard_host_mib=183296,
            wall='2-00:00:00',collector_cpu=8,collector_host_mib=24576,collector_wall='02:00:00',
            reserve_bytes=8*1024**3,ETA='NOT_MEASURED; wall is request only'),
        native=dict(root=str(native_source),closure=closure,hparams=yaml,stats_dir=old['stats_root'],
            logical_model_name='gpt2-xl',device=0,projector=m['projector'],effective_hparams=parsed,
            context_reuse=dict(contexts=ready['contexts_member'],ready=old['input_reuse_ready'],
                source_members=m['context_sources'],runtime=reuse['runtime'],
                model_asset_identity=m['model_asset_identity'],ordered_ids_sha256=ORDERED_SHA,
                cold_W0_H0=m['cold_W0_H0'])),
        noCP=True,exact_resume='NOT_AVAILABLE',z_disk_cache=False,
        native_provenance=dict(HEAD=subprocess.run(['git','rev-parse','HEAD'],cwd=native,capture_output=True,text=True,check=True).stdout.strip(),
            dirty_diff_sha256=digest(diff),upstream_pristine=False))
    from .native import bind_context
    for module in (modules.memit,modules.alphaedit):bind_context(modules,c,module)
    out.mkdir(parents=True)
    write(out/'native-local-diff.json',dict(diff=diff,sha256=digest(diff),원source수정=False))
    write(out/'config.json',c)
    write(out/'preparation.json',dict(상태='CPU_METADATA_BOUND_NOT_GPU_PASS',authority=c['authority'],
        input_ready=old['input_reuse_ready'],W0='EXACT_READ_ONLY_REFERENCE_MODEL_STATE_PROJECTION',
        context='native cold get_context_templates; RNG restored',model_loads=0,native_apply=0,
        covariance_rebuild=0,projector_rebuild=0,checkpoint_saved=False,큰payload검증='기존 SHA + 현재 size/inode/mtime seal',config=member(out/'config.json')))
    return out/'config.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    print(prepare(a.out.resolve(),a.attempt.resolve()))
if __name__=='__main__':main()

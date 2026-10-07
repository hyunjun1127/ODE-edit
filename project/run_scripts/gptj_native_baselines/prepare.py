"""CPU asset/native pinning only; no models, fits, projector or C0 generation."""
import gc
import json
import shutil
import subprocess
from dataclasses import asdict
import numpy as np
import torch
from .common import *
from .native import NATIVE_FILES,load_native,closure
from scripts.fixed_counterfact import load_prefix

def stat_seal(row):
    s=Path(row['path']).stat()
    require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT_CHANGED')
    return row

def prepare():
    authority();out=LOCAL/'preparation-r1';require(not out.exists(),'CREATE_ONCE')
    old=json.loads(Path('/mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0-cohort-curves/preparation-r1/config.json').read_text())
    for row in old['assets']+old['runtime']['members']+[old['observer_manifest'],old['input_source']]:stat_seal(row)
    records=load_prefix(Path(old['dataset']),2000);list(batches(records))
    source=Path('/mnt/raid5/janghj/EasyEdit/easyeditor');private=out/'native-source'
    private.mkdir(parents=True)
    for relative in NATIVE_FILES:
        p=source/relative;q=private/relative
        require(p.is_file() and not p.is_symlink(),'NATIVE_REGULAR');q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,q);require(sha(p)==sha(q),'PRIVATE_COPY_EXACT')
    modules=load_native(private);members=closure(modules)
    yaml={};parsed={};easy=source.parent
    for writer in ('memit','alphaedit'):
        src=easy/'hparams'/('MEMIT' if writer=='memit' else 'AlphaEdit')/'gpt-j-6B.yaml'
        dst=out/(writer+'.yaml');shutil.copyfile(src,dst);require(sha(src)==sha(dst),'YAML_EXACT')
        yaml[writer]=member(dst)
        cls=modules.memit.MEMITHyperParams if writer=='memit' else modules.alphaedit.AlphaEditHyperParams
        hp=cls.from_hparams(str(dst));parsed[writer]=asdict(hp)
        require(hp.layers==list(LAYERS) and (hp.v_lr,hp.v_num_grad_steps,hp.v_loss_layer,hp.v_weight_decay,hp.clamp_norm_factor,hp.kl_factor)==(.5,25,27,.5,.75,.0625),'NATIVE_YAML')
        require(hp.mom2_update_weight==15000 if writer=='memit' else hp.L2==10 and hp.nullspace_threshold==.02,'NATIVE_COEFFICIENT')
    stats_dir=easy/'examples/data/stats';stats=[];stats_checks=[]
    torch.set_num_threads(2)
    for layer in LAYERS:
        p=stats_dir/'gpt-j-6b/wikipedia_stats'/f'transformer.h.{layer}.mlp.fc_out_float32_mom2_100000.npz'
        require(p.is_file(),'LAYER_STATS_MISSING')
        with np.load(p,allow_pickle=False) as z:
            require('mom2.mom2' in z.files and 'mom2.count' in z.files,'MOM2_KEYS')
            a=z['mom2.mom2'];count=float(z['mom2.count'])
            require(a.shape==(16384,16384) and a.dtype==np.float32 and count>0,'MOM2_SCHEMA')
            require(all(np.isfinite(a[i:i+256]).all() for i in range(0,16384,256)),'MOM2_FINITE')
            stats_checks.append(dict(layer=layer,shape=list(a.shape),dtype=str(a.dtype),count=count,
                native_moment='stored mom2 / count; no transformed cache saved'))
            del a
        stats.append(member(p));gc.collect()
    projector=easy/'examples/null_space_project_gpt-j-6b.pt'
    require(projector.is_file(),'SIX_LAYER_PROJECTOR_MISSING')
    P=torch.load(projector,map_location='cpu',mmap=True,weights_only=True)
    require(isinstance(P,torch.Tensor) and P.shape==(6,16384,16384) and P.dtype==torch.float32,'SIX_LAYER_PROJECTOR_SCHEMA')
    require(all(bool(torch.isfinite(P[j,i:i+256]).all()) for j in range(6) for i in range(0,16384,256)),'P_FINITE')
    slots=[dict(slot=j,physical_layer=l,shape=list(P[j].shape)) for j,l in enumerate(LAYERS)]
    del P;gc.collect();project=member(projector)
    # Model file mmap reads only selected six projections for actual cold binding.
    weight=next(r for r in old['assets'] if r['snapshot_path'].endswith('/pytorch_model.bin'))
    sd=torch.load(weight['path'],map_location='cpu',mmap=True,weights_only=True)
    cold={}
    for l in LAYERS:
        t=sd[f'transformer.h.{l}.mlp.fc_out.weight']
        require(t.shape==(4096,16384),'FC_OUT_LAYOUT')
        cold[str(l)]=tensor_sha(t.float())
    del sd;gc.collect()
    evaluator=['project/run_scripts/jlz_price_gptj/scores.py','project/run_scripts/jlz_realization/inputs.py',
        'project/run_scripts/jlz_realization/observe.py','project/run_scripts/jlz_interference_l1/comparison_bridge.py']
    c=dict(instruction_id=NONCE,task_id=TASK,seed=20261002,stream=old['input_source']['path'],model=old['model'],
        model_revision=old['revision'],model_assets=old['assets'],assets=old['assets']+stats+[project],
        runtime=old['runtime'],ordered_ids_sha256=ORDERED_SHA,cold_W=cold,observer_identity=old['observer_manifest'],
        evaluator_sources=evaluator,authority=member(ROOT/ENVELOPE),tracking=dict(env_file=old['tracking_env']),
        W0_reference='/mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0-cohort-curves/attempt-r1',
        resources=dict(cpu=6,gpu=1,host_mib=59392,wall='2-00:00:00',collector_cpu=6,collector_host_mib=24576,
            collector_wall='04:00:00',project_cap=2,task_cap=2,reserve_bytes=8*1024**3,wall_is_eta=False,
            host_budget='model load23GiB + Alpha P6/H6/rollbackH6/W1.5 + transient4 <58GiB; MEMIT C0six6GiB',
            gpu_budget='FP32 model22.55GiB + six weights copy1.5 + native fit activations / FP64 dense solve; no dtype reduction'),
        native=dict(root=str(private),closure=members,hparams=yaml,stats_dir=str(stats_dir),logical_model_name='gpt-j-6b',
            device=0,projector=project,effective_hparams=parsed,context_reuse=None),
        native_provenance=dict(original_root=str(source),original_members=[dict(relative=r,**member(source/r)) for r in NATIVE_FILES],
            private_diff='NONE: current native source already supports GPTJ cache/P initialization',
            original_HEAD=subprocess.run(['git','rev-parse','HEAD'],cwd=easy,check=True,capture_output=True,text=True).stdout.strip()),
        noCP=True,z_disk_cache=False,exact_resume='NOT_AVAILABLE')
    write(out/'asset-checks.json',dict(stats=stats_checks,projector_slots=slots,model_loads=0,GPU=0,
        P_slot_binding='native YAML ordered L3..8 plus existing six-slot GPTJ projector; not five-slot ours',
        P_spectral_recompute=False,finite=True,native_source_changed=False))
    write(out/'config.json',c)
    print(json.dumps(dict(config=str(out/'config.json'),source_members=len(members),slots=slots,model_loaded=False)))

if __name__=='__main__':prepare()

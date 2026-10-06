"""CPU-only native parser, exact assets and immutable six-cell binding."""
import argparse,ast,gc,hashlib,importlib,json,os,sys,types
from dataclasses import asdict
from .common import *
from .profile import arm_profile
from .storage import storage_plan,guard
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding
from project.run_scripts.jlz_interference_l1.prepare import verify_schedule
from scripts.fixed_counterfact import load_prefix

EASY=Path('/mnt/raid5/janghj/EasyEdit')
PYTHON=str(EASY/'.venv/bin/python')
STREAM=Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')

def native_hparams(writer):
    """Execute the actual native constructor/parser, excluding only package import."""
    package='memit' if writer=='memit' else 'alphaedit'
    filename='memit_hparams.py' if writer=='memit' else 'AlphaEdit_hparams.py'
    clsname='MEMITHyperParams' if writer=='memit' else 'AlphaEditHyperParams'
    base=EASY/'easyeditor/util/hparams.py';source=EASY/'easyeditor/models'/package/filename
    mod=types.ModuleType('gpt2_pinned_'+package);sys.modules[mod.__name__]=mod
    exec(compile(base.read_text(),str(base),'exec'),mod.__dict__)
    tree=ast.parse(source.read_text())
    tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.level)]
    exec(compile(tree,str(source),'exec'),mod.__dict__)
    yaml=EASY/'hparams'/('MEMIT' if writer=='memit' else 'AlphaEdit')/'gpt2-xl.yaml'
    hp=getattr(mod,clsname).from_hparams(str(yaml))
    require((hp.layers,hp.v_lr,hp.v_num_grad_steps,hp.v_loss_layer,hp.v_weight_decay,hp.kl_factor,
        hp.clamp_norm_factor,hp.mom2_update_weight)==([13,14,15,16,17],.5,20,47,.5,.0625,.75,20000),'NATIVE_FIELDS')
    require(hp.rewrite_module_tmp=='transformer.h.{}.mlp.c_proj' and hp.fact_token=='subject_last','NATIVE_MODULE')
    if writer=='alphaedit':require(hp.L2==10 and hp.nullspace_threshold==.02,'NATIVE_ALPHA_FIELDS')
    return hp,[member(base),member(source),member(yaml)]

def authority():
    manifest=ROOT/DESIGN/'dispatch-package-manifest.json'
    require(sha(manifest)=='4f547b83fad352ddec6ccca782cccc525fd01dc1fc8016a5038ba1d8da0ede08','CANONICAL_MANIFEST')
    require(sha(ROOT/ENVELOPE)=='94c9a0de71ce88821ae2225342bdfb8757c261b8f5f81f3b5f2baa9ff72e263e','CANONICAL_ENVELOPE')
    rows=[member(manifest),member(ROOT/ENVELOPE)]
    for r in json.loads(manifest.read_text())['files']:
        actual=member(ROOT/r['path']);require((actual['bytes'],actual['sha256'])==(r['bytes'],r['sha256']),'CANONICAL_MEMBER')
        rows.append(actual)
    return rows

def prepare(out,attempt,preflight):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    require(not torch.cuda.is_initialized(),'CPU_ONLY')
    require(attempt.parent==LOCAL and not attempt.exists() and not list(LOCAL.glob('*/submitted-*.json')),'NO_DUPLICATE_ATTEMPT')
    members=authority();contract=json.loads((ROOT/DESIGN/'authorized-run-contract.json').read_text())
    model=Path(contract['model']['payload']['path']).parent
    require(os.popen('git -C '+str(model)+' rev-parse HEAD').read().strip()==contract['model']['revision'],'MODEL_REVISION')
    ready=contract['asset_readiness'];assets=[]
    for row in [contract['model']['payload'],ready['READY'],ready['projector']]+[s['asset'] for s in ready['stats']]:
        value=member(row['path']);require((value['bytes'],value['sha256'])==(row['bytes'],row['sha256']),'ASSET_SHA')
        assets.append(value)
    for p in sorted(set(model.glob('*.json'))|set(model.glob('*.txt'))):
        assets.append(member(p))
    stream=member(STREAM);require(stream['sha256']=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1','STREAM_SHA')
    assets.append(stream);records=load_prefix(STREAM.parent,2000);verify_schedule(ROOT/SCHEDULE,records)
    require(digest([r['case_id'] for r in records])==ORDERED_SHA,'FIRST2000')
    mc=json.loads((model/'config.json').read_text())
    require((mc['n_embd'],mc['n_layer'],mc['n_positions'],mc['vocab_size'])==(1600,48,1024,50257),'GPT2_CONFIG')
    W={}
    with safe_open(str(model/'model.safetensors'),framework='pt',device='cpu') as f:
        for l in range(13,18):
            t=f.get_tensor(f'transformer.h.{l}.mlp.c_proj.weight')
            require(t.dtype==torch.float32 and tuple(t.shape)==(6400,1600),'NATIVE_STORAGE')
            W[str(l)]=tensor_sha(t);del t
    h=hashlib.sha256(str(((6400,6400),'torch.float32')).encode());zero=bytes(8*1024**2);remaining=6400*6400*4
    while remaining:
        n=min(remaining,len(zero));h.update(memoryview(zero)[:n]);remaining-=n
    cold=dict(W=W,H={str(l):h.hexdigest() for l in range(13,18)})
    project=next(r for r in assets if r['path']==ready['projector']['path'])
    project=dict(project,physical_layers=list(range(13,18)),shape=[5,6400,6400],dtype='torch.float32',cutoff=.02)
    stats={str(s['layer']):s['asset']['path'] for s in ready['stats']}
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    # Conservative generated prefix allowance, rechecked against actual packs.
    widths=[]
    for r in records:
        rw=r['requested_rewrite']
        widths.append(len(tok.encode(rw['prompt'].format(rw['subject'])+' '+rw['target_new']['str']))+64)
        for p in [rw['prompt'].format(rw['subject'])]+r['paraphrase_prompts']+r['neighborhood_prompts']:
            for label in ('target_new','target_true'):
                require(len(tok.encode(p))+len(tok.encode(' '+rw[label]['str']))-1<=1024,'EVAL_POSITION_LIMIT')
    require(max(widths)<=1024,'NATIVE_POSITION_BOUND')
    T=max(sum(widths[i:i+100])*7 for i in range(0,2000,100));G=1024**3;d=6400;hidden=1600
    host=dict(H_rollback=2*5*d*d*4/G,raw_A_or_C0=5*d*d*8/G,selected_RAM=5*d*hidden*4/G,
        boundaries=5*(d+hidden)*4*T/G,entry_token_metadata=3.,teacher=1.,loader_and_misc=8.,sidecar=4.)
    gpu=dict(model=contract['model']['payload']['bytes']/G,factors=5*d*d*8/G,
        factor_transients=4*d*d*8/G,entry_and_candidate=2*5*d*hidden*4/G,owner_autograd_head_reserve=12.)
    require(sum(host.values())<60 and sum(gpu.values())<44,'RESOURCE_MEMORY_ESTIMATE')
    budget=dict(host_parts_GiB=host,host_peak_GiB=sum(host.values()),GPU_parts_GiB=gpu,GPU_peak_GiB=sum(gpu.values()),
        max_owner_width=max(widths),max_owner_padded_tokens=T,measured=False,extra_model_calls=0)
    runtime=runtime_binding()
    for name in ('transformers.models.gpt2.modeling_gpt2','transformers.pytorch_utils'):
        runtime['source_members'].append(dict(module=name,**member(importlib.import_module(name).__file__)))
    runtime['source_root_sha256']=digest(runtime['source_members'])
    require((str(runtime['torch']),runtime['transformers'])==('2.9.1+cu128','4.57.1'),'PINNED_RUNTIME')
    context_sources=[member(EASY/'easyeditor/util/generate.py'),member(EASY/'easyeditor/models/memit/memit_main.py')]
    native=list(context_sources);models={};mapping={}
    evidence=json.loads((ROOT/DESIGN/'source-evidence.json').read_text())
    for row in evidence['EasyEdit_sources']+evidence['installed_runtime_sources']:
        value=member(row['path']);require((value['bytes'],value['sha256'])==(row['bytes'],row['sha256']),'NATIVE_SOURCE_FROZEN')
        native.append(value)
    evaluator=[member(ROOT/p) for p in ('project/run_scripts/jlz_price_gpt2xl/scores.py',
        'project/run_scripts/jlz_price_gpt2xl/observer.py','project/run_scripts/jlz_realization/inputs.py','project/run_scripts/jlz_pilot/prompts.py')]
    for writer,name in (('memit','MEMIT'),('alphaedit','ALPHAEDIT')):
        hp,refs=native_hparams(writer);native+=refs
        profiles={a:arm_profile(hp,a,writer) for a in ARMS}
        for p in profiles.values():p.update(projector_sha256=project['sha256'],projector_cutoff=.02)
        mapping[writer]=dict(parsed=asdict(hp),effective=profiles,
            Alpha_mom2_update_weight_usage='UNUSED_NATIVE_FIELD_NO_C0_HYBRID' if writer=='alphaedit' else 'lambda_C',
            constructor='actual native from_hparams; only relative HyperParams import replaced by exact source class')
        models[name]=dict(model=str(model),model_asset_identity=digest(assets),assets=assets,stats=stats,
            projector=project,cold_W0_H0=cold,profiles=profiles,resource_binding=budget,packs=[],contexts=None,
            W0_reuse=dict(status='NOT_AVAILABLE_BEFORE_FIRST_CELL'),context_sources=context_sources,evaluator_sources=evaluator)
    native=list({r['path']:r for r in native}.values())
    storage=storage_plan();capacity=guard(LOCAL,storage['reserve_bytes'],10000)
    c=dict(task_id=TASK,instruction_id=NONCE,attempt=str(attempt),run_instance=dict(date='2026-10-07',attempt=attempt.name),
        stream=str(STREAM),seed=20261002,runtime=runtime,models=models,assets=assets,
        native_root=str(EASY),stats_root=str(EASY/'examples/data/stats'),native_hparams=str(EASY/'hparams/MEMIT/gpt2-xl.yaml'),
        ordered_ids_sha256=ORDERED_SHA,native_reference=native,dependency_sources=evaluator,authority_members=members,
        cpu_preflight=member(preflight),storage=storage,tracking=dict(env_file='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env',
            cpu_review=member(preflight)),hparams_policy=mapping,
        settings=dict(B=100,batches=20,requests=2000,arms=list(ARMS),build_cap=20,subject_forward_cap=20,
            subject_backward_cap=19,no_B21=True,save_checkpoints=False,observer_microbatch=2),
        resources=dict(task_cap=2,project_cap=2,gpu=1,cpu=8,host_mib=65536,hard_host_mib=183296,
            collector_cpu=8,collector_host_mib=24576,wall='2-00:00:00',collector_wall='02:00:00',
            reserve_bytes=storage['reserve_bytes'],combined_reserve_bytes=storage['reserve_bytes'],
            free_bytes=capacity['free_bytes'],free_inodes=capacity['free_inodes'],memory_plans={w:budget for w in MODELS},
            ETA='NOT_MEASURED;48h is requested wall only'))
    write(out,c)
    return dict(config=str(out),cells=list(CELLS),memory=budget,reserve_bytes=storage['reserve_bytes'],actual_GPU='NOT_OBSERVED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.out,a.attempt,a.preflight)))

"""CPU asset/token/static preparation, without model instantiation or numerical toys."""
import argparse,ast,copy,gc,hashlib,importlib,json,os,shutil
from pathlib import Path
from .cap_common import *
from .cap_profile import arm_profile
from .cap_storage import storage_plan
from .prepare import verify_schedule
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding,asset_binding
from scripts.fixed_counterfact import load_prefix

OLD=Path('/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/repair-59721')
QMODEL=Path('/data/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
QLOG=Path('/data/janghj/ODE-edit/local/logs/official-layer-realization-debt-sequential-b10x10-v1/off_layer_debt_seq_s4-31593_3.out')
QRESULT=Path('/data/janghj/ODE-edit/local/results/official-layer-realization-debt-sequential-b10x10-v1/sequential-tech-r3/qwen2.5-7b-inst-memit-sequential-b10x10/result.json')

def native_binding(model,contexts,records,out):
    from transformers import AutoTokenizer
    from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    bench=CounterFactAdapter(tok,json.loads(Path(contexts).read_text()));packs=[];bindings=[];identities=[]
    for i in range(20):
        pack=bench.prepare(records[i*100:(i+1)*100]);rows=[];widths=[]
        for j,kind in enumerate(pack['row_kind']):
            n=int(pack['tokens']['attention_mask'][j].sum())
            rows.append(dict(row=j,owner=pack['row_request'][j],kind=kind,lookup=pack['lookup'][j],
                input_ids=pack['tokens']['input_ids'][j,:n].tolist(),targets=pack['targets'][j,:n].tolist()))
        for owner in range(100):
            own=[r for r in rows if r['owner']==owner]
            require(len(own)==pack['n_rw']+1 and sum(r['kind']=='kl' for r in own)==1,'NATIVE_OWNER_ROWS')
            widths.append(max(len(r['input_ids']) for r in own))
        packs.append(dict(batch=i+1,identity=pack['identity'],ids=pack['record_ids']))
        bindings.append(dict(batch=i+1,pack=pack['identity'],row_count=len(rows),
            native_tokens_sha256=digest(rows),owner_padded_tokens=sum(widths)*(pack['n_rw']+1),
            max_owner_width=max(widths),key_prefix_exact=pack['entry_key_prefix_exact']))
    for r in records:
        rw=r['requested_rewrite']
        for kind,prompts in bench.panels(r).items():
            for i,prompt in enumerate(prompts):
                row=dict(case_id=r['case_id'],kind=kind,prompt_index=i,
                    identity=digest([r['case_id'],kind,i,prompt,rw['target_new']['str'],rw['target_true']['str']]))
                for label in ('new','true'):
                    ids,targets=bench.evaluation_ids(prompt,rw['target_'+label]['str'])
                    row[label+'_token_identity']=digest([ids,targets])
                identities.append(row)
    require(len(identities)==26000,'OBSERVER_26000')
    write(out/'native-input-alignment.json',dict(packs=packs))
    write(out/'native-full-input-binding.json',dict(all20=bindings,no_model_load=True,no_forward=True))
    write(out/'observer-identity.json',dict(rows=identities))
    return packs,bindings,identities

def cold_hash(model,d):
    import torch
    from safetensors import safe_open
    index=json.loads((Path(model)/'model.safetensors.index.json').read_text())['weight_map'];W={}
    for l in range(4,9):
        key=f'model.layers.{l}.mlp.down_proj.weight'
        with safe_open(str(Path(model)/index[key]),framework='pt',device='cpu') as f:t=f.get_tensor(key).float()
        W[str(l)]=tensor_sha(t);del t;gc.collect()
    h=hashlib.sha256(str(((d,d),'torch.float32')).encode());zero=bytes(8*1024**2);remaining=d*d*4
    while remaining:
        n=min(remaining,len(zero));h.update(memoryview(zero)[:n]);remaining-=n
    return dict(W=W,H={str(l):h.hexdigest() for l in range(4,9)})

def memory_plan(mc,bindings):
    h,d,n,v=map(mc.__getitem__,('hidden_size','intermediate_size','num_hidden_layers','vocab_size'))
    kv=h*mc['num_key_value_heads']//mc['num_attention_heads'];G=1024**3
    params=2*v*h+n*(2*h*h+2*h*kv+3*h*d+2*h)+h
    T=max(b['owner_padded_tokens'] for b in bindings)
    host=dict(H_and_rollback=2*5*d*d*4/G,raw_A=5*d*d*8/G,selected_RAM_weights=5*h*d*4/G,
        entry_and_BUILD_offload=5*(h+d)*4*T/G,teacher_subject_transients=4.,C0_read_transient=2*d*d*4/G,
        scalar_observer_runtime_reserve=3.,wandb_sidecar_limit=4.)
    # Load phase has no history/factor/entry allocated yet.
    host_peak=max(sum(host.values()),params*4/G+8.)
    gpu=dict(model=params*4/G,five_factors=5*d*d*8/G,factor_transients=2*d*d*8/G,
        entry_and_candidate_weights=2*5*h*d*4/G,one_owner_checkpoint_head_workspace=24.)
    require(host_peak<58,'RESOURCE_BLOCKED_HOST_ESTIMATE');require(sum(gpu.values())<95,'RESOURCE_BLOCKED_VRAM_ESTIMATE')
    return dict(host_parts_GiB=host,host_peak_GiB=host_peak,GPU_parts_GiB=gpu,GPU_peak_GiB=sum(gpu.values()),
        measured=False,all_rows_whole_B=True,scientific_simplification=False,load_phase_separate=True,
        W_B_sidecar_included=True,ETA='NOT_MEASURED;48h requested ceiling')

def prepare(out,attempt,preflight,cpus=8):
    out,attempt,preflight=map(lambda p:Path(p).resolve(),(out,attempt,preflight))
    require(out.is_relative_to(LOCAL) and attempt.parent==LOCAL and not attempt.exists(),'TASK_CREATE_ONCE')
    require(1<=cpus<=8,'CPU_RESOURCE_BINDING')
    manifest=ROOT/DESIGN/'dispatch-package-manifest.json'
    require(sha(manifest)=='b67816867429492ac45301047a0d9cdcfe798446d90c2a74d1fa583af54409d2','MANIFEST_SHA')
    require(sha(ROOT/ENVELOPE)=='8872e143ebf9bd15a3fc1f3fcd2418ea2439410a269c850d4c9c36e124242fa5','ENVELOPE_SHA')
    authority=[member(manifest),member(ROOT/ENVELOPE)]
    for row in json.loads(manifest.read_text())['files']:
        p=ROOT/row['path'];require(not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'PACKAGE_SHA_SIZE')
        authority.append(member(p))
    old=json.loads((OLD/'config.json').read_text());oldlock=json.loads((OLD/'execution.lock.json').read_text())
    require(sha(OLD/'config.json')==oldlock['config_sha256'],'OLD_CONFIG')
    for row in old['native_reference']+old['dependency_sources']:verify(row)
    runtime=runtime_binding()
    runtime['source_members'].append(dict(module='transformers.models.qwen2.modeling_qwen2',
        **member(importlib.import_module('transformers.models.qwen2.modeling_qwen2').__file__)))
    runtime['source_root_sha256']=digest(runtime['source_members'])
    require((runtime['torch'],runtime['transformers'])==(old['runtime']['torch'],old['runtime']['transformers']),'RUNTIME_PINS')
    records=load_prefix(Path(old['stream']).parent,2000);verify_schedule(ROOT/SCHEDULE,records)
    require(digest([r['case_id'] for r in records])==ORDERED_SHA,'FIRST2000_ORDER')
    context=None
    with QLOG.open() as f:
        for line in f:
            if 'Cached context templates' in line and '[[' in line:
                context=ast.literal_eval(line[line.index('[['):]);break
    qr=json.loads(QRESULT.read_text());require(context is not None and digest(context)==qr['warm_state']['context_identity']==
        '6688fe84115305610c00b328df931b0f0f4bc6c8eced70386ef975c58d85a9c3','EXISTING_QWEN_CONTEXT_IDENTITY')
    write(out/'QWEN/contexts.json',context)
    models={};allassets=[];memory={}
    for name in MODELS:
        folder=out/name;model=Path(old['model']) if name=='LLAMA' else QMODEL
        contexts=Path(old['contexts']) if name=='LLAMA' else folder/'contexts.json'
        stats=old['stats'] if name=='LLAMA' else {str(l):f'/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.{l}.mlp.down_proj_float32_mom2_100000.npz' for l in range(4,9)}
        mc=json.loads((model/'config.json').read_text());expected=(4096,14336,128256,32) if name=='LLAMA' else (3584,18944,152064,28)
        require(tuple(mc[k] for k in ('hidden_size','intermediate_size','vocab_size','num_hidden_layers'))==expected,'MODEL_ARCHITECTURE')
        packs,bindings,identities=native_binding(str(model),contexts,records,folder)
        if name=='LLAMA':
            require([(p['identity'],p['ids']) for p in packs]==[(p['identity'],p['ids']) for p in old['packs']],'UNCHANGED_LLAMA_PACKS')
            require(identities==json.loads(verify(old['observer_identity']).read_text())['rows'],'UNCHANGED_LLAMA_OBSERVER_TOKENS')
        paths=[Path(old['stream']),contexts]+list(map(Path,stats.values()))
        index=json.loads((model/'model.safetensors.index.json').read_text())
        paths+=sorted(set(model/x for x in index['weight_map'].values())|set(model.glob('*.json'))|set(model.glob('*.txt')))
        previous={r['path']:r for r in old['assets']}
        assets=[asset_binding(p,previous,member(OLD/'config.json')) for p in paths]
        if name=='QWEN':
            seal=json.loads((ROOT/'agents/server4/p4-hf-consumed-closure-seal.json').read_text())['models']['qwen2.5-7b-inst']
            for row in seal['required_members']:
                a=next(a for a in assets if a['requested_path']==str(model/row['relative_path']))
                require(a['sha256']==row['sha256'] and a['bytes']==row['size'],'QWEN_HF_CANONICAL_CLOSURE')
        cold=old['cold_W0_H0'] if name=='LLAMA' else cold_hash(model,mc['intermediate_size'])
        memory[name]=memory_plan(mc,bindings)
        obsidentity=digest(dict(model=str(model),assets=[(a['requested_path'],a['sha256']) for a in assets],
            rows=digest(identities),observer=sha(ROOT/'project/run_scripts/jlz_realization/observe.py'),
            input_adapter=sha(ROOT/'project/run_scripts/jlz_realization/inputs.py'),microbatch=2,cold=cold))
        reuse=dict(status='NOT_AVAILABLE',reason='No identity-qualified raw yet; own-model prior cell may supply exact W0')
        if name=='LLAMA':
            for path in ('project/run_scripts/jlz_realization/observe.py','project/run_scripts/jlz_realization/inputs.py','project/run_scripts/jlz_interference_l1/observer_io.py'):
                require(sha(ROOT/path)==sha(OLD/'source'/path),'W0_READONLY_EVALUATOR_SOURCE')
            from .w0_reuse import source_chunks
            source=OLD/'PRICE/W0';chunks=source_chunks(source,cold) if (source/'reuse.json').exists() else sorted(source.glob('chunk-*.json'))
            reuse=dict(status='QUALIFIED_EXACT_REUSE',cold_state=cold,chunks=[member(p) for p in chunks],
                summary=member(source/'summary.json'),runtime=member(OLD/'PRICE/runtime.json'),
                observation_identity=obsidentity,source_folder=str(source),qualification='same assets/tokens/cold/evaluator; live device/runtime check before reuse')
        models[name]=dict(model=str(model),contexts=str(contexts),stats=stats,assets=assets,packs=packs,
            native_input_alignment=member(folder/'native-input-alignment.json'),native_full_input_binding=member(folder/'native-full-input-binding.json'),
            observer_identity=member(folder/'observer-identity.json'),observation_identity=obsidentity,cold_W0_H0=cold,
            profiles={arm:arm_profile(old['profile'],arm,name) for arm in ARMS},W0_reuse=reuse,memory_plan=memory[name])
        allassets+=assets
    storage=storage_plan();free=shutil.disk_usage(LOCAL).free;require(free>=storage['reserve_bytes'],'RESOURCE_BLOCKED_STORAGE')
    c={k:copy.deepcopy(old[k]) for k in ('stream','native_root','stats_root','native_hparams','settings')}
    c['settings']['arms']=list(ARMS)
    c.update(instruction_id=NONCE,task_id=TASK,attempt=str(attempt),run_instance=dict(date='2026-10-07',attempt=attempt.name),
        seed=20261002,runtime=runtime,models=models,assets=allassets,authority_members=authority,cpu_preflight=member(preflight),
        native_reference=old['native_reference'],dependency_sources=old['dependency_sources'],ordered_ids_sha256=ORDERED_SHA,
        tracking=dict(env_file=str(LOCAL/'tracking.env')),storage=storage,
        resources=dict(task_cap=2,project_cap=2,gpu=1,cpu=cpus,host_mib=59392,hard_host_mib=60416,
            collector_cpu=8,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00',
            reserve_bytes=storage['reserve_bytes'],free_bytes=free,free_inodes=os.statvfs(LOCAL).f_favail,
            memory_plans=memory,ETA='NOT_MEASURED; requested finite upper bound',effective_user_cap_override=2))
    write(out/'configuration.json',c)
    write(out/'binding.json',dict(task=TASK,canonical=authority,models={m:dict(packs=20,observer_rows=26000,
        memory=memory[m],context=member(models[m]['contexts'])) for m in MODELS},Qwen_context_provenance=dict(log=member(QLOG),result=member(QRESULT)),
        no_model_instantiation=True,no_GPU=True,no_toy=True,no_new_forward=True,actual_B1='NOT_OBSERVED',storage=storage))
    return dict(config=str(out/'configuration.json'),cells=list(CELLS),actual_B1='NOT_OBSERVED')

def finalize(config,preflight,out):
    """Bind final reviewed source without redoing token/model asset preparation."""
    c=json.loads(Path(config).read_text());p=json.loads(Path(preflight).read_text())
    require(c['task_id']==p['task']==TASK and p['passed'] and p['numeric_tests']==0,'FINAL_STATIC_AUDIT')
    for row in p['source']:verify(row)
    for m in MODELS:
        mc=c['models'][m]
        for field in ('native_input_alignment','native_full_input_binding','observer_identity'):verify(mc[field])
        for row in mc['assets']:
            st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_FRESH_STAT')
    # Every inherited scientific Python byte remains identical to the completed
    # PRICE archive; fresh adapter and orchestration live only in cap_* files.
    oldlock=json.loads((OLD/'execution.lock.json').read_text());bound=[]
    for row in oldlock['source_members']:
        rel=Path(row['path']).relative_to(OLD/'source')
        if str(rel).endswith('.py') and str(rel).startswith('project/run_scripts/'):
            require(sha(ROOT/rel)==row['sha256'],'READONLY_INHERITED_SOURCE:'+str(rel));bound.append(str(rel))
    expected_stats=('935af1e9a7c5fe690471c668af225b882b3ae49039435f7143cd3c136c1f049b',
        '7e4730511077be9b7370f29e94036d3ca08e7e0f0023395341e7496e16340519',
        'd47f4bb2454555740c6bc46ab7f894e170cf6f5fe4bc6c26a3fc93179e18dc2a',
        '61526068f1e1283d2e8554b514f9c7ef726957a8582b62c181f393e5589c0b0d',
        'f5b00a555c9d860af1732ae3c48b1a04ebe8e3427873081fb0cb4cc8b5b30d46')
    for layer,expected in zip(range(4,9),expected_stats):
        a=next(a for a in c['models']['QWEN']['assets'] if a['requested_path']==c['models']['QWEN']['stats'][str(layer)])
        require(a['sha256']==expected,'QWEN_NATIVE_C0_SHA')
    c['cpu_preflight']=member(preflight)
    c['readonly_closure']=dict(source='0415aba3c160170d306be8196792f198dad4d122',unchanged_python=bound,
        new_source='cap_* task modules only',shared_modules_changed=[])
    write(out,c);return dict(config=str(out),source_review='OWNER_STATIC_ONLY',actual_B1='NOT_OBSERVED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--attempt');p.add_argument('--preflight',required=True);p.add_argument('--cpus',type=int,default=8);p.add_argument('--finalize-config')
    a=p.parse_args();print(json.dumps(finalize(a.finalize_config,a.preflight,a.out) if a.finalize_config else prepare(a.out,a.attempt,a.preflight,a.cpus)))

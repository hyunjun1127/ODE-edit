"""Standalone cold FP32 author numeric settings/current-z/basic EOT.

Host owners execute; no scheduler, historical checkpoint, or selection input.
Only model/edit precision changes. Native FP64 solve and FP32 history stay native.
"""
import argparse
import fcntl
import importlib
import inspect
import os
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import author_basic_eot_benchmark as author
from official.diagnostics.fe_qwen_current import basic_eot_benchmark as common
from official.diagnostics.fe_qwen_current import failure_eot_prefix as parent
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.diagnostics.fe_qwen_current import observer
from official.diagnostics.fe_qwen_current.patch import native_source
from official.diagnostics.fe_native import sequential_z
from official.diagnostics.fe_qwen_current.storage import admit_remaining,state_digest,finite_tensor,tensor_sha
from official.diagnostics.fe_qwen_current.benchmark import native_records
from official.evaluation import zsre_paper
from official.evaluation.zsre_query_parity import compare_queries
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,verify,member,require
from official.tracking import init
from official.tracking.schema import bind_job_identity,job_identity

SCHEMA='author-basic-eot-fp32-v1'
PROFILE='FE_AUTHOR_HPARAMS_CURRENT_Z_BASIC_EOT_FP32'
STATUS='AUTHOR_BASIC_EOT_FP32_W20_COMPLETE'
STREAMS={'cf':(4145654,'66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37'),
 'zsre':(1490821,'f42ee4bc6e98b1133e48dc81102201ccd85ccc917a97160ee3c6a0f3a38f378c')}
REFERENCES={'cf':'820f241a015b4397','zsre':'a2615463bd5d4b75'}
ADAPTER_SHA='3b043010cd576fbd9843ca34ec2e4514eea45e2fd68dc09005ca1d7b24399755'
NATIVE_SHA={'algs/memit/compute_z.py':'2ecb196ca908a88aa5d4690b24e64a2ad3645c2420e031d1650c1bd738cef685',
 'algs/memit/compute_ks.py':'257abf6c1444c67384fc97f0bde935f368827eeb96b803bae879d0892d535879',
 'locate_edit_utils/repr_tools.py':'cb8ae7c5a62c6dfdcb30c5e1281d4a61db77f552f04e58ef7fb946944de3927e',
 'util/nethook.py':'5f122abb29249b73ee728c3d03927b404e20690bfb4564c72d04f934d4e779e0'}


def audit_settings(cfg,profiles):
    actual=OmegaConf.to_container(cfg,resolve=True)
    require(actual['model_dtype']=='float32','COLD_FP32_REQUIRED')
    # Reuse ALL native numeric/mode guards without weakening the frozen BF16 API.
    check=OmegaConf.create(actual);check.model_dtype='bfloat16';author.audit_settings(check,profiles)
    require(cfg.diagnostic_z_reference=='batch_entry' and cfg.gpu==0,'CURRENT_ENTRY_GPU0')
    return actual


def payload_bound(model_config,layers=5):
    return layers*(int(model_config['hidden_size'])*int(model_config['intermediate_size'])*4
        +int(model_config['intermediate_size'])**2*4)+16*1024**2


def validate(b):
    require(b['schema']==SCHEMA and b['dataset'] in STREAMS,'FP32_SCOPE')
    e=b['execution'];require(e['server'] in ('server4','rent') and len(e['source_commit'])==40
        and all(c in '0123456789abcdef' for c in e['source_commit']) and str(e['job_id']) and e['attempt']
        and e['source_run_id']==REFERENCES[b['dataset']],'EXECUTION')
    require(len(e['admission_id'])==64 and all(c in '0123456789abcdef' for c in e['admission_id']),'ADMISSION_SLOT_ID')
    require(member(verify(b['source']))['sha256']==member(__file__)['sha256'] and member(eot.__file__)['sha256']==ADAPTER_SHA,'SOURCE_ADAPTER')
    for m in b['official_members']:verify(m)
    required=[author.__file__,common.__file__,parent.__file__,eot.__file__,observer.__file__,sequential_z.__file__]
    supplied={Path(m['path']).resolve():m['sha256'] for m in b['official_members']}
    require(all(supplied.get(Path(p).resolve())==member(p)['sha256'] for p in required),'OFFICIAL_RUNTIME_MEMBERS')
    n=b['native'];require(n['author_commit']==author.AUTHOR,'AUTHOR_PIN');root=Path(n['path']).resolve()
    members={Path(verify(m)).resolve():m for m in n['members']}
    require(all(p.is_relative_to(root) for p in members),'NATIVE_MEMBER_ROOT')
    original=Path(verify(n['original_fe'])).read_text()
    require((root/'algs/memit/memit_main.py').resolve() in members
        and (root/'algs/memit/memit_main.py').read_text()==native_source(original),'EXACT_NATIVE_WRITER_AND_CALLBACKS')
    require((root/'algs/memit/native_sequential_z.py').resolve() in members
        and (root/'algs/memit/native_sequential_z.py').read_bytes()==Path(sequential_z.__file__).read_bytes(),'EXACT_CURRENT_TARGET_SUPPLIER')
    for relative,sha in NATIVE_SHA.items():
        require((root/relative).resolve() in members and members[(root/relative).resolve()]['sha256']==sha,'PINNED_NATIVE_DEPENDENCY_'+relative)
    cfg=OmegaConf.create(read(verify(b['native_config'])));resolved=audit_settings(cfg,author.author_profiles(b))
    m=b['main_stream'];require((m['bytes'],m['sha256'])==STREAMS[b['dataset']],'EXACT_ORIGINAL_STREAM')
    rows=common.main_stream(b,[]) # content relocation; no old host/CP/heldout gate
    for m in b['model']['members']:verify(m)
    config=read(verify(b['model']['config']));require(config['model_type']=='qwen2','QWEN_MODEL')
    snapshot=Path(b['model']['snapshot']);model_members={Path(m['path']).resolve() for m in b['model']['members']}
    require(Path(b['model']['config']['path']).resolve() in model_members and (snapshot/'config.json').resolve()==Path(b['model']['config']['path']).resolve(),'MODEL_CONFIG_ALIAS')
    index=snapshot/'model.safetensors.index.json'
    require(index.resolve() in model_members and all((snapshot/p).resolve() in model_members for p in set(read(index)['weight_map'].values())),'ALL_PRETRAINED_SHARDS')
    require((snapshot/'tokenizer.json').resolve() in model_members and (snapshot/'tokenizer_config.json').resolve() in model_members,'TOKENIZER_MEMBERS')
    for m in b['C0']:
        verify(m['original']);require(Path(m['link']).resolve()==Path(m['original']['path']).resolve(),'C0_READONLY_ALIAS')
    for l in cfg.llms.layers:
        suffix=('-'+cfg.cache_filename_suffix) if cfg.cache_filename_suffix else ''
        path=Path(cfg.cache_dir)/'stats'/cfg.llms.name.replace('/','-')/f'layer-{l}{suffix}.npz'
        require(path.is_file() and any(path.resolve()==Path(m['original']['path']).resolve() for m in b['C0']),'NO_C0_REBUILD')
    s=b['storage'];minimum=(8 if e['server']=='server4' else 32)*1024**3
    require(type(b['runtime']['allow_tf32']) is bool and type(b['runtime']['cudnn_allow_tf32']) is bool,'EXPLICIT_TF32_FLAGS')
    require(s['reserve_bytes']>=minimum and s['payload_bound']>=payload_bound(config)
        and s['raw_margin_bytes']>=0 and s['admitted'],'FP32_STORAGE_POLICY')
    return cfg,resolved,rows


def admission(out,b):
    s=b['storage'];lock=Path(s['lock_path']);lock.parent.mkdir(parents=True,exist_ok=True)
    require(not lock.is_symlink() and lock.parent.stat().st_dev==out.stat().st_dev,'HOST_LOCK_FS')
    configs=s['admitted'];require(len({str(Path(c['output']).resolve()) for c in configs})==len(configs),'UNIQUE_ADMITTED_OUTPUTS')
    require(any(Path(c['output']).resolve()==out.resolve() and c['config_sha256']==b['execution']['admission_id'] for c in configs),'OWN_STORAGE_ADMISSION')
    with lock.open('a+b') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:
            st=os.statvfs(out)
            return admit_remaining(dict(planned_latest_count=len(configs),checkpoint_payload_bound=s['payload_bound'],
                raw_margin_bytes=s['raw_margin_bytes'],reserve_bytes=s['reserve_bytes']),configs,st.f_bavail*st.f_frsize)
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def tracking(b,env=None):
    values=author.tracking(b,{},env)
    values.update(task_id='fe-qwen-author-basic-eot-fp32-20261012',arm='AUTHOR-'+b['dataset']+'-FP32-L2-0-clamp1-basicEOT',
        baseline='FE-author-hparams-current-z-basicEOT-FP32',evaluation_profile=PROFILE)
    return values


def identity(b,resolved):
    return dict(stage=PROFILE,dataset=b['dataset'],input_policy=common.POLICY,L2=0,clamp_norm_factor=1.,no_mask=True,
        model_dtype='float32',source=b['execution']['source_commit'],job_id=b['execution']['job_id'],
        config_sha256=b['execution']['admission_id'],bindings_sha256=digest(b),stream=b['main_stream'],
        author_commit=author.AUTHOR,author_settings_sha256=digest(resolved),adapter=member(eot.__file__),z_reference='batch_entry')


def observed_write(writer,cfg,model,tok,requests,device,H,targets,batch):
    """Passive exact native frame observation, no extra solve or dtype casts."""
    require(sys.gettrace() is None,'TRACE_BUSY');records={};key_dtypes={}
    def trace(frame,event,arg):
        if frame.f_code is not writer.__code__:return None
        if event=='line':
            v=frame.f_locals;layer=v.get('layer');name=v.get('weight_name')
            if layer in cfg.llms.layers and 'layer_ks' in v and v['layer_ks'].dtype==torch.float32:
                key_dtypes[layer]=str(v['layer_ks'].dtype)
            if layer in cfg.llms.layers and name==cfg.llms.rewrite_module_tmp.format(layer)+'.weight' and 'upd_matrix' in v and layer not in records:
                W=v['weights'][name];D=v['upd_matrix']
                records[layer]=dict(layer=int(layer),weight=str(W.dtype),target=str(targets[layer].dtype),K_native=key_dtypes.get(layer),
                    K_solve=str(v['layer_ks'].dtype),residual_solve=str(v['resid'].dtype),covariance_solve=str(v['cov'].dtype),
                    solve_output=str(D.dtype),history=str(H.dtype),weight_norm=float(W.detach().norm()),update_norm=float(D.detach().norm()),
                    target_norm=float(targets[layer].detach().norm()))
        return trace
    try:
        sys.settrace(trace);writer(cfg,model,tok,requests,device,H,targets)
    finally:sys.settrace(None)
    require(set(records)==set(cfg.llms.layers),'ALL_NATIVE_LAYERS_OBSERVED')
    require(all(r['weight']==r['target']==r['history']==r['K_native']=='torch.float32' and
        r['K_solve']==r['residual_solve']==r['covariance_solve']==r['solve_output']=='torch.float64' for r in records.values()),'NATIVE_DTYPE_BOUNDARIES')
    params=dict(model.named_parameters())
    require(finite_tensor(H) and all(finite_tensor(params[cfg.llms.rewrite_module_tmp.format(l)+'.weight']) for l in cfg.llms.layers),'FINITE_NATIVE_STATE')
    return dict(batch=batch,case_ids=[r['case_id'] for r in requests],layers=list(records.values()),masked_requests=0,additional_solves=0)


class Observer(common.FinalObserver):
    def __init__(self,out,b,cfg,model,tok,native,z,s,stream,resolved):
        self.out=out;self.b=b;self.cfg=cfg;self.stream=stream;self.batch=0;self.raw=[];self.latest=None
        self.adapter=eot.InputAdapter(tok,z.find_fact_lookup_idx);self.fit=self.adapter.fit(z.compute_z)
        self.supply=self.adapter.supplier(s.current_batch_targets);self.writer=self.adapter.writer(native.batch_edit,native.compute_ks);self.z=z
        self.names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
        require(all(p.dtype==torch.float32 for p in model.parameters()),'COLD_ALL_FP32')
        self.nonedited={n:(id(p),p._version) for n,p in model.named_parameters() if n not in self.names}
        self.identity=identity(b,resolved);self.values=tracking(b);actual=job_identity(self.values)
        require(actual['execution_backend']!='local' and str(actual['job_id'])==str(b['execution']['job_id']),'ACTUAL_HOST_JOB')
        self.tracker=init(env_file=b['tracking_env_file'],spool=out/'tracking',config=self.values)
        write_new(out/'runtime.json',dict(identity=self.identity,native_config=resolved,tracking=self.values,profile=PROFILE,
            runtime=b['runtime'],attention_backend=model.config._attn_implementation,argv=sys.argv,
            model_dtype=str(next(model.parameters()).dtype),tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,device=torch.cuda.get_device_name(),
            pure_unmodified_author_repo=False,cold_pretrained=True,checkpoint_restore=False))
    def write(self,writer,cfg,model,tok,requests,device,H,targets,batch):
        require(batch==self.batch+1,'BATCH_ORDER')
        value=observed_write(self.writer,cfg,model,tok,requests,device,H,targets,batch)
        self.batch=batch;write_new(self.out/'writers'/f'B{batch:02d}.json',value)


def run(bindings,output):
    b=read(bindings);cfg,resolved,stream=validate(b);out=Path(output)
    require(not out.exists() and not out.is_symlink(),'NEW_OUTPUT')
    actual_job=job_identity(tracking(b));require(actual_job['execution_backend']!='local'
        and str(actual_job['job_id'])==str(b['execution']['job_id']),'ACTUAL_JOB_BEFORE_MODEL')
    import transformers,numpy,tokenizers
    actual=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,tokenizers=tokenizers.__version__)
    require(all(b['runtime'][k]==v for k,v in actual.items()),'BOUND_RUNTIME')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'HOST_GPU1')
    torch.backends.cuda.matmul.allow_tf32=b['runtime']['allow_tf32'];torch.backends.cudnn.allow_tf32=b['runtime']['cudnn_allow_tf32']
    out.mkdir(parents=True);write_new(out/'admission.json',admission(out,b));write_new(out/'author-settings.json',dict(resolved=resolved,author_yaml=b['author_yaml'],precision_change_only=True))
    snap=b['model']['snapshot'];tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    require(tok.pad_token_id==151643 and tok.padding_side=='right','NATIVE_TOKENIZER')
    full=None
    if b['dataset']=='zsre':
        parity=compare_queries(tok,stream,model_family='qwen25');require(parity['status']=='PASS_CPU_QUERY_ONLY','PUBLIC_QUERY_PARITY')
        full=zsre_paper.build_queries(tok,stream,model_family='qwen25');write_new(out/'query-parity.json',parity)
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.float32,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==b['runtime']['attention_backend'],'SAME_BACKEND_SELECTION')
    sys.path.insert(0,b['native']['path']);native=importlib.import_module('algs.memit.memit_main');z=importlib.import_module('algs.memit.compute_z');s=importlib.import_module('algs.memit.native_sequential_z')
    for mod in (native,z,s):require(Path(mod.__file__).resolve().is_relative_to(Path(b['native']['path']).resolve()),'NATIVE_IMPORT')
    require(native.CONTEXT_TEMPLATES_CACHE is None and native.covs==[] and observer._STATE is None,'COLD_NATIVE_H0')
    obs=Observer(out,b,cfg,model,tok,native,z,s,stream,resolved);obs.full_queries=full;status='TECHNICAL_FAILURE'
    try:
        with patch.object(observer,'snapshot',obs.snapshot),patch.object(observer,'observed_targets',obs.targets),patch.object(observer,'observed_write',obs.write):
            native.apply_memit_to_model(model,tok,native_records(stream),cfg)
        require(obs.batch==20 and obs.latest['batch']==20 and audit_settings(cfg,author.author_profiles(b))==resolved,'COMPLETE_UNCHANGED_CONFIG')
        status=STATUS;write_new(out/'benchmark-result.json',dict(status=status,dataset=b['dataset'],identity=obs.identity,bindings=member(bindings),source=member(__file__),
            edits=2000,batches=20,checkpoint=obs.latest,pointer=member(out/'checkpoints/latest.json'),runtime=member(out/'runtime.json'),
            commits=[member(out/'commits'/f'B{i:02d}.json') for i in range(1,21)],evaluations=obs.raw,recovery_claim=False))
    except common.cont.QualityEarlyStop:status='QUALITY_EARLY_STOP';raise
    finally:write_new(out/'tracking-finish.json',dict(status=status,finish=obs.tracker.finish(exit_code=0 if status==STATUS else 2 if status=='QUALITY_EARLY_STOP' else 1,timeout=45)))
    return member(out/'benchmark-result.json')


def audit_checkpoint(proof,stream,resolved):
    cp=proof['checkpoint'];ptr=read(verify(proof['pointer']));require(ptr==cp and cp['batch']==20,'W20_POINTER')
    payload=torch.load(verify(cp),map_location='cpu',weights_only=False,mmap=True)
    require(payload['batch']==20 and payload['identity']==proof['identity']==cp['identity'],'CP_IDENTITY')
    cur=payload['cursor'];require(cur['edits']==2000 and cur['case_ids']==[r['case_id'] for r in stream] and cur['native_config']==resolved
        and cur['input_adapter']['sha256']==ADAPTER_SHA,'CP_CURSOR_CONFIG')
    cfg=OmegaConf.create(resolved);names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
    require(set(payload['weights'])==names and all(w.dtype==torch.float32 and finite_tensor(w) for w in payload['weights'].values()),'CP_FP32_NOT_BF16')
    H=payload['cache_c'];width=next(iter(payload['weights'].values())).shape[1]
    require(H.dtype==torch.float32 and H.shape==(5,width,width) and H.is_contiguous() and finite_tensor(H),'NATIVE_FP32_H')
    require(payload['integrity_sha256']==state_digest({k:v for k,v in payload.items() if k!='integrity_sha256'}),'CP_INTEGRITY')


def collect(result,output):
    proof=read(result);require(proof['status']==STATUS and proof['edits']==2000 and proof['batches']==20,'ACTUAL_W20')
    require(proof['source']['sha256']==member(__file__)['sha256'],'FP32_SOURCE')
    b=read(verify(proof['bindings']));cfg,resolved,stream=validate(b)
    require(proof['identity']==identity(b,resolved) and proof['dataset']==b['dataset'],'RESULT_BINDING')
    runtime=read(verify(proof['runtime']));env={('SLURM_JOB_ID' if b['execution']['server']=='server4' else 'ODEEDIT_RENT_JOB_ID'):str(b['execution']['job_id'])}
    require(runtime['identity']==proof['identity'] and runtime['native_config']==resolved and runtime['tracking']==tracking(b,env)
        and runtime['model_dtype']=='torch.float32' and runtime['runtime']==b['runtime'] and runtime['attention_backend']==b['runtime']['attention_backend']
        and runtime['tf32']==b['runtime']['allow_tf32'] and runtime['cudnn_allow_tf32']==b['runtime']['cudnn_allow_tf32'],'ACTUAL_FP32_RUNTIME')
    import transformers
    tok=transformers.AutoTokenizer.from_pretrained(b['model']['snapshot'],trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    reduced=common.audit_chain(proof,proof['commits'],stream,b['dataset'],tok,list(range(1,21)),('W0_first2000',2000))
    audit_checkpoint(proof,stream,resolved)
    report=dict(status='AUTHOR_BASIC_EOT_FP32_W20_CPU_AUDITED',result=member(result),dataset=b['dataset'],profile=PROFILE,
        W0=reduced[(0,'W0_first2000')],W20=reduced[(20,'all_seen/post')],commits=20,endpoints=25,recovery_claim=False)
    write_new(Path(output),report);return member(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run');r.add_argument('bindings');r.add_argument('output')
    r=sub.add_parser('collect');r.add_argument('result');r.add_argument('output');args=p.parse_args()
    print(run(args.bindings,args.output) if args.command=='run' else collect(args.result,args.output))

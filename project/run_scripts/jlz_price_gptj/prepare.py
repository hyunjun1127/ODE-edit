"""CPU metadata/token-budget binding. No GPU, fit or numerical toy."""
import argparse,copy,gc,hashlib,importlib,json,os,shutil
from .common import *
from .storage import storage_plan
from .profile import arm_profile
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_interference_l1.prepare import verify_schedule
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding

PREVIOUS=Path('/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/logging-repair-20261007')
DOWNLOAD=Path('/var/tmp/janghj-price-gptj-model-Wv5YCJTj/download-receipt.json')
EASY=Path('/data/janghj/EasyEdit')

def prepare(out,attempt,preflight):
    import torch,yaml
    from transformers import AutoTokenizer
    require(not torch.cuda.is_initialized(),'CPU_PREP_ONLY')
    old=json.loads((PREVIOUS/'config.json').read_text())
    oldlock=json.loads((PREVIOUS/'execution.lock.json').read_text())
    require(oldlock['source_commit']=='2440e548be39e55a99747d7847d21a88df419b93'
        and sha(PREVIOUS/'config.json')==oldlock['config_sha256'],'PREDECESSOR_SOURCE')
    receipt=json.loads(DOWNLOAD.read_text());model=Path(receipt['snapshot'])
    require(receipt['revision']=='47e169305d2e8376be1d31e765533382721b2cc1','GPTJ_REVISION')
    assets=[]
    for row in receipt['members']:
        m=member(row['path']);require((m['bytes'],m['sha256'])==(row['bytes'],row['sha256']),'GPTJ_DOWNLOAD_SHA')
        assets.append(dict(m,requested_path=row['path']))
    mc=json.loads((model/'config.json').read_text())
    require((mc['model_type'],mc['n_embd'],mc['n_layer'],mc['vocab_size'])==('gptj',4096,28,50400),'GPTJ_CONFIG')
    hp={w:yaml.safe_load((EASY/'hparams'/w/'gpt-j-6B.yaml').read_text()) for w in ('MEMIT','AlphaEdit')}
    for p in hp.values():
        require(p['rewrite_module_tmp']=='transformer.h.{}.mlp.fc_out' and p['v_loss_layer']==27
            and p['layers']==list(range(3,9)) and p['mom2_n_samples']==100000,'NATIVE_HPARAM_MAPPING')
    project=member(EASY/'examples/null_space_project_gpt-j-6b.pt')
    from .alpha_geometry import projector
    N=projector(project['path']);require(tuple(N.shape)==(6,16384,16384),'PROJECTOR_PHYSICAL_LAYER_SCHEMA')
    project.update(shape=list(N.shape),dtype=str(N.dtype),physical_layers=list(range(3,9)),cutoff=.02)
    del N;projector.cache_clear();assets.append(project)
    stats={str(l):str(EASY/f'examples/data/stats/gpt-j-6b/wikipedia_stats/transformer.h.{l}.mlp.fc_out_float32_mom2_100000.npz') for l in range(3,9)}
    assets.extend(member(p) for p in stats.values());assets.append(member(old['stream']))
    # mmap base checkpoint only; no model object/forward. Hash selected cold
    # weights; no saved copy or new tensor checkpoint is created.
    sd=torch.load(model/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    W={};params=0
    for t in sd.values():params+=t.numel()
    for l in range(3,9):
        t=sd[f'transformer.h.{l}.mlp.fc_out.weight'];require(tuple(t.shape)==(4096,16384),'NATIVE_WEIGHT_SHAPE')
        W[str(l)]=tensor_sha(t.float())
    del sd;gc.collect()
    h=hashlib.sha256(str(((16384,16384),'torch.float32')).encode());z=bytes(8*1024**2)
    for _ in range(128):h.update(z)
    cold=dict(W=W,H={str(l):h.hexdigest() for l in range(3,9)})
    runtime=runtime_binding();runtime['source_members'].append(dict(module='transformers.models.gptj.modeling_gptj',
        **member(importlib.import_module('transformers.models.gptj.modeling_gptj').__file__)))
    runtime['source_root_sha256']=digest(runtime['source_members'])
    records=load_prefix(Path(old['stream']).parent,2000);verify_schedule(ROOT/SCHEDULE,records)
    require(digest([r['case_id'] for r in records])==ORDERED_SHA,'ORDERED2000')
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    # Generated prefixes have max10 native tokens. Reserve +32 for punctuation,
    # boundary retokenization and native KL wording. The actual ready pass checks
    # both the bound and actual memory formula before allocating entry buffers.
    widths=[]
    for r in records:
        rw=r['requested_rewrite'];text=rw['prompt'].format(rw['subject'])+' '+rw['target_new']['str']
        widths.append(len(tok.encode(text))+32)
    T=max(sum(widths[i:i+100])*7 for i in range(0,2000,100));G=1024**3;d=16384;hsize=4096
    host=dict(history_and_rollback=2*6*d*d*4/G,raw_A=6*d*d*8/G,
        selected_weight_rollback=6*hsize*d*4/G,entry_and_boundaries=6*(d+2*hsize)*4*T/G,
        C0_read_transient=2*d*d*4/G,teacher_and_misc=7.,wandb_sidecar=4.)
    gpu=dict(model=params*4/G,factors=6*d*d*8/G,factor_transient=4*d*d*8/G,
        entry_candidate_weights=2*6*hsize*d*4/G,owner_checkpoint_head_workspace=20.)
    # Alpha replaces CPU rawA with FP32 C0 and mmap projector; <= MEMIT rawA.
    peak=max(sum(host.values()),params*4/G+8)
    require(peak<58 and sum(gpu.values())<95,'RESOURCE_BLOCKED_GPTJ_MEMORY_ESTIMATE')
    budget=dict(max_owner_width=max(widths),max_owner_padded_tokens=T,host_parts_GiB=host,
        host_peak_GiB=peak,GPU_parts_GiB=gpu,GPU_peak_GiB=sum(gpu.values()),measured=False,
        SDK_included=True,method_reduction=False)
    sourcefiles=[EASY/'easyeditor/util/generate.py',EASY/'easyeditor/models/memit/memit_main.py']
    native=[member(p) for p in sourcefiles+[EASY/'easyeditor/models/memit/compute_ks.py',
        EASY/'easyeditor/models/alphaedit/AlphaEdit_main.py',EASY/'hparams/MEMIT/gpt-j-6B.yaml',
        EASY/'hparams/AlphaEdit/gpt-j-6B.yaml']]
    evaluator=[member(ROOT/p) for p in ('project/run_scripts/jlz_price_gptj/scores.py',
        'project/run_scripts/jlz_price_gptj/observer.py','project/run_scripts/jlz_realization/inputs.py')]
    models={}
    for writer in MODELS:
        profiles={a:arm_profile(old['models']['LLAMA']['profiles']['CAP075'],a,
            'memit' if writer=='MEMIT' else 'alphaedit') for a in ARMS}
        if writer=='ALPHA':
            for p in profiles.values():p.update(lambda_alpha=10.,projector_sha256=project['sha256'],projector_cutoff=.02,
                lambda_C=0.,diagnostic_C0_scale=1.,factor_backend='A0_lambdaI_plus_NH_LU')
        models[writer]=dict(model=str(model),model_asset_identity=digest(assets),assets=assets,
            stats=stats,projector=project,cold_W0_H0=cold,profiles=profiles,resource_binding=budget,
            packs=[],contexts=None,W0_reuse=dict(status='NOT_AVAILABLE_BEFORE_FIRST_ARM'),
            context_sources=native[:2],evaluator_sources=evaluator)
    storage=storage_plan();free=shutil.disk_usage(LOCAL).free
    # Both previous Qwen triples are explicitly cancelled. Retain the entire
    # remaining Llama triple reserve from each predecessor, no delete assumed.
    existing=old['storage']['reserve_bytes']+sum(old['storage'][k] for k in (
        'atomic_reserve_bytes','collector_reserve_bytes','source_archive_reserve_bytes','error_reserve_bytes'))
    require(free>=storage['reserve_bytes']+existing,'RESOURCE_BLOCKED_COMBINED_STORAGE')
    c={k:copy.deepcopy(old[k]) for k in ('stream','native_root','stats_root','settings')}
    c.update(task_id=TASK,instruction_id=NONCE,attempt=str(attempt),run_instance=dict(date='2026-10-07',attempt=attempt.name),
        native_hparams=str(EASY/'hparams/MEMIT/gpt-j-6B.yaml'),models=models,assets=assets,
        seed=20261002,runtime=runtime,ordered_ids_sha256=ORDERED_SHA,native_reference=native,
        dependency_sources=evaluator,authority_members=[member(ROOT/'messages/acks/server4/jlz-price-gptj-2k.json')],
        cpu_preflight=member(preflight),storage=storage,tracking=dict(env_file=str(LOCAL/'tracking.env')),
        resources=dict(task_cap=2,project_cap=3,gpu=1,cpu=8,collector_cpu=8,host_mib=59392,
            hard_host_mib=60416,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00',
            reserve_bytes=storage['reserve_bytes'],remaining_existing_Llama_reserve_bytes=existing,
            combined_reserve_bytes=storage['reserve_bytes']+existing,free_bytes=free,
            free_inodes=os.statvfs(LOCAL).f_favail,memory_plans={w:budget for w in MODELS},ETA='NOT_MEASURED'),
        hparams_policy=dict(native=hp,effective=dict(layers=[3,4,5,6,7,8],lr=.5,Alpha_lambda=10.),
            reason='USER GPT-J EasyEdit hparams 채택; PRICE method/cap/base 유지'))
    write(out,c)
    return dict(config=str(out),model_SHA_checked=True,context='FIRST_REAL_ARM_NATIVE_PREPARATION',
        memory=budget,actual_B1='NOT_OBSERVED',cells=list(CELLS))

def finalize(config,preflight,out,attempt,tracking_review=None):
    c=json.loads(Path(config).read_text());p=json.loads(Path(preflight).read_text())
    require(c['task_id']==p['task']==TASK and p['passed'] and p['numeric_tests']==0,'FINAL_STATIC_SOURCE')
    for row in p['source']:verify(row)
    for row in c['assets']:
        st=Path(row['path']).stat()
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'FINAL_ASSET_STAT')
    old=json.loads((PREVIOUS/'config.json').read_text())
    existing=old['storage']['reserve_bytes']+sum(old['storage'][k] for k in (
        'atomic_reserve_bytes','collector_reserve_bytes','source_archive_reserve_bytes','error_reserve_bytes'))
    free=shutil.disk_usage(LOCAL).free
    require(free>=existing+c['storage']['reserve_bytes'],'RESOURCE_BLOCKED_COMBINED_STORAGE')
    c['resources'].update(remaining_existing_Llama_reserve_bytes=existing,
        combined_reserve_bytes=existing+c['storage']['reserve_bytes'],free_bytes=free)
    for mc in c['models'].values():
        budget=mc['resource_binding'];budget['host_parts_GiB']['unused_projector_slice_conservative']=0.
        budget['host_peak_GiB']=sum(budget['host_parts_GiB'].values())
        require(budget['host_peak_GiB']<58,'RESOURCE_BLOCKED_HOST_ESTIMATE')
    c['resources']['memory_plans']={w:c['models'][w]['resource_binding'] for w in MODELS}
    # Bind reviewed final evaluator bytes, including formatting-only changes.
    c['dependency_sources']=[member(r['path']) for r in c['dependency_sources']]
    for mc in c['models'].values():mc['evaluator_sources']=c['dependency_sources']
    require(attempt.parent==LOCAL and not attempt.exists(),'NEW_UNSUBMITTED_ATTEMPT')
    c['attempt']=str(attempt);c['run_instance']['attempt']=attempt.name
    c['cpu_preflight']=member(preflight)
    if tracking_review is not None:
        review=json.loads(Path(tracking_review).read_text())
        require(review.get('helper_ready') is True and review.get('integration')=='FAKE_SDK_PAYLOAD_AXES_IDENTITY_PASS',
            'TRACKING_INTEGRATION_NOT_READY')
        for row in review['helper_sources']:verify(row)
        c['tracking']['cpu_review']=member(tracking_review)
    write(out,c)
    return dict(config=str(out),sha256=sha(out),actual_B1='NOT_OBSERVED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True);p.add_argument('--finalize',type=Path)
    p.add_argument('--tracking-review',type=Path)
    a=p.parse_args();print(json.dumps(finalize(a.finalize,a.preflight,a.out,a.attempt,a.tracking_review) if a.finalize else prepare(a.out,a.attempt,a.preflight)))

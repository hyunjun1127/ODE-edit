"""Pinned author FE batch_edit + W0 firstforward, not the FE_HISTORY writer."""
import argparse,copy,importlib.util,json,os,sys,time,traceback,fcntl
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM,AutoTokenizer
from official.experiments.prepare import digest,write_new,file_sha
from official.experiments.checkpoint import rng_snapshot,rng_restore
from official.runners.server1.common import read,verify,member,factual_payload
from official.runners.server1.native import tensor_sha
from official.baselines import registry
from official.evaluation.factual import evaluate_counterfact
from official.evaluation import zsre_paper
from official.tracking import init,official_zsre_metrics

INSTRUCTION='USER-FE-ORIGINAL-W0-RESET-20261011-R1'
AUTHOR='478134dfb24b43f4e18b47e8500893ce3f9cc50f'
NAMES={'llama3':'llama3-8b','gptj':'gpt-j-6b','qwen25':'qwen2.5-7b'}

def modules(path):
    sys.path.insert(0,str(path))
    import precompute_z
    spec=importlib.util.spec_from_file_location('algs.memit.fe_original',str(path/'algs/memit/FE-memit_main.py'))
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    return precompute_z,native

def config_native(root,model,dataset):
    cfg=OmegaConf.load(root/'configs/config.yaml')
    cfg.llms=OmegaConf.load(root/'configs/llms'/f'{NAMES[model]}.yaml')
    cfg.algs=OmegaConf.load(root/'configs/algs/memit.yaml')
    cfg.bs=100;cfg.num_edits=2000;cfg.num_z_samples=2000;cfg.seed=0;cfg.data=dataset;cfg.z_method='firstforward'
    assert cfg.model_dtype=='bfloat16' and cfg.algs.add_old_keys and cfg.algs.L2==0
    return cfg

def validate(c):
    assert c['instruction']==INSTRUCTION and c['author_commit']==AUTHOR
    assert c['model'] in NAMES and c['dataset'] in ('cf','zsre')
    assert c['server'] in ('server1','server2') and (c['requests'],c['batch_size'],c['seed'])==(2000,100,0)
    assert c['milestones']==[5,10,15,20] and c['dtype']=='bfloat16'
    assert isinstance(c['storage_min_free_bytes'],int) and c['storage_min_free_bytes']>=0
    assert c['config_sha256']==digest({k:v for k,v in c.items() if k!='config_sha256'})

def save_latest(folder,weights,H,contexts,z_member,batch,identity,cursor,*,lock_path=None,reserve_bytes=0):
    folder.mkdir(parents=True,exist_ok=True)
    lock_path=Path(lock_path) if lock_path else folder.parent/'checkpoint-serialization.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    assert not lock_path.is_symlink() and folder.stat().st_dev==lock_path.parent.stat().st_dev
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:return _save_locked(folder,weights,H,contexts,z_member,batch,identity,cursor,reserve_bytes)
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def _save_locked(folder,weights,H,contexts,z_member,batch,identity,cursor,reserve_bytes):
    assert batch in (5,10,15,20)
    folder.mkdir(parents=True,exist_ok=True);tmp=folder/'latest.pt.partial';dest=folder/'latest.pt'
    assert not tmp.exists() and not tmp.is_symlink() and not dest.is_symlink()
    estimated=sum(v.numel()*v.element_size() for v in weights.values())+H.numel()*H.element_size()+16*1024**2
    space=os.statvfs(folder)
    assert space.f_bavail*space.f_frsize>=reserve_bytes+estimated,'CHECKPOINT_STORAGE_ADMISSION'
    payload=dict(weights={k:v.detach().cpu().clone() for k,v in weights.items()},cache_c=H.detach().cpu().clone(),contexts=copy.deepcopy(contexts),
        z_member=z_member,batch=batch,cursor=cursor,identity=identity,rng=rng_snapshot())
    try:
        with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        actual=tmp.stat().st_size;space=os.statvfs(folder)
        assert actual<=estimated and space.f_bavail*space.f_frsize>=reserve_bytes,'CHECKPOINT_ACTUAL_SERIALIZED_STORAGE'
        os.replace(tmp,dest)
        fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    except BaseException:
        if tmp.exists() and not tmp.is_symlink():tmp.unlink()
        raise
    receipt=dict(member(dest),batch=batch,identity=identity,final_W20=batch==20)
    meta=folder/'latest.json.partial'
    with meta.open('x') as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(meta,folder/'latest.json')
    fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return receipt

def load_latest(folder,identity):
    receipt=read(folder/'latest.json');verify({k:receipt[k] for k in ('path','bytes','sha256')})
    saved=torch.load(receipt['path'],map_location='cpu',weights_only=False)
    assert saved['identity']==identity==receipt['identity'] and saved['batch']==receipt['batch']
    return saved

def versions(model,excluded=()):return {k:(v._version,str(v.dtype),tuple(v.shape)) for k,v in model.named_parameters() if k not in excluded}

def run(config,lock_path,resume=False):
    c=read(config);validate(c);lock=read(lock_path)
    disk=os.statvfs(Path(c['output']).parent)
    assert disk.f_bavail*disk.f_frsize>=c['storage_min_free_bytes'],'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE'
    assert Path(__file__).resolve().is_relative_to(Path(lock['source_directory']).resolve())
    for m in lock['source_members']+c['author_members']:verify(m)
    cfg=config_native(Path(c['author_path']),c['model'],c['dataset'])
    assert OmegaConf.to_container(cfg.llms,resolve=True)==c['llms']
    assets=read(verify(c['assets']));records=read(verify(c['stream']))
    records=records['records'] if isinstance(records,dict) else records
    assert len(records)==2000 and digest(records)==c['stream_sha256']
    from official.runners.server1.assets import member as asset_member
    for m in assets['model']['members']:
        actual=asset_member(m['path'],allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
        assert actual['real_path']==m['real_path']
    pre,native=modules(Path(c['author_path']))
    pre.set_random_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(assets['model']['snapshot'],local_files_only=True,torch_dtype=torch.bfloat16,trust_remote_code=True,attn_implementation='eager').to('cuda:0').eval()
    tok=AutoTokenizer.from_pretrained(assets['model']['snapshot'],local_files_only=True,trust_remote_code=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    tok.padding_side='right'
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    layers=list(cfg.llms.layers)
    native.covs.clear()
    for layer in layers:
        m=c['C0'][str(layer)];verify(m)
        with np.load(m['path'],allow_pickle=False) as data:
            count=int(data['mom2.count']);summed=np.asarray(data['mom2.mom2'])
            assert count>0 and summed.dtype==np.float32 and np.isfinite(summed).all()
            native.covs.append(torch.from_numpy(summed.copy())/count)
    weights={cfg.llms.rewrite_module_tmp.format(l)+'.weight':native.nethook.get_parameter(model,cfg.llms.rewrite_module_tmp.format(l)+'.weight') for l in layers}
    dimension=native.get_fc_dim(model,cfg);H=torch.zeros((len(layers),dimension,dimension),device='cpu')
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    identity=dict(config_sha256=c['config_sha256'],source=lock['source_commit'],author=AUTHOR,stream=c['stream_sha256'],dtype='bfloat16')
    values=dict(server=c['server'],task_id='fe-original-w0-2k-20261011',model=c['model'],model_family={'llama3':'llama','gptj':'gptj','qwen25':'qwen2'}[c['model']],
        writer='fe_author_w0_fixed',baseline='FE-author-repo-W0-fixed-z-sequential',role='scientific',arm=c['arm'],attempt=c['attempt'],
        source_sha=lock['source_commit'],config_sha=c['config_sha256'],dataset=c['dataset'],metric_schema='official-baselines-scalar-v1',instruction_id='USER-OFFICIAL-BASELINES-20261008-R1')
    if c['dataset']=='cf':values['generation_schedule']='DEFERRED_CHECKPOINT_EVALUATION'
    tracker=init(env_file=c['tracking_env_file'],spool=out/('tracking-resume' if resume else 'tracking'),config=values)
    def evaluate(rows,prefix,batch,label):
        before=versions(model);rng=rng_snapshot()
        try:
            result=evaluate_counterfact(model,tok,rows,identity=identity,batch_size=16,device='cuda:0') if c['dataset']=='cf' else zsre_paper.evaluate(model,tok,rows,model_family=c['model'],identity=identity,batch_size=16,device='cuda:0')
        finally:rng_restore(rng)
        assert before==versions(model)
        write_new(out/'factual'/f'{label}.json',result)
        payload=factual_payload(result,prefix,batch*100) if c['dataset']=='cf' else official_zsre_metrics(result['summary'],config_values=values,endpoint=prefix,edits=batch*100,post_state_edits=batch*100)
        assert tracker.log(payload) is not False
        return member(out/'factual'/f'{label}.json')
    exit_code=1
    try:
        requests=registry.requests(records,'MEMIT_FE',c['model'])
        for i,r in enumerate(requests):r['sample_idx']=i
        initial=versions(model);initial_selected={k:tensor_sha(v) for k,v in weights.items()}
        zfile=out/'z-cache'/(digest(dict(identity,llms=c['llms']))+'.pt')
        if resume:
            saved=load_latest(out/'checkpoint',identity);start=saved['batch']
            assert start<20
            with torch.no_grad():
                for k,v in weights.items():v.copy_(saved['weights'][k])
            H=saved['cache_c'];contexts=saved['contexts'];z_member=saved['z_member'];verify(z_member)
            zs=torch.load(z_member['path'],map_location='cpu',weights_only=True)
            native.CONTEXT_TEMPLATES_CACHE=contexts;pre.CONTEXT_TEMPLATES_CACHE=contexts;rng_restore(saved['rng']);del saved
        else:
            assert not (out/'checkpoint/latest.json').exists() and not list((out/'z-cache').glob('*.pt')),'NO_OLD_STATE_REUSE'
            contexts=pre.get_context_templates(model,tok);native.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(contexts)
            zfile=out/'z-cache'/(digest(dict(identity,llms=c['llms'],contexts=contexts,revision=assets['model']['identity']['revision']))+'.pt')
            write_new(out/'contexts.json',dict(contexts=contexts,identity=identity,source='author.precompute_z.get_context_templates'))
            zs=pre.compute_z_batch_first_forward(model,tok,requests,cfg,'firstforward',layers,2000,contexts)
            assert set(zs)==set(layers) and all(z.shape[1]==2000 and torch.isfinite(z).all() for z in zs.values())
            assert initial==versions(model) and initial_selected=={k:tensor_sha(v) for k,v in weights.items()}
            zfile.parent.mkdir(parents=True,exist_ok=True);torch.save({k:v.detach().cpu() for k,v in zs.items()},zfile);z_member=member(zfile)
            zs={k:v.detach().cpu() for k,v in zs.items()};start=0
        # Author apply wrapper seeds immediately before its batch loop.
        if not resume:native.set_random_seed(0)
        for p in model.parameters():p.requires_grad_(False)
        model.eval();nonedited=versions(model,weights)
        for batch in range(start+1,21):
            chunk=copy.deepcopy(requests[(batch-1)*100:batch*100])
            for r in chunk:r['target_new']=' '+r['target_new']
            before={k:tensor_sha(v) for k,v in weights.items()};started=time.monotonic()
            native.batch_edit(cfg,model,tok,chunk,torch.device('cuda:0'),H,zs)
            assert nonedited==versions(model,weights) and all(torch.isfinite(v).all() for v in weights.values()) and torch.isfinite(H).all()
            assert native.CONTEXT_TEMPLATES_CACHE==contexts
            current=evaluate(records[(batch-1)*100:batch*100],'current/post',batch,f'current-{batch:02d}')
            cursor=dict(batch=batch,edits=batch*100,request_sha256=digest(requests[(batch-1)*100:batch*100]),current=current,elapsed=time.monotonic()-started,
                selected_before=before,selected_after={k:tensor_sha(v) for k,v in weights.items()})
            if batch in c['milestones']:
                cursor['all_seen']=evaluate(records[:batch*100],'all_seen/post',batch,f'all-seen-{batch:02d}')
                disk=os.statvfs(out)
                assert disk.f_bavail*disk.f_frsize>=c['storage_min_free_bytes'],'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE'
                cursor['checkpoint']=save_latest(out/'checkpoint',weights,H,contexts,z_member,batch,identity,dict(batch=batch,edits=batch*100),lock_path=c['checkpoint_lock'],reserve_bytes=c['storage_min_free_bytes'])
            write_new(out/'commits'/f'batch-{batch:02d}.json',dict(identity=identity,cursor=cursor,z_sha256=z_member['sha256']))
        write_new(out/'COMPLETE.json',dict(identity=identity,requests=2000,commits=20,final_cursor=cursor,checkpoint=member(out/'checkpoint/latest.pt'),generation='DEFERRED' if c['dataset']=='cf' else 'NOT_APPLICABLE'))
        exit_code=0
    except BaseException as error:
        write_new(out/'FAILURE.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checkpoint_keep=True));raise
    finally:tracker.finish(exit_code=exit_code,timeout=45)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--lock',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.config,a.lock,a.resume)

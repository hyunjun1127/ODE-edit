"""완료된 6-chain만 CPU 검증하고 exact 전송 allowlist를 생성한다."""
import csv,hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/data/janghj/ODE-edit')
REPORT=ROOT/'worktree/experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2'
sys.path.insert(0,str(ROOT));from checkpoint_loader import tensor_sha
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def member(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False)
    p.chmod(0o600)
def rows(n):return list(csv.DictReader((REPORT/n).open()))
def safe(p):
    assert p.is_file() and p.resolve()==p and p.stat().st_uid==os.getuid(),str(p)
    assert p.is_relative_to(REPO/'local/blue-lifelong-b100x100')
def main():
    import torch
    torch.set_num_threads(4)
    out=ROOT/'source-package';out.mkdir(exist_ok=False)
    report_manifest=json.loads((REPORT/'analysis-manifest.json').read_text())
    # Verify published source tables against sealed report member entries.
    print('REPORT_KEYS',list(report_manifest),flush=True)
    reportmembers=report_manifest.get('members',[])
    for n in ['raw-member-inventory.csv','checkpoint-tensors.csv','source-config-compatibility.csv','chain-integrity.csv']:
        matches=[m for m in reportmembers if m['path']==n or m['path'].endswith('/'+n)]
        assert len(matches)==1,n
        assert sha(REPORT/n)==matches[0]['sha256'],n
    raw={r['path']:r for r in rows('raw-member-inventory.csv')}
    tensors={(r['arm'],int(r['batch']),r['key']):r for r in rows('checkpoint-tensors.csv')}
    integrity={r['arm']:r for r in rows('chain-integrity.csv')}
    entries=[];allow=[];metadata_files={};base_refs={}
    schedule=[1,5,10,20,30,40,50,60,70,80,90,100]
    for c in rows('source-config-compatibility.csv'):
        arm=c['arm'];root=Path(c['raw_root']);attempt=root.parent.parent
        terminalpath=root/'terminal.json';safe(terminalpath)
        assert sha(terminalpath)==integrity[arm]['terminal_sha256']
        terminal=json.loads(terminalpath.read_text());assert terminal['status']=='TERMINAL_VALID' and terminal['batches']==100 and terminal['requests']==10000
        assert terminal['failure']==terminal['nonfinite']==0
        tm={m['path']:m for m in terminal['manifest_members']}
        lockpath=attempt/'execution.lock.json';lock=json.loads(lockpath.read_text())
        runtimepath=root/'runtime.json';runtime=json.loads(runtimepath.read_text())
        assert sha(runtimepath)==c['runtime_sha256'] and sha(lockpath)==runtime['lock_sha256']
        assert lock['sample_root']==c['sample_root']=='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
        spec=runtime['spec'];method=spec['method'];layers=spec['layers']
        assert layers==([4,8] if 'ORIGINAL' in arm else [4] if 'L4_ONLY' in arm else [8])
        for m in lock['members']:
            if m['path'].startswith(lock['snapshot']+'/'):base_refs[m['path']]=m
        for p in [terminalpath,runtimepath,lockpath,Path(c['source_archive']),Path(spec['config'])]:metadata_files[str(p)]=p
        assert sha(Path(c['source_archive']))==c['source_archive_sha256']
        for b in schedule:
            path=root/f'B{b:03d}/W-method-state.pt';safe(path)
            m=member(path);published=raw[str(path)];tsealed=tm[str(path.relative_to(root))]
            assert m['bytes']==int(published['bytes'])==tsealed['bytes']
            assert m['sha256']==published['sha256']==tsealed['sha256']
            cp=torch.load(path,map_location='cpu',weights_only=True)
            assert set(cp)=={'weights','cache_c','metadata'}
            assert set(cp['weights'])=={f'model.layers.{l}.mlp.down_proj.weight' for l in layers}
            weights={}
            for k,t in cp['weights'].items():
                tr=tensors[(arm,b,k)];h=tensor_sha(t)
                assert h==tr['tensor_sha256'] and list(t.shape)==[4096,14336] and t.dtype==torch.float32 and torch.isfinite(t).all()
                weights[k]=dict(shape=list(t.shape),dtype=str(t.dtype),sha256=h,physical_layer=int(k.split('.')[2]))
            cache=cp['cache_c'];ch=tensor_sha(cache)
            assert list(cache.shape)==([len(layers),14336,14336] if method=='AlphaEdit' else [0])
            assert ch==tr['cache_sha256'] and cache.dtype==torch.float32 and torch.isfinite(cache).all()
            meta=cp['metadata'];assert meta['batch']==b and len(meta['seen_ids'])==b*100
            assert meta['sample_root']==lock['sample_root'] and meta['base_model_revision']==lock['revision']
            assert digest(meta['contexts'])==tr['contexts_sha256']
            commit=root/f'B{b:03d}/commit.json';cr=json.loads(commit.read_text())
            assert sha(commit)==tm[str(commit.relative_to(root))]['sha256']
            assert cr['checkpoint']['sha256']==m['sha256'] and cr['endpoint']['weights']=={k:v['sha256'] for k,v in weights.items()}
            assert cr['endpoint']['cache']==ch and meta['state']==cr['endpoint']
            metadata_files[str(commit)]=commit
            m['destination_relative']='payload/'+str(path.relative_to(REPO))
            entries.append(dict(arm=arm,display_label=c['arm_display_label'],method=method,variant=spec['variant'],layers=layers,batch=b,editcount=b*100,job=c['job'],source_head=c['source_head'],source_tree=c['source_tree'],blue_head=c['blue_head'],blue_tree=c['blue_tree'],source_archive_sha256=c['source_archive_sha256'],source_lock_sha256=sha(lockpath),model_revision=c['model_revision'],base_selected_weights=runtime['W0']['weights'],weights=weights,method_state=dict(shape=list(cache.shape),dtype=str(cache.dtype),sha256=ch,semantics='AlphaEdit selected-layer history; NOT forward weights' if method=='AlphaEdit' else 'empty sentinel; NO history'),file=m,metadata_sha256=digest(meta),context_sha256=digest(meta['contexts']),sample_root=meta['sample_root'],rng_components=sorted(meta['rng']),terminal=member(terminalpath),commit=member(commit),validation='CPU weights_only load + all selected tensor hashes/finite + terminal/report/commit linkage; no model/GPU/forward',full_model_checkpoint=False))
            allow.append(str(path.relative_to(REPO)))
            del cp,cache,weights,t
        print('VERIFIED',arm,12,flush=True)
    assert len(entries)==72 and len(set(allow))==72
    # Source closure for schema/restoration; do not copy model/P/stats/dataset/raw prompts.
    for p in [ROOT/'checkpoint_loader.py',ROOT/'build_inventory.py',ROOT/'authoritative-envelope.txt',ROOT/'worktree/PROTOCOL.md']:
        metadata_files[str(p)]=p
    helper=Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/source-tech-r2')
    for p in [helper/'project/run_scripts/blue_alphaedit_sequential_comparison/integrity.py']:
        metadata_files[str(p)]=p
    copies=[]
    for i,p in enumerate(sorted(metadata_files.values())):
        relative='closure/'+str(i).zfill(3)+'-'+p.name;dest=out/relative;dest.parent.mkdir(exist_ok=True)
        with dest.open('xb') as f:f.write(p.read_bytes())
        copies.append(dict(source=str(p),relative=relative,bytes=p.stat().st_size,sha256=sha(p)))
    save(out/'checkpoint-manifest.json',dict(instruction_id='ODEEDIT-S06-BLUE-CHECKPOINT-DOWNSTREAM-S2-S4-V1',checkpoints=entries,count=72,total_checkpoint_bytes=sum(e['file']['bytes'] for e in entries),base_model_members=list(base_refs.values()),source_closure=copies,report=member(REPORT/'factual-report-ko.md'),report_manifest=member(REPORT/'analysis-manifest.json'),omissions=[dict(jobs=['40441','40442'],status='NOT_READY_RUNNING_NO_LIVE_SCIENTIFIC_FILE_READ'),dict(jobs=['40443','40444','40445','40446'],status='NOT_READY_PENDING'),dict(jobs=['42657','42658'],status='NOT_AVAILABLE_PENDING_NATIVE_BASELINE')],scientific_promotion=False))
    with (ROOT/'payload-files.txt').open('x') as f:f.write('\n'.join(allow)+'\n')
    files=[dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    save(out/'source-seal.json',dict(members=files,member_root=digest(files)))
    print(json.dumps(dict(status='LOCAL_72CP_VERIFIED',bytes=sum(e['file']['bytes'] for e in entries),manifest_sha=sha(out/'checkpoint-manifest.json'),source_seal_sha=sha(out/'source-seal.json'))),flush=True)
if __name__=='__main__':main()

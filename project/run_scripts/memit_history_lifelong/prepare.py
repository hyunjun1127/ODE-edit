"""CPU-only asset/source/token order preflight, no Slurm or model forward."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from .io import save,file_sha,digest

def main(root):
    import torch,numpy as np,transformers
    from transformers import AutoTokenizer
    from scripts.fixed_counterfact import load_prefix,verify
    root=Path(root);w=Path(__file__).resolve().parents[3];blue=root/'blue-upstream'
    old=json.loads((root/'inputs/baseline/execution.lock.json').read_text())
    ready=json.loads((w/'agents/server3/experiment-ready-paths-20260919-v1.json').read_text())
    spec=ready['models']['llama3-8b-inst'];members=[]
    def member(p,expected=None):
        p=Path(p);sha=file_sha(p)
        if expected:assert sha==expected,'SHA_MISMATCH:'+str(p)
        r=dict(path=str(p),bytes=p.stat().st_size,sha256=sha);members.append(r);return r
    assets=[]
    for x in old['members']:
        if x['path'].startswith(spec['snapshot']+'/'):
            assets.append(member(x['path'],x['sha256']))
    stats=[]
    for l,s in spec['assets']['covariance'].items():
        stats.append(member(s['path'],s['sha256']))
        with np.load(s['path']) as a:
            x=a['mom2.mom2'];assert x.shape==(14336,14336) and x.dtype==np.float32 and np.isfinite(x).all()
    data=verify(ready['dataset_root']);records=load_prefix(ready['dataset_root'],10000)
    for name in ('counterfact.json','source-sample.lock.json','receipt.json'):member(Path(ready['dataset_root'])/name)
    member(root/'inputs/baseline/contexts.json','33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e')
    assert subprocess.check_output(['git','-C',str(blue),'rev-parse','HEAD'],text=True).strip()=='311b076a92e4ed0f14f5c8b4909732da781bc5f7'
    assert not subprocess.check_output(['git','-C',str(blue),'status','--porcelain'],text=True).strip()
    names=subprocess.check_output(['git','-C',str(blue),'ls-files'],text=True).splitlines()
    for rel in names:
        if rel.endswith(('.py','.yml','.yaml','.json')):member(blue/rel)
    for x in old['members']:
        if '/source-tech-r2/' in x['path'] and any(k in x['path'] for k in ('/contracts.py','/evaluator.py','/evaluation.py','/integrity.py','/counterfact_locality_evaluator.py')):
            p=w/x['path'].split('/source-tech-r2/')[1]
            if p.is_file():member(p,x['sha256'])
    member(w/'project/run_scripts/memit_history_lifelong/hparams.json','1d701acd9d2f3ca7bbd91f0b646cda376e57996148a2baebbe853a9631c99b15')
    tok=AutoTokenizer.from_pretrained(spec['snapshot'],local_files_only=True);tok.pad_token_id=tok.eos_token_id;tok.add_bos_token=False
    contexts=json.loads((root/'inputs/baseline/contexts.json').read_text())
    token_rows=[];max_tokens=0
    for i,r in enumerate(records):
        rr=r['requested_rewrite'];target=rr['target_new']['str'];target=target if target.startswith(' ') else ' '+target
        ids=tok(target)['input_ids']
        if ids[0] in (tok.bos_token_id,tok.unk_token_id):ids=ids[1:]
        assert ids
        texts=[(c.format(rr['prompt'])+tok.decode(ids[:-1])).format(rr['subject']) for group in contexts for c in group]+['{} is a'.format(rr['subject'])]
        encoded=tok(texts)['input_ids'];max_tokens=max(max_tokens,max(map(len,encoded)))
        token_rows.append(dict(ordinal=i,case_id=r['case_id'],request=digest(rr),native_target_ids=ids,native_prompt_hash=digest(encoded)))
    save(root/'preparation/token-order.json',token_rows)
    save(root/'preparation/assets.json',dict(members=members,dataset=data,model=assets,stats=stats,max_native_z_prompt_tokens=max_tokens,native_context_sequences=7,token_order_sha256=digest(token_rows),batch_order=[digest(token_rows[i:i+100]) for i in range(0,10000,100)],torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,gpu_forward=False))
    print(json.dumps(dict(status='CPU_ASSET_PREFLIGHT_PASS',members=len(members),max_native_z_prompt_tokens=max_tokens,records=len(records))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)

"""Dataset-to-evaluator identity and checkpoint recovery metadata checks; CPU only."""
import torch
from .common import *

def main(out):
    s=sample();lk=read(attempt(0)/'execution.lock.json');data={r['case_id']:r for r in read(lk['dataset'])}
    expected={tag:[] for tag in MULT}
    for r,inv in zip(s['records'],s['inventory']):
        d=data[r['case_id']];rw=d['requested_rewrite'];assert digest([rw,d['paraphrase_prompts'],d['neighborhood_prompts']])==inv['prompt_inventory_sha256']
        p={'RS':[rw['prompt'].format(rw['subject'])],'PS':d['paraphrase_prompts'],'NS':d['neighborhood_prompts']}
        for tag in MULT:
            assert len(p[tag])==MULT[tag]
            expected[tag]+=[digest([r['case_id'],i,x,rw['target_new']['str'],rw['target_true']['str']]) for i,x in enumerate(p[tag])]
    del data
    rows=[];meta=[]
    torch.set_num_threads(2)
    for cell,arm in enumerate(ARMS):
        rt=read(root(cell)/'runtime.json');isalpha=rt['spec']['method']=='AlphaEdit';cov=None
        for b in range(1,101):
            paths=['current.json']+(['seen-full.json'] if b in SCHEDULE else [])
            for name in paths:
                d=read(root(cell)/f'B{b:03d}'/name)
                for tag,m in MULT.items():
                    exp=expected[tag][(b-1)*100*m:b*100*m] if name=='current.json' else expected[tag][:b*100*m]
                    assert [x['identity'] for x in d['metrics'][tag]['rows']]==exp
                rows.append(dict(arm=arm,batch=b,scope=name,requests=d['requests'],dataset_prompt_target_identity='PASS'))
            if b not in SCHEDULE:continue
            cp=torch.load(root(cell)/f'B{b:03d}'/'W-method-state.pt',map_location='cpu',weights_only=True,mmap=True)
            md=cp['metadata'];assert md['method']==rt['spec']['method'] and md['stats_members']==lk['stats_members']
            cc=md['covariance']
            if isalpha:assert cc=={}
            else:
                normalized={k:{x:v[x] for x in ['sha256','dtype','shape']} for k,v in cc.items()}
                if cov is not None:assert normalized==cov
                cov=normalized;assert len(cc)==len(rt['spec']['layers'])
                assert all(v['shape']==[14336,14336] and v['dtype']=='torch.float32' for v in cc.values())
            assert md['rng']['torch'] and md['rng']['cuda'] and md['contexts']
            meta.append(dict(arm=arm,batch=b,covariance_tensor_hashes=digest(cc),history=isalpha,static_covariance_stable=not isalpha,rng_context_present=True,scope='CPU_METADATA; full GPU continuation not run'))
            del cp
        print('DATASET_IDENTITY_PASS',arm,flush=True)
    csvwrite(out/'dataset-evaluator-identity.csv',rows);csvwrite(out/'checkpoint-recovery-metadata.csv',meta)
    controls=[]
    for name in ['pause-receipt.json','control-receipt.json']:
        p=BASE/'local/blue-lifelong-b100x100/attempt-v1/main-queue-override-v1'/name
        controls.append(dict(path=str(p),sha256=sha(p),receipt=read(p)))
    assert controls[0]['receipt']['smoke_terminal']=='CANCELLED by1025, elapsed00:00:00, no allocation'
    save(out/'supplemental-control-receipt.json',dict(members=controls,smoke='SKIPPED_USER_DIRECTED_NOT_PASS',evaluator_actual_tensor_padding='manual LEFT; tokenizer property right',position_ids='NOT_EXPLICITLY_PASSED; native model default',new_gpu_parity='NOT_RUN'))

if __name__=='__main__':main(cli().out)

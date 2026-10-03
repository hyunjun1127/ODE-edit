"""Fixed Wentry/Weff row-wise exact F.linear selector; no new writer solve."""
from contextlib import contextmanager
import time
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_realized_subject.observe import reduce_rows,active_flags
from project.run_scripts.jlz_realized_subject.writer import rng_snapshot,rng_equal
from .common import require,digest,state,write
from .lookup import masks


@contextmanager
def select_linears(a,entry,effective,selected):
    old={}
    for l in a.sites:
        module=a.blocks[l].mlp.down_proj
        old[l]=module.__dict__.get('forward')
        def forward(x,l=l):
            require(tuple(x.shape[:2])==tuple(selected.shape),'MASK_ACTIVATION_SHAPE')
            base=F.linear(x,entry[l]);edited=F.linear(x,effective[l])
            return torch.where(selected[:,:,None],edited,base)
        module.forward=forward
    try:yield
    finally:
        for l,value in old.items():
            module=a.blocks[l].mlp.down_proj
            if value is None:del module.__dict__['forward']
            else:module.forward=value


@torch.no_grad()
def score(a,bench,pairs,lookups,entry,effective,mode,microbatch=1):
    values=[]
    for start in range(0,len(pairs),microbatch):
        group=pairs[start:start+microbatch];encoded=[bench.evaluation_ids(p,t) for p,t in group]
        width=max(len(p)+len(t)-1 for p,t in encoded)
        require(width<=a.model.config.max_position_embeddings,'NO_LENGTH_TRUNCATION')
        choices=masks(encoded,lookups[start:start+microbatch],width,a.device)
        ids=torch.full((len(group),width),bench.tokenizer.pad_token_id,device=a.device,dtype=torch.long)
        attention=choices['ALL'].long();positions=[]
        for i,(p,t) in enumerate(encoded):
            row=(p+t)[:-1];offset=width-len(row)
            ids[i,offset:]=torch.tensor(row,device=a.device)
            positions.append(list(range(offset+len(p)-1,width)))
        with select_linears(a,entry,effective,choices[mode]):
            h=a.model.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state
        logits=a.model.lm_head(torch.cat([h[i,pos] for i,pos in enumerate(positions)])).float()
        lp,pred=logits.log_softmax(-1),logits.argmax(-1);cursor=0
        require(bool(torch.isfinite(logits).all()),'NONFINITE_MASK_LOGITS')
        for p,t in encoded:
            labels=torch.tensor(t,device=a.device);n=len(t);local=lp[cursor:cursor+n];guess=pred[cursor:cursor+n]
            values.append(dict(nll=float(-local.gather(1,labels[:,None]).mean()),token_count=n,
                token_correct=int((guess==labels).sum()),strict=bool((guess==labels).all()),token_identity=digest([p,t])))
            cursor+=n
    return values


def observe_masks(a,bench,records,H,lookup,W0,W25,out,microbatch=1):
    selected={r['identity']:r for r in lookup['rows'] if r['identifiable']}
    if not selected:
        write(out/'coverage.json',dict(status='NOT_IDENTIFIABLE',denominator=0,full_N=lookup['denominator']))
        return {}
    before=state(a,H);guard=a.guard();hooks=a.hook_signature();rng=rng_snapshot();ctx=digest(bench.contexts)
    entry={l:w.to(a.device) for l,w in W0.items()};effective={l:w.to(a.device) for l,w in W25.items()}
    output={};flags=active_flags(records)
    try:
        for mode in ('NONE','SUBJECT_ONLY','NONSUBJECT_ONLY','ALL'):
            started=time.monotonic();rows=[]
            for begin in range(0,len(records),10):
                specs=[];pairs=[];lookups=[]
                for r in records[begin:begin+10]:
                    rw=r['requested_rewrite']
                    for ix,prompt in enumerate(r['neighborhood_prompts']):
                        identity=digest([r['case_id'],'N',ix,prompt,rw['target_new']['str'],rw['target_true']['str']])
                        if identity not in selected:continue
                        specs.append(dict(identity=identity,case_id=r['case_id'],kind='N',prompt_index=ix,
                            endpoint=25,mask=mode,active_at_endpoint=flags[r['case_id']]))
                        pairs.extend([(prompt,rw['target_new']['str']),(prompt,rw['target_true']['str'])])
                        lookups.extend([selected[identity]['lookup']]*2)
                values=score(a,bench,pairs,lookups,entry,effective,mode,microbatch)
                for i,row in enumerate(specs):
                    for label,value in zip(('new','true'),values[2*i:2*i+2]):
                        row.update({label+'_'+k:v for k,v in value.items()})
                        require(row[label+'_token_identity']==selected[row['identity']][label+'_token_identity'],'D2_TOKEN_IDENTITY')
                    row['margin_true_minus_new']=row['true_nll']-row['new_nll']
                    row['NS_margin_new_minus_true']=-row['margin_true_minus_new']
                rows.extend(specs);write(out/mode/f'chunk-{begin:04d}.json',dict(rows=specs,state=before))
            require([r['identity'] for r in rows]==lookup['common_four_mask_subset'],'FOUR_MASK_COVERAGE')
            require(state(a,H)==before and a.guard()==guard and a.hook_signature()==hooks and rng_equal(rng) and digest(bench.contexts)==ctx,'MASK_OBSERVER_MUTATION')
            summary=dict(summary=reduce_rows(rows),seconds=time.monotonic()-started,rows=len(rows),
                writer_solves=0,refits=0,history_appends=0,nonmutation=True)
            write(out/mode/'summary.json',summary);output[mode]=rows
    finally:
        require(state(a,H)==before and a.guard()==guard and a.hook_signature()==hooks and rng_equal(rng),'MASK_RESTORE_FAILURE')
    return output

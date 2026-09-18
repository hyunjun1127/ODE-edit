"""Native old/new TF union plus explicit canonical union, no P/N access."""
import copy
import torch
from project.run_scripts.single_layer_zflow.native_binding import build_training_sequences
from project.run_scripts.bg_tw_reference.ep_tw.model_adapter import _token_contracts
from .common import digest

def pack(ids):
    ids=torch.tensor([list(ids)],dtype=torch.long)
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(ids.shape[1])[None])

def protected_sequences(tok,eval_tok,requests,contexts):
    caches=[];lookup={};rows=[];missing_old=[]
    def add(ids,positions,labels,meta):
        ids=tuple(ids)
        if ids not in lookup:lookup[ids]=len(caches);caches.append(pack(ids))
        assert all(0<=p<len(ids) for p in positions)
        rows.append(dict(cache=lookup[ids],positions=list(positions),labels=list(labels),**meta))
    for ri,request in enumerate(requests):
        for branch in ('new','old'):
            label=request.get('target_new' if branch=='new' else 'target_true',{}).get('str')
            if not label:
                if branch=='old':missing_old.append(request['case_id'])
                continue
            r=copy.deepcopy(request);r['target_new']['str']=label
            native=build_training_sequences(tok,[r],contexts)
            for seq in native.sequences:
                if seq.kind!='edit':continue
                add(seq.ids,seq.edit_positions,seq.edit_labels,dict(case_id=request['case_id'],kind='native',branch=branch,context=seq.context_index,sequence_id=f'{request["case_id"]}:native:{seq.context_index}:{branch}'))
            prompt=request['prompt'].format(request['subject'])
            prompt_ids=_token_contracts().prompt_token_ids(eval_tok,prompt)
            target=_token_contracts().target_token_ids(eval_tok,label)
            ids=prompt_ids+target[:-1]
            add(ids,range(len(prompt_ids)-1,len(ids)),target,dict(case_id=request['case_id'],kind='canonical',branch=branch,context=0,sequence_id=f'{request["case_id"]}:canonical:{branch}'))
    # Keys deduplicate only identical complete input-prefix bytes. All aliases retained.
    seen={};unique=[];aliases=[]
    for ci,p in enumerate(caches):
        ids=p['input_ids'][0].tolist()
        for pos in range(len(ids)):
            prefix=tuple(ids[:pos+1])
            if prefix not in seen:seen[prefix]=len(unique);unique.append((ci,pos))
            aliases.append(dict(cache=ci,position=pos,key_column=seen[prefix],prefix_sha=digest(prefix)))
    return caches,rows,unique,dict(sequences=len(rows),distinct_inputs=len(caches),key_columns=len(unique),key_aliases=aliases,
        missing_old=missing_old,essence_in_lock=False,official_P_N=False,padding_in_lock=False,new_EOS=False,
        native_tokenizer_add_bos=tok.add_bos_token,canonical_tokenizer_add_bos=eval_tok.add_bos_token)

def quality_ok(candidate,anchor,epsilon=1e-4):
    if set(candidate)!=set(anchor):raise ValueError('QUALITY_ID_INVENTORY')
    reasons=[]
    for sid,a in anchor.items():
        c=candidate[sid]
        if a['branch']=='new' and c['nll']>a['nll']+epsilon:reasons.append((sid,'PER_SEQUENCE_NLL'))
        if a['branch']=='new' and a['strict'] and not c['strict']:reasons.append((sid,'STRICT_ID_LOST'))
    for sid,a in anchor.items():
        if a['kind']=='canonical' and a['branch']=='new':
            old=sid.removesuffix('new')+'old'
            if old in anchor and a['nll']<anchor[old]['nll']:
                if not candidate[sid]['nll']<candidate[old]['nll']:reasons.append((sid,'PAIR_ID_LOST'))
    return not reasons,reasons

def score_rows(oracle,weight,rows):
    result={};bycache={}
    for row in rows:bycache.setdefault(row['cache'],[]).append(row)
    with torch.no_grad():
        for ci,group in bycache.items():
            positions=sorted(set(p for r in group for p in r['positions']))
            logits=oracle.logits_at(ci,weight,positions) # [positions,vocab]
            lp=logits.log_softmax(-1);pred=logits.argmax(-1)
            for row in group:
                idx=[positions.index(p) for p in row['positions']]
                label=torch.tensor(row['labels'],device=logits.device)
                nll=float(-lp[idx,label].double().mean())
                if not torch.isfinite(torch.tensor(nll)):raise FloatingPointError('NONFINITE_GUARD')
                result[row['sequence_id']]={**row,'nll':nll,'strict':bool((pred[idx]==label).all()),'predictions':pred[idx].cpu().tolist()}
    return result

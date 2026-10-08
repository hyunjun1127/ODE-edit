"""Native text adapter only; no v4 writer/optimizer is imported."""
import unicodedata
import torch
from official.ours.core.jlz_pilot.prompts import prepare as native_prepare
from official.ours.common import require, digest

def fact_identity(record):
    r=record['requested_rewrite']
    return digest([unicodedata.normalize('NFC', ' '.join(r['subject'].split())),r['relation_id']])

def version(record):
    t=record['requested_rewrite']['target_new']
    return str(t.get('id',t['str']))

class CounterFactAdapter:
    def __init__(self, tokenizer, contexts):
        self.tokenizer,self.contexts=tokenizer,contexts

    def prepare(self, records):
        pack=native_prepare(self.tokenizer,[r['requested_rewrite']|{'case_id':r['case_id']} for r in records],self.contexts,'cpu')
        pack['record_ids']=[r['case_id'] for r in records]
        pack['records']=[{'case_id':r['case_id'],'requested_rewrite':r['requested_rewrite']} for r in records]
        proofs=[]
        for i,row in enumerate(pack['rw_rows']):
            p=pack['lookup'][row]; q=pack['key_lookup'][i]
            proofs.append(p==q and all(torch.equal(pack['tokens'][k][row,:p+1],pack['key_tokens'][k][i,:q+1]) for k in ('input_ids','attention_mask')))
        pack['entry_key_prefix_exact']=all(proofs)
        require(pack['entry_key_prefix_exact'],'NATIVE_KEY_PREFIX_MISMATCH')
        pack['position_policy']='original right-padded arange; no offset or retokenization'
        return pack

    def panels(self, record):
        r=record['requested_rewrite']
        return {'R':[r['prompt'].format(r['subject'])], 'P':record['paraphrase_prompts'], 'N':record['neighborhood_prompts']}

    def evaluation_ids(self,prompt,target):
        tok=self.tokenizer; p=list(tok(prompt,add_special_tokens=True)['input_ids'])
        t=list(tok.encode(target if target.startswith(' ') else ' '+target,add_special_tokens=False))
        while t and t[0] in {tok.bos_token_id,tok.unk_token_id}:t=t[1:]
        require(p and t,'EMPTY_EVAL_TOKENS');return p,t

def row_tokens(pack,row):
    n=int(pack['tokens']['attention_mask'][row].sum())
    # KL's readout has no dependency on its future tokens. Preserve its prefix.
    if pack['row_kind'][row]=='kl':n=pack['lookup'][row]+1
    return {k:v[row,:n].clone() for k,v in pack['tokens'].items()}

def make_rows(pack):
    rows=[]
    for i,kind in enumerate(pack['row_kind']):
        tok=row_tokens(pack,i);n=len(tok['input_ids'])
        rows.append(dict(tokens=tok,lookup=pack['lookup'][i],target=pack['targets'][i,:n].clone(),
            kind=kind,request=pack['row_request'][i],global_row=i))
    return rows

def batches(rows,microbatch,pad_id,device):
    for start in range(0,len(rows),microbatch):
        group=rows[start:start+microbatch];width=max(len(r['tokens']['input_ids']) for r in group)
        ids=torch.full((len(group),width),pad_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
        for j,r in enumerate(group):
            n=len(r['tokens']['input_ids']);ids[j,:n]=r['tokens']['input_ids'].to(device);mask[j,:n]=r['tokens']['attention_mask'].to(device)
        yield group,dict(input_ids=ids,attention_mask=mask)

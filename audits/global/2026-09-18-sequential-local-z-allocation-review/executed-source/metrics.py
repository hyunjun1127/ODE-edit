"""Forward-only v2 online metrics; no repair/gradient/JVP/official P/N path.

Native-context E and canonical Current mean are deliberately separate. Only
E is a Current mean guard. Canonical strict/pair IDs and margins feed the
controller; full canonical rows remain local diagnostic evidence.
"""
from __future__ import annotations
from copy import deepcopy
import math
import time

import torch

from project.run_scripts.bg_tw_reference.ep_tw.model_adapter import EpisodeAdapter, TeacherStore


def normalized_requests(records):
    result=[dict(deepcopy(r['requested_rewrite']),case_id=r['case_id']) for r in records]
    for r in result:
        if not r['target_new']['str'].startswith(' '):r['target_new']['str']=' '+r['target_new']['str']
    return result


class OnlineMetrics:
    def __init__(self,model,writer_tok,canonical_tok,native_module,hp4,contexts,teacher):
        self.model,self.writer_tok,self.tok=model,writer_tok,canonical_tok
        self.module,self.hp,self.contexts,self.teacher=native_module,hp4,deepcopy(contexts),teacher
        self.device=next(model.parameters()).device
        # Read-only helpers supply exact canonical token grouping and fixed-W0
        # full-vocab KL. No residual episode is configured or differentiated.
        self._canonical=EpisodeAdapter(model,canonical_tok,'model.layers.4.mlp.down_proj.weight',teacher)
        self._native_objective_source='READ_ONLY_NATIVE_COMPUTE_Z_REWRITE_MASK_AND_LOSS_READOUT'

    @torch.no_grad()
    def training_E(self,records):
        records=list(records)
        if not records:raise ValueError('EMPTY_NATIVE_CURRENT_PANEL')
        hp,tok,module=self.hp,self.writer_tok,self.module
        rows=[];start=time.monotonic();contexts=0;tokens=0;scored=0
        lm_w=module.nethook.get_parameter(self.model,f'{hp.lm_head_module}.weight').T
        ln_f=module.nethook.get_module(self.model,hp.ln_f_module)
        try:lm_b=module.nethook.get_parameter(self.model,f'{hp.lm_head_module}.bias')
        except LookupError:lm_b=next(self.model.parameters()).new_zeros(self.model.config.vocab_size)
        with self._canonical._state_guard():
            for r in normalized_requests(records):
                target=tok(r['target_new']['str'],return_tensors='pt').to(self.device)['input_ids'][0]
                if target[0] in (tok.bos_token_id,tok.unk_token_id):target=target[1:]
                if target.numel()==0:raise ValueError('EMPTY_NATIVE_TARGET')
                rewrite=[c.format(r['prompt'])+tok.decode(target[:-1]) for group in self.contexts for c in group]
                inputs=tok([p.format(r['subject']) for p in rewrite+['{} is a']],return_tensors='pt',padding=True).to(self.device)
                targets=torch.full((len(rewrite),inputs['input_ids'].shape[1]),-100,device=self.device,dtype=torch.long)
                for i in range(len(rewrite)):
                    end=int(inputs['attention_mask'][i].sum());targets[i,end-len(target):end]=target
                with module.nethook.TraceDict(self.model,[hp.layer_module_tmp.format(hp.v_loss_layer)],
                        retain_input=False,retain_output=True) as trace:
                    self.model(**inputs)
                output=trace[hp.layer_module_tmp.format(hp.v_loss_layer)].output[0]
                if output.shape[1]!=targets.shape[1]:output=output.transpose(0,1)
                full=output[:len(rewrite)]
                logp=torch.log_softmax(ln_f(full)@lm_w.to(full.device)+lm_b.to(full.device),dim=2)
                values=logp.gather(2,torch.where(targets!=-100,targets,0).unsqueeze(2)).squeeze(2)
                loss=-(values*(targets!=-100).float()).sum(1)/target.size(0)
                if not torch.isfinite(loss).all():raise ValueError('NONFINITE_NATIVE_E')
                rows.append(dict(case_id=int(r['case_id']),context_nll=loss.cpu().tolist(),nll=float(loss.mean()),
                    target_token_ids=target.cpu().tolist(),native_context_count=len(rewrite)))
                contexts+=len(rewrite);tokens+=int(inputs['attention_mask'].sum());scored+=len(rewrite)*len(target)
        return dict(E=math.fsum(r['nll'] for r in rows)/len(rows),rows=rows,requests=len(rows),contexts=contexts,
            model_forwards=len(rows),input_tokens=tokens,scored_tokens=scored,seconds=time.monotonic()-start,
            reduction='NATIVE_TOKEN_MEAN_CONTEXT_MEAN_REQUEST_MEAN',gradients=0,
            loss_layer=hp.v_loss_layer,source_semantics=self._native_objective_source,
            KL_context_in_input=True,KL_penalty_in_E=False,decay_penalty_in_E=False)

    @torch.no_grad()
    def canonical(self,records):
        records=list(records)
        if not records:
            return dict(E=None,rows=[],strict_ids=[],pair_ids=[],pair_status='EMPTY',pair_available_ids=[],
                token_margins={},pair_margins={},counts=dict(forwards=0,input_tokens=0,scored_tokens=0))
        if len({int(r['case_id']) for r in records})!=len(records):raise ValueError('DUPLICATE_CANONICAL_ID')
        rows=[];old={};start=time.monotonic();counts=dict(forwards=0,new_forwards=0,old_forwards=0,input_tokens=0,scored_tokens=0)
        with self._canonical._state_guard():
            for key in ('target_new','target_true'):
                panel=[]
                for record in records:
                    rw=record['requested_rewrite']
                    if key not in rw or not rw[key] or not rw[key].get('str'):
                        if key=='target_new':raise ValueError('MISSING_CANONICAL_TARGET')
                        continue
                    item=deepcopy(record);item['requested_rewrite']['target_new']=deepcopy(rw[key]);panel.append(item)
                if not panel:continue
                for group,ids,mask,positions in self._canonical._current_groups(panel):
                    logits=self.model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
                    losses,values=self._canonical._current_values(logits,group,positions)
                    if not torch.isfinite(losses).all():raise ValueError('NONFINITE_CANONICAL_NLL')
                    for i,value in enumerate(values):
                        cid=value['case_id']
                        if key=='target_true':old[cid]=value['nll'];continue
                        scores=logits[i,positions[i],:]
                        target=torch.tensor(value['target_token_ids'],device=self.device)
                        other=scores.clone();other.scatter_(1,target[:,None],-torch.inf)
                        competitor=other.argmax(-1)
                        margins=scores.gather(1,target[:,None]).flatten()-scores.gather(1,competitor[:,None]).flatten()
                        if not torch.isfinite(margins).all():raise ValueError('NONFINITE_CANONICAL_MARGIN')
                        rows.append(dict(**value,new_nll=value['nll'],token_margins=margins.cpu().tolist(),
                            min_token_margin=float(margins.min()),competitor_ids=competitor.cpu().tolist()))
                    counts['forwards']+=1;counts['new_forwards' if key=='target_new' else 'old_forwards']+=1
                    counts['input_tokens']+=int(mask.sum());counts['scored_tokens']+=sum(len(r[3]) for r in group)
        for row in rows:
            row['old_nll']=old.get(row['case_id'])
            row['pair_margin']=None if row['old_nll'] is None else row['old_nll']-row['new_nll']
            row['pair_success']=row['pair_margin'] is not None and row['pair_margin']>0
        return dict(E=math.fsum(r['new_nll'] for r in rows)/len(rows),denominator=len(rows),rows=rows,
            strict_ids=[r['case_id'] for r in rows if r['all_tokens_correct']],
            pair_ids=[r['case_id'] for r in rows if r['pair_success']],pair_available_ids=sorted(old),
            pair_status='AVAILABLE' if len(old)==len(records) else 'PARTIAL' if old else 'NOT_AVAILABLE',
            token_margins={r['case_id']:r['min_token_margin'] for r in rows},
            pair_margins={r['case_id']:r['pair_margin'] for r in rows if r['pair_margin'] is not None},
            counts=counts,seconds=time.monotonic()-start,reduction='TOKEN_MEAN_CANONICAL_CONTEXT1_REQUEST_MEAN',
            strict_tie='TORCH_ARGMAX_FIRST_TOKEN_INDEX',pair_tie='FAILURE',
            layout='MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE',gradients=0)

    def base(self,role='S64'):
        # Dev128 is available to a post-selection observer, never score().
        return self._canonical.generic(role)

    def score(self,current,past):
        native=self.training_E(current);cur=self.canonical(current)
        old=self.canonical(past) if past else None
        base=self.base('S64')
        pair=lambda p:None if p is None or p['pair_status'] in ('EMPTY','NOT_AVAILABLE') else frozenset(p['pair_ids'])
        controller=dict(base_kl=base['D'],training_e=native['E'],canonical_e=cur['E'],
            current_strict=frozenset(cur['strict_ids']),current_pair=pair(cur),
            past_h=None if old is None else old['E'],past_strict=frozenset() if old is None else frozenset(old['strict_ids']),
            past_pair=pair(old),current_token_margins=cur['token_margins'],current_pair_margins=cur['pair_margins'],
            past_token_margins={} if old is None else old['token_margins'],past_pair_margins={} if old is None else old['pair_margins'])
        return dict(E=native['E'],H=controller['past_h'],B=base['D'],controller=controller,
            details=dict(training=native,current=cur,past=old,generic=base),official_P_N_access=0,
            Dev_access=0,gradients=0,canonical_Current_mean_is_guard=False)

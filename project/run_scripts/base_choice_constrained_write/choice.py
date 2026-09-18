"""Fixed W0 prefixes, full-vocabulary choice scans, pair-scalar factor VJPs."""
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import FullWeightLlamaOracle
from project.run_scripts.single_layer_edit_preserving_correction.binding import pack
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha
from .provenance import create_json

@torch.no_grad()
def raw_greedy(model,input_ids,max_new_tokens=16):
    eos=model.generation_config.eos_token_id or model.config.eos_token_id
    eos=[int(eos)] if isinstance(eos,int) else list(eos)
    if not eos:raise ValueError('ORIGINAL_EOS_REQUIRED')
    ids=list(input_ids);new=[];steps=[];start=time.monotonic()
    device=next(model.parameters()).device
    for step in range(max_new_tokens):
        x=torch.tensor([ids],device=device)
        # Physical decoder with original positions, no processor/KV reuse/chat wrapper.
        hidden=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,return_dict=True).last_hidden_state
        logits=model.lm_head(hidden[0,-1]).float()
        if not torch.isfinite(logits).all():raise FloatingPointError('NONFINITE_RAW_GREEDY')
        token=int(logits.argmax()) # PyTorch lowest ID on an exact tie.
        order=torch.argsort(logits,descending=True,stable=True)[:8]
        competitor=int(order[1]);margin=float(logits[token].double()-logits[competitor].double())
        steps.append(dict(token=token,competitor=competitor,margin=margin,logp=float(logits.log_softmax(0)[token]),
            top8_ids=order.cpu().tolist(),top8_logits=logits[order].cpu().tolist()))
        new.append(token);ids.append(token)
        if token in eos:break
    return dict(input_ids=list(input_ids),y0=new,steps=steps,original_eos_ids=eos,
        eos=bool(new and new[-1] in eos),censored=bool(len(new)==max_new_tokens and new[-1] not in eos),
        max_new_tokens=max_new_tokens,seconds=time.monotonic()-start,raw_argmax=True,
        tie='lowest token ID',logits_processors=[],KV_cache=False)

def build_capsules(rt,inputs,directory):
    directory=Path(directory);capsules=[]
    if not torch.equal(rt.W.detach().cpu(),rt.W0):raise ValueError('CAPSULE_NOT_W0')
    for row in inputs:
        answer=raw_greedy(rt.model,row['input_ids'])
        item={**row,**answer,'W0':rt.identity['W0']}
        # Prompt+answer[:-1], includes actual generated EOS as label, no fake EOS.
        item['tf_input_ids']=item['input_ids']+item['y0'][:-1]
        item['positions']=list(range(len(item['input_ids'])-1,len(item['tf_input_ids'])))
        create_json(directory/row['role']/f'{row["ordinal"]:03d}.json',item);capsules.append(item)
        if row['ordinal']%32==0:print('CAPSULE_PROGRESS',row['role'],row['ordinal'],flush=True)
    rt.guard();return capsules

class ChoiceOracle(FullWeightLlamaOracle):
    def __init__(self,rt,capsules):
        self.capsules=capsules
        super().__init__(rt.model,[pack(r['tf_input_ids']) for r in capsules],require_reference_length=None)
        rt.oracles.append(self)
        self.pair_work=dict(pair_scalar_backwards=0,pair_backward_seconds=0.,full_vocab_scan_documents=0,
                           full_vocab_scan_positions=0,gradient_full_dense_stored=0,
                           pair_suffix_forwards=0,pair_forward_input_tokens=0,pair_backward_input_tokens=0,
                           pair_head_rows=0)

    @torch.no_grad()
    def scan(self,weight,tau,exposed=()):
        docs=[]
        for i,row in enumerate(self.capsules):
            logits=self.logits_at(i,weight,row['positions'])
            if not torch.isfinite(logits).all():raise FloatingPointError('NONFINITE_CHOICE_SCAN')
            targets=torch.tensor(row['y0'],device=self.device);ix=torch.arange(len(targets),device=self.device)
            choice=logits.argmax(-1);lp=logits.log_softmax(-1)[ix,targets]
            desired=logits[ix,targets].double();other=logits.clone();other[ix,targets]=-torch.inf
            competitor=other.argmax(-1);margin=desired-other[ix,competitor].double()
            positions=[]
            for j,(p,y) in enumerate(zip(row['positions'],row['y0'],strict=True)):
                base=row['steps'][j];kappa=min(base['margin'],tau)
                positions.append(dict(position=p,target=y,competitor=int(competitor[j]),choice=int(choice[j]),
                    margin=float(margin[j]),kappa=kappa,mu=float(margin[j])-kappa,
                    logp=float(lp[j]),d=base['logp']-float(lp[j]),preserved=int(choice[j])==y,
                    outside_base_top8=int(competitor[j]) not in base['top8_ids']))
            worst=min(positions,key=lambda x:(x['mu'],x['position'],x['competitor']))
            retained=[]
            for pair in exposed:
                if pair['index']==i:
                    j=row['positions'].index(pair['position'])
                    raw_margin=float(logits[j,pair['target']].double()-logits[j,pair['competitor']].double())
                    retained.append(dict(pair_id=pair['pair_id'],margin=raw_margin,mu=raw_margin-pair['kappa']))
            docs.append(dict(index=i,source_row_id=row['source_row_id'],role=row['role'],positions=positions,retained_pairs=retained,
                worst=worst,all_choices=all(x['preserved'] for x in positions),eos=row['eos'],censored=row['censored']))
            self.pair_work['full_vocab_scan_documents']+=1;self.pair_work['full_vocab_scan_positions']+=len(positions)
        return dict(documents=docs,document_count=len(docs),positions=sum(len(x['positions']) for x in docs),
            all_choices=all(d['all_choices'] for d in docs),sequence_retained=sum(d['all_choices'] for d in docs),
            token_flips=sum(not p['preserved'] for d in docs for p in d['positions']),weight_sha256=tensor_sha(weight))

    def pair_factor(self,index,weight,position,target,competitor):
        """Fresh two-row backward through all suffix tokens; K remains full input."""
        self._guard();cache=self.caches[index];w=self._weight(weight)
        packed=self._on_device(cache.packed)
        with torch.no_grad():z=F.linear(cache.keys.to(self.device),w)
        leaf=z.detach().requires_grad_(True)
        hidden=cache.residual.to(self.device)+leaf;args=self._args(hidden,packed)
        begin=time.monotonic()
        for layer in self.decoder.layers[5:]:hidden=layer(hidden,**args)[0]
        hidden=self.decoder.norm(hidden)
        two=F.linear(hidden[0,position],self.model.lm_head.weight[[target,competitor]])
        value=two[0]-two[1]
        a,=torch.autograd.grad(value,leaf)
        if not torch.isfinite(a).all() or not torch.isfinite(value):raise FloatingPointError('NONFINITE_PAIR_VJP')
        self._sync();self.pair_work['pair_scalar_backwards']+=1
        self.pair_work['pair_backward_seconds']+=time.monotonic()-begin
        self.pair_work['pair_suffix_forwards']+=1
        self.pair_work['pair_forward_input_tokens']+=cache.packed['input_ids'].numel()
        self.pair_work['pair_backward_input_tokens']+=cache.packed['input_ids'].numel()
        self.pair_work['pair_head_rows']+=2
        A=a[0].detach().T.cpu().contiguous();K=cache.keys[0].T
        self._guard()
        return A,K,float(value.detach())

    def direct_pair(self,index,weight,position,target,competitor):
        with self.physical_weight(weight,gradient=True) as leaf:
            hidden=self._physical_hidden(index)
            two=F.linear(hidden[0,position],self.model.lm_head.weight[[target,competitor]])
            scalar=two[0]-two[1]
            g,=torch.autograd.grad(scalar,leaf)
            if not torch.isfinite(g).all():raise FloatingPointError('NONFINITE_DIRECT_AD')
            return float(scalar.detach()),g.detach().double().cpu()

    @torch.no_grad()
    def pair_value(self,index,weight,position,target,competitor):
        hidden=self.hidden(index,weight)
        two=F.linear(hidden[0,position],self.model.lm_head.weight[[target,competitor]])
        return float((two[0]-two[1]).detach())

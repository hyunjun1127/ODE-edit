"""Full-vocabulary R512 + own at-write history; one accumulated GPU gradient.

Readonly inherited Llama suffix/cache implementations; no original controller,
quality guards, per-document dense D2H, or persisted edited state.
"""
from __future__ import annotations
import contextlib, copy, hashlib, json, time
import numpy as np
import torch
from project.run_scripts.en_execution_reuse.generated_teacher import GeneratedTeacherStore
from project.run_scripts.en_execution_reuse.generated_oracle import GeneratedReferenceOracle
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import FullWeightLlamaOracle, signed_forward_kl
from project.run_scripts.single_layer_edit_preserving_correction.binding import pack, _token_contracts
from project.run_scripts.single_layer_mechanism_first.history import active_history, receive_all, registry_status

class Objective:
    def __init__(self, model, root, inputs):
        from pathlib import Path
        root=Path(root); raw=(root/'manifest.json').read_bytes(); manifest=json.loads(raw)
        self.store=GeneratedTeacherStore(root,root/'manifest.json',expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),inputs_path=inputs,expected_binding=manifest['binding'],require_upstream_cache=True,verify_payloads=False)
        self.ref=GeneratedReferenceOracle(model,self.store,max_cache_bytes=19*2**30)
        self.history={};self.model=model;self.sweeps=[]
        self.counts=dict(reference_gradient=0,reference_candidate=0,reference_observer=0,history_gradient=0,history_candidate=0,history_teacher_requests=0,history_teacher_bytes=0,technical_sweeps=0)

    def rebind(self, expected):
        self.ref.acknowledge_selected_write(expected)
        for h in self.history.values():h['oracle'].acknowledge_selected_write(expected)

    def capture_history(self, arm, records, tokenizer, weight):
        """At-write target distributions live in RAM; no weight/update save."""
        start=time.monotonic();contracts=_token_contracts();packs=[];rows=[]
        for rec in records:
            r=rec['requested_rewrite'];prefix=contracts.prompt_token_ids(tokenizer,r['prompt'].format(r['subject']));labels=contracts.target_token_ids(tokenizer,r['target_new']['str'])
            packs.append(pack(prefix+labels[:-1]));rows.append(dict(case_id=rec['case_id'],positions=list(range(len(prefix)-1,len(prefix)+len(labels)-1)),labels=labels))
        oracle=FullWeightLlamaOracle(self.model,packs,require_reference_length=None)
        teachers=[];bindings=[]
        with torch.no_grad():
            leaf=weight.to(oracle.device,dtype=torch.float32)
            for i,row in enumerate(rows):
                hidden=oracle.suffix_hidden(i,leaf);lp=oracle._head(hidden,torch.tensor(row['positions'])).log_softmax(-1)
                assert bool(torch.isfinite(lp).all());teacher_cpu=lp.cpu();teachers.append(teacher_cpu)
                bindings.append(dict(case_id=row['case_id'],positions=row['positions'],labels=row['labels'],input_identity=oracle.caches[i].input_identity,teacher_sha256=hashlib.sha256(teacher_cpu.contiguous().numpy().tobytes()).hexdigest(),teacher_shape=list(teacher_cpu.shape),teacher_dtype=str(teacher_cpu.dtype)))
                del hidden,lp
        key=(arm,len(self.history));self.history[key]=dict(arm=arm,oracle=oracle,rows=rows,teachers=teachers)
        size=sum(x.numel()*x.element_size() for x in teachers);self.counts['history_teacher_requests']+=len(rows);self.counts['history_teacher_bytes']+=size
        return dict(key=list(key),bindings=bindings,requests=len(rows),teacher_bytes=size,seconds=time.monotonic()-start,checkpoint=False,teacher_policy='FINAL_AT_WRITE_FP32_FULLVOCAB_CANONICAL_NEW_TF')

    def evaluate(self, weight, *, active_ids=(),arm=None, gradient=False, role='R512',kind='candidate',indices=None,route='cached'):
        """Block means summed equally. All train documents except explicit T0."""
        start=time.monotonic();technical=indices is not None
        if role=='Dev128' and gradient:raise ValueError('DEV_OBSERVER_ONLY')
        selected=tuple(indices) if technical else self.store.indices(role)
        if technical and not 1<=len(selected)<=4:raise ValueError('T0_REFERENCE_FOUR_MAX')
        active=set(active_ids);hist=[]
        for h in self.history.values():
            if h['arm']==arm:
                hist.extend((h,i) for i,row in enumerate(h['rows']) if row['case_id'] in active)
        if len(hist)!=len(active):raise ValueError('HISTORY_MEMBERSHIP_MISMATCH')
        accumulation=torch.zeros(weight.shape,dtype=torch.float64,device=self.ref.device) if gradient else None
        result={'reference':[],'history':[]};values={}
        context=self.ref.physical_weight(weight.to(self.ref.device),gradient=gradient) if route=='physical' else contextlib.nullcontext(weight.to(self.ref.device).detach().requires_grad_(gradient))
        with context as leaf, torch.set_grad_enabled(gradient):
            for block,entries in [('reference',selected),('history',hist)]:
                for entry in entries:
                    if block=='reference':
                        oracle=self.ref;i=entry;cap=self.ref._capsules[i];positions=cap['score_positions'];labels=cap['y0'];teacher=self.ref._teacher(i);ident={'index':i,'ordinal':cap['ordinal'],'role':cap['role']}
                    else:
                        h,i=entry;oracle=h['oracle'];row=h['rows'][i];positions=row['positions'];labels=row['labels'];teacher=h['teachers'][i].to(oracle.device);ident={'case_id':row['case_id']}
                    hidden=oracle.suffix_hidden(i,leaf) if route=='cached' else oracle._physical_hidden(i)
                    logits=oracle._head(hidden,torch.tensor(positions));loss=signed_forward_kl(logits,teacher)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('NONFINITE_OBJECTIVE')
                    with torch.no_grad():
                        lab=torch.tensor(labels,device=logits.device);lp=logits.log_softmax(-1);pred=logits.argmax(-1)
                        desired=logits.gather(1,lab[:,None]).squeeze(1);masked=logits.clone();masked.scatter_(1,lab[:,None],-torch.inf);margin=desired-masked.max(-1).values
                        row=dict(**ident,loss=float(loss.detach()),positions=len(positions),choice_mismatches=int((pred!=lab).sum()),target_logp_mean=float(lp.gather(1,lab[:,None]).double().mean()),minimum_choice_margin=float(margin.min()),phi=float(torch.clamp(-margin.min().double(),min=0).square()))
                        del lp,pred,masked,margin,desired,lab
                    if gradient:
                        g,=torch.autograd.grad(loss,leaf);assert g.dtype==torch.float32 and bool(torch.isfinite(g).all());accumulation.add_(g.double(),alpha=1/len(entries));del g
                    result[block].append(row);del hidden,logits,loss,teacher
                values[block]=sum(r['loss'] for r in result[block])/len(entries) if entries else 0.
        G=accumulation.cpu() if gradient else None
        if technical:self.counts['technical_sweeps']+=1
        else:
            self.counts['reference_'+('gradient' if gradient else 'observer' if kind=='observer' else 'candidate')]+=1
            if hist:self.counts['history_'+('gradient' if gradient else 'candidate')]+=1
        receipt=dict(L_R=values['reference'],L_H=values['history'],J=sum(values.values()),rows=result,gradient=gradient,role=role,kind=kind,reference_documents=len(selected),history_requests=len(hist),reference_positions=sum(x['positions'] for x in result['reference']),seconds=time.monotonic()-start,dense_gradient_D2H=int(gradient),route=route,technical=technical)
        self.sweeps.append({k:v for k,v in receipt.items() if k!='rows'})
        return receipt['J'],G,receipt

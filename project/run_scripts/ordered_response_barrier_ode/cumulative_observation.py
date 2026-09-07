"""Observation-only historical response ledger and exact all-seen evaluation.

Targets are frozen at their original edit entry, never recomputed. Evaluators
run after commit only. Layer response probes see the same live FP32 overlay,
but cannot return a value to the controller. No additional writer/key/solve.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,os,shutil,time
from pathlib import Path
from typing import Any
import torch
from .contracts import TechnicalBoundary
from .fp32_overlay import tensor_set_sha256,tensor_sha256
from .preflight import canonical_hash
from .artifacts import reduce_evaluation_payload


def check(ok,why):
    if not ok:raise TechnicalBoundary('CUMULATIVE_OBSERVER: '+why)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def publish(path,payload):
    body=dict(payload);body['identity_sha256']=canonical_hash(body)
    with path.open('x') as f:json.dump(body,f,sort_keys=True,allow_nan=False,separators=(',',':'))
    return {'path':path.name,'sha256':sha(path),'identity_sha256':body['identity_sha256']}


def signature(family):
    tensors=list(family.parameters.values())
    cache=getattr(family.module,'cache_c',None)
    if family.family=='AlphaEdit' and isinstance(cache,torch.Tensor):tensors.append(cache)
    if family.family=='MEMIT':tensors+=list(family.module.COV_CACHE.values())
    return tuple((id(t),t.data_ptr(),t._version,str(t.dtype)) for t in tensors)


def response_rows(target,origin,before,after,action):
    """Per-request geometry; batch-level action is not an independent sample."""
    target,origin,before,after=(t.detach().double().cpu() for t in (target,origin,before,after))
    r0=target-origin;r=target-before;y=after-before;ra=target-after
    den=(r0*r0).sum(0);rden=(r*r).sum(0);out=[]
    for i in range(target.shape[1]):
        valid=float(den[i])>0;active=float(rden[i])>0
        aligned=float((y[:,i]*r[:,i]).sum()/rden[i]) if active else None
        ortho=float(torch.linalg.vector_norm(y[:,i]-aligned*r[:,i])) if active else None
        reduction=float(torch.linalg.vector_norm(r[:,i])-torch.linalg.vector_norm(ra[:,i]))
        out.append({'origin_zero':not valid,'current_residual_zero':not active,
            'q_before':float(torch.sqrt(rden[i]/den[i])) if valid else None,
            'q_after':float(torch.linalg.vector_norm(ra[:,i])/torch.sqrt(den[i])) if valid else None,
            'target_aligned_fraction_of_remaining_residual':aligned,
            'orthogonal_response_over_origin':ortho/float(torch.sqrt(den[i])) if valid and active else None,
            'normalized_potential_reduction':float((rden[i]-(ra[:,i]*ra[:,i]).sum())/(2*den[i])) if valid else None,
            'residual_norm_reduction':reduction,'residual_reduction_per_parameter_action':reduction/action if action>0 else None,
            'zero_action':action==0,'response_norm':float(torch.linalg.vector_norm(y[:,i]))})
    return out


def command_rows(cohort,before,after,command,prediction=None):
    a=command.detach().double().cpu();y=after.detach().double().cpu()-before.detach().double().cpu();e=a-y
    pm=None if prediction is None else y-prediction.detach().double().cpu()
    rows=[]
    for i,r in enumerate(cohort.sealed):
        an=float(torch.linalg.vector_norm(a[:,i]));yn=float(torch.linalg.vector_norm(y[:,i]))
        scale=float(torch.linalg.vector_norm(cohort.target[:,i].double()-cohort.origin[:,i].double()))
        en=float(torch.linalg.vector_norm(e[:,i]));rho=float(torch.dot(a[:,i],y[:,i]))/(an*an) if an else None
        rows.append({'request_sha256':r['request_sha256'],'A_norm':an,'Y_norm':yn,'E_norm':en,
            'A_over_origin':an/scale if scale else None,'Y_over_origin':yn/scale if scale else None,'E_over_origin':en/scale if scale else None,
            'M_norm':float(torch.linalg.vector_norm(pm[:,i])) if pm is not None else None,
            'M_over_origin':float(torch.linalg.vector_norm(pm[:,i]))/scale if pm is not None and scale else None,
            'cos_A_Y':float(torch.dot(a[:,i],y[:,i]))/(an*yn) if an and yn else None,
            'rho_command':rho,'E_parallel_coefficient':1-rho if rho is not None else None,
            'tau_command':float(torch.linalg.vector_norm(y[:,i]-rho*a[:,i]))/an if an else None,
            'zero_command':an==0,'origin_zero':scale==0})
    return rows


@dataclass
class Cohort:
    batch:int
    requests:tuple
    records:tuple
    sealed:tuple
    target:torch.Tensor
    origin:torch.Tensor
    last:torch.Tensor
    immediate:torch.Tensor|None=None


class CumulativeObserver:
    def __init__(self,root:Path,arm:str):
        self.root=root/f'arm-{arm}'/'cumulative';self.root.mkdir(parents=True,exist_ok=False)
        self.arm=arm;self.cohorts=[];self.members=[];self.layer_path={l:0. for l in range(4,9)}
        self.layer_squared={l:0. for l in range(4,9)};self.original_global=None;self.official_original=None
        self.forward_probes=0;self.layer_visits=0;self.evaluator_cohorts=0
        self.builds={};self.build_observations=[]
        self.command_members=[];self.probe_seconds=0.
        self.workload={l:0. for l in range(4,9)}
        self.progress={l:0. for l in range(4,9)}
        self.stable_action={l:0. for l in range(4,9)}

    def capture(self,f,c):
        before=signature(f);started=time.perf_counter()
        with torch.no_grad():
            value=f._repr_tools.get_reprs_at_word_tokens(model=f.model,tok=f.tokenizer,layer=8,
                context_templates=[str(r['prompt']) for r in c.requests],words=[str(r['subject']) for r in c.requests],
                module_template=f.hparams.layer_module_tmp,track='out',subtoken=str(f.hparams.fact_token).removeprefix('subject_'))
            value=value.T.detach().to(device='cpu',dtype=torch.float32).contiguous()
        check(signature(f)==before,'activation probe mutated state')
        check(bool(torch.isfinite(value).all()) and value.shape==c.target.shape,'terminal probe shape/finite')
        self.probe_seconds+=time.perf_counter()-started
        self.forward_probes+=1;return value

    def begin(self,f,idx,batch,fixed_z):
        self.family=f;self.batch=idx+1;self.visit=0
        self.builds={};self.build_observations=[];self.command_members=[]
        if self.original_global is None:self.original_global={k:t.detach().cpu().clone() for k,t in f.parameters.items()}
        with torch.no_grad():origin=f.terminal()
        c=Cohort(idx+1,tuple(f.requests),tuple(f.endpoint_records),tuple(batch),fixed_z.values.detach().cpu().clone(),origin.clone(),origin.clone())
        self.cohorts.append(c)
        for old in self.cohorts[:-1]:old.last=self.capture(f,old)
        self.batch_entry={c.batch:c.last.clone() for c in self.cohorts}
        self.step_path=self.root/f'layer-response-B{self.batch:02d}.jsonl'
        with self.step_path.open('x'):pass
        targetpath=self.root/f'frozen-target-origin-B{self.batch:02d}.pt'
        with targetpath.open('xb') as handle:torch.save({'target':c.target,'origin':c.origin,'request_sha256':[r['request_sha256'] for r in batch]},handle)
        self.members.append({'path':targetpath.name,'sha256':sha(targetpath),'target_sha256':tensor_sha256(c.target),'origin_sha256':tensor_sha256(c.origin)})

    def on_build(self,build,keys):
        current=keys.detach().cpu();previous=self.builds.get(build.layer)
        row={'layer':build.layer,'state_version':build.built_state_version,'keys_sha256':tensor_sha256(current),
             'keys_norm':float(torch.linalg.vector_norm(current.double())), 'factor_rank':build.factor_rank,
             'solve_backward_error':build.solve_backward_error,'build_identity':build.build_identity,
             'comparison':'SAME_LAYER_PREVIOUS_VISIT_NOT_SAME_ENTRY_COUNTERFACTUAL'}
        if previous is not None:
            k,l,r=previous;kn=float(torch.linalg.vector_norm(k.double()))
            row['key_revisit_relative_difference']=float(torch.linalg.vector_norm(current.double()-k.double()))/kn if kn else None
            left,right=build.left.double(),build.right.double();l,r=l.double(),r.double()
            norm2=lambda a,b:float(torch.sum((a.T@a)*(b.T@b)))
            old=norm2(l,r);cross=float(torch.sum((left.T@l)*(right.T@r)))
            row['writer_revisit_relative_difference']=(max(0.,norm2(left,right)+old-2*cross)/old)**.5 if old>0 else None
        self.builds[build.layer]=(current.clone(),build.left.clone(),build.right.clone())
        self.build_observations.append(row)

    def transition(self,*,layer,sweep,visit,terminal,record,terminal_before,command,prediction):
        action=float(record.applied_update_frobenius)
        self.workload[layer]+=record.euler_alpha
        self.progress[layer]+=record.actual_reduction
        self.stable_action[layer]+=record.resolution_stable_path_increment_frobenius_squared
        # Current-edit command/realization decomposition; historical cohorts have
        # no writer command and are kept separate in the response ledger.
        rows=command_rows(self.cohorts[-1],terminal_before,terminal,command,prediction)
        self.command_members.append(publish(self.root/f'command-realization-B{self.batch:02d}-visit{visit:02d}.json',{
            'schema':'orbode.command-realization.v1','layer':layer,'sweep':sweep,'visit':visit,'rows':rows,
            'g':record.g,'r':record.r,'a_star':max(0.,record.g)/record.r if record.r is not None and record.r>0 else None,
            'u':record.coefficient_u,'alpha':record.euler_alpha,'action_step_frobenius_squared':record.applied_update_frobenius_squared,
            'resolution_stable_action_frobenius':record.resolution_stable_path_increment_frobenius_squared,
            'predicted_reduction':record.predicted_reduction,'actual_reduction':record.actual_reduction,
            'native_Creg_action':'UNBOUND_IN_ACCEPTED_SOURCE_NOT_FROBENIUS_EQUIVALENT'}))
        self._observe(layer,sweep,visit,action,terminal)

    def _observe(self,layer,sweep,visit,action,current=None):
        f=self.family;before=signature(f)
        self.layer_path[layer]+=action;self.layer_squared[layer]+=action**2
        with self.step_path.open('a') as out:
            for c in self.cohorts:
                after=current.detach().cpu() if c is self.cohorts[-1] and current is not None else self.capture(f,c)
                rows=response_rows(c.target,c.origin,c.last,after,action)
                for request,row in zip(c.sealed,rows,strict=True):
                    value={'arm':self.arm,'at_batch':self.batch,'cohort_batch':c.batch,'request_sha256':request['request_sha256'],
                        'layer':layer,'sweep':sweep,'visit':visit,'parameter_action_Frobenius':action,
                        'parameter_action_definition':'ACTUAL_STORED_WEIGHT_DIFFERENCE' if self.arm=='O' else 'NOMINAL_LOW_RANK_ALPHA_B_NOT_FP32_ROUNDING_DIFFERENCE',
                        'cumulative_layer_path_magnitude':self.layer_path[layer],**row}
                    out.write(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n')
                c.last=after.clone()
        check(before==signature(f),'transition observer state mutation')
        self.layer_visits+=1;self.visit+=1

    def install_official(self,f):
        check(self.official_original is None,'nested Official patch')
        self.official_original=f.module.get_module_input_output_at_words;self.official_calls=0
        self.previous_weights={k:v.detach().cpu().clone() for k,v in f.parameters.items()}
        def observed(*args,**kwargs):
            result=self.official_original(*args,**kwargs)
            j=self.official_calls;check(j<5,'Official layer-loop count >5')
            if j>0:self._official_step(f,3+j)
            self.official_calls+=1
            return result
        f.module.get_module_input_output_at_words=observed

    def _official_step(self,f,layer):
        name=next(k for k in f.parameters if f'.{layer}.' in k)
        with torch.no_grad():action=float(torch.linalg.vector_norm(f.parameters[name].detach().cpu().double()-self.previous_weights[name].double()))
        cohort=self.cohorts[-1];after=self.capture(f,cohort)
        command=(cohort.target-cohort.last)/(9-layer)
        self.command_members.append(publish(self.root/f'command-realization-B{self.batch:02d}-visit{layer-4:02d}.json',{
            'schema':'orbode.command-realization.v1','layer':layer,'sweep':0,'visit':layer-4,
            'command_semantics':'OFFICIAL_CURRENT_RESIDUAL_OVER_REMAINING_LAYERS',
            'rows':command_rows(cohort,cohort.last,after,command),
            'action_step_frobenius_squared':action**2,
            'native_Creg_action':'UNBOUND_IN_ACCEPTED_SOURCE_NOT_FROBENIUS_EQUIVALENT'}))
        self._observe(layer,0,layer-4,action,after)

    def finish_official(self,f):
        check(self.official_calls==5,'Official layer-loop count !=5')
        self._official_step(f,8)
        self.restore_official(f)

    def restore_official(self,f):
        if self.official_original is not None:
            original=self.official_original;f.module.get_module_input_output_at_words=original
            check(f.module.get_module_input_output_at_words is original,'Official function restoration')
            self.official_original=None;self.previous_weights=None

    def committed(self,f,commit):
        """Same frozen W_t evaluates ALL seen cohorts; includes t=1000 exactly once."""
        from .counterfact_locality_evaluator import evaluate_counterfact_with_canonical_ns
        before=signature(f);weight_sha=tensor_set_sha256(f.parameters);cache_sha=f.method_state_identity();started=time.perf_counter()
        check(weight_sha==commit['committed_weight_sha256'],'cumulative W_t binding')
        directory=self.root/f'W-after-B{self.batch:02d}';directory.mkdir(exist_ok=False)
        evaluations=[];geometry=[]
        for c in self.cohorts:
            with torch.no_grad():raw=evaluate_counterfact_with_canonical_ns(f.model,f.tokenizer,c.records,device=f.device,microbatch_size=16)
            order=[str(r['request_sha256']) for r in c.sealed]
            reduced=reduce_evaluation_payload(raw,case_ids=[int(r['case_id']) for r in c.sealed],request_sha256=order,request_order_sha256=canonical_hash(order))
            evaluations.append(publish(directory/f'cohort-{c.batch:02d}-evaluation.json',{
                'schema':'orbode.cumulative-cohort-evaluation.v1','evaluation_type':'CHECKPOINT_W_ON_ALL_SEEN_REQUESTS',
                'at_batch':self.batch,'cohort_batch':c.batch,'W_sha256':weight_sha,'evaluation':reduced}))
            self.evaluator_cohorts+=1
            after=self.capture(f,c)
            rows=response_rows(c.target,c.origin,self.batch_entry[c.batch],after,0.)
            if c is self.cohorts[-1]:c.immediate=after.clone()
            for request,row,j in zip(c.sealed,rows,range(len(rows)),strict=True):
                row.update(request_sha256=request['request_sha256'],cohort_batch=c.batch,
                    distance_from_own_immediate_activation=float(torch.linalg.vector_norm(after[:,j].double()-c.immediate[:,j].double())))
                geometry.append(row)
        geom=publish(directory/'historical-residual-retention.json',{'schema':'orbode.historical-residual-retention.v1',
            'at_batch':self.batch,'request_count':100*self.batch,'rows':geometry})
        builds=publish(directory/'key-writer-revisit-drift.json',{'schema':'orbode.key-writer-revisit-drift.v1','rows':self.build_observations})
        layer=[]
        for name,w in f.parameters.items():
            l=int(name.split('.')[2]);now=w.detach().cpu().double()
            net=float(torch.linalg.vector_norm(now-self.original_global[name].double()))
            batchnet=float(torch.linalg.vector_norm(now-f.w0[name].detach().cpu().double()))
            layer.append({'layer':l,'batch_net_update_magnitude':batchnet,'net_from_original_W0_magnitude':net,
                'path_magnitude_sum':self.layer_path[l],'squared_step_norm_sum':self.layer_squared[l],
                'dynamic_workload_sum_h_u':self.workload[l] if self.arm!='O' else None,
                'dynamic_signed_potential_progress_sum':self.progress[l] if self.arm!='O' else None,
                'dynamic_resolution_stable_frobenius_action':self.stable_action[l] if self.arm!='O' else None})
        # Exact evaluation-only recovery, not a resume checkpoint: non-edited weights remain model-original.
        check(shutil.disk_usage(self.root).free>128*2**30,'storage reserve <128GiB')
        cp=directory/'edited-weights-fp32.pt'
        with cp.open('xb') as handle:torch.save({k:w.detach().cpu().clone() for k,w in f.parameters.items()},handle)
        restored=torch.load(cp,map_location='cpu',weights_only=True)
        check(set(restored)==set(f.parameters),'checkpoint member keys')
        check(all(v.dtype==torch.float32 and torch.equal(v,f.parameters[k].detach().cpu()) for k,v in restored.items()),'checkpoint exact recoverability')
        checkpoint_tensor_sha256={k:tensor_sha256(v) for k,v in restored.items()}
        del restored
        check(signature(f)==before and tensor_set_sha256(f.parameters)==weight_sha and f.method_state_identity()==cache_sha,'evaluation/checkpoint state mutation')
        all_order=[r['request_sha256'] for c in self.cohorts for r in c.sealed]
        receipt={'schema':'orbode.cumulative-checkpoint.v1','status':'CUMULATIVE_CHECKPOINT_VALID',
            'evaluation_type':'CHECKPOINT_W_ON_ALL_SEEN_REQUESTS','arm':self.arm,'at_batch':self.batch,
            'seen_requests':100*self.batch,'rewrite_prompts':100*self.batch,'rephrase_prompts':200*self.batch,'locality_prompts':1000*self.batch,
            'request_order_sha256':canonical_hash(all_order),'weight_sha256':weight_sha,'method_state_sha256':cache_sha,
            'checkpoint_file':cp.name,'checkpoint_sha256':sha(cp),'checkpoint_bytes':cp.stat().st_size,
            'checkpoint_tensor_sha256':checkpoint_tensor_sha256,'checkpoint_cpu_reload_exact':True,
            'checkpoint_scope':'FIVE_EDITED_FP32_WEIGHTS_PLUS_PINNED_ORIGINAL_MODEL_EVALUATION_ONLY',
            'cache_state_recovery':'HASH_ONLY_NOT_EDIT_RESUME','evaluation_members':evaluations,'geometry_member':geom,
            'key_writer_drift_member':builds,'command_members_relative_to_cumulative_root':self.command_members,
            'layer_summary':layer,'layer_response_file':str(self.step_path.relative_to(self.root)),
            'layer_response_sha256':sha(self.step_path),'layer_visits_this_batch':self.visit,
            'observer_terminal_cohort_calls_cumulative':self.forward_probes,'evaluator_cohort_calls_cumulative':self.evaluator_cohorts,
            'observer_terminal_probe_seconds_cumulative':self.probe_seconds,
            'observer_compute_z_writer_key_solve_backward_history_mutation':0,'controller_feedback':0,
            'weight_pointer_version_bytes_unchanged':True,'cache_unchanged':True,'wall_seconds':time.perf_counter()-started,
            'full_fp32':True,'nonfinite':0,'duplicate':0,'imputation':0,'scientific_promotion':False}
        binding=publish(directory/'receipt.json',receipt)
        binding['path']=str(directory.relative_to(self.root))+'/'+binding['path']
        self.members.append(binding)
        return {**binding,'root':str(self.root)}

    def terminal(self):
        return publish(self.root/'terminal.json',{'schema':'orbode.cumulative-arm-terminal.v1','arm':self.arm,
            'checkpoints':len(self.cohorts),'expected_seen_evaluations':5500,'members':self.members,
            'layer_visits':self.layer_visits,'evaluator_cohort_calls':self.evaluator_cohorts,
            'source_targets_recomputed':0,'controller_feedback':0})

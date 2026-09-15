"""Saved-episode diagnostic only, followed by an externally fail-closed runner.

This module never imports a native fitter/target/solver. All diagnostic rows,
gradients and perturbations are private, create-once evidence, not publication.
"""
import argparse
import copy
from dataclasses import fields
import json
import os
from pathlib import Path
import resource
import time
import traceback

from .control import identity, save, sha, verify_dispatch
from .repair_control import verify_repair


def verify_members(lock):
    """Reuse prior full SHA for unchanged heavy assets; hash new source fully."""
    stats={x['path']:x for x in lock['member_stats']}
    fresh=reused=0
    for m in lock['members']:
        p=Path(m['path']);s=p.stat();expected=stats[str(p)]
        assert dict(path=str(p),dev=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size)==expected,('MEMBER_STAT_DRIFT',str(p))
        assert s.st_size==m['bytes']
        if p.is_relative_to(lock['source_root']) or s.st_size < 8*(1<<20):
            assert sha(p)==m['sha256'],('MEMBER_SHA_DRIFT',str(p));fresh+=1
        else:reused+=1
    assert identity(lock['source_archive']['path'])==lock['source_archive']
    verify_dispatch(lock['dispatch']['path']);verify_repair(lock['repair_dispatch']['path'])
    return dict(new_full_sha_members=fresh,prior_full_sha_stable_stat_reuse=reused,
                new_whole_model_or_teacher_rehash=False)


def recorder(root):
    """Split tensor leaves from JSON; synchronous durable save before return."""
    import torch
    from .repair_checks import _save_json, _save_tensors
    root=Path(root)
    def record(stage,payload):
        tensors={}
        def split(value,path):
            if isinstance(value,torch.Tensor):
                tensors[path]=value.detach().cpu().clone()
                return dict(tensor_key=path,shape=list(value.shape),dtype=str(value.dtype))
            if isinstance(value,dict):return {str(k):split(v,path+'/'+str(k)) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [split(v,path+'/'+str(i)) for i,v in enumerate(value)]
            return value
        description=split(payload,'payload')
        member=_save_tensors(root/(stage+'.pt'),tensors) if tensors else None
        _save_json(root/(stage+'.json'),dict(payload=description,tensors=member,
            recording='BEFORE_NEXT_STAGE_OR_FAILURE',local_only=True))
    return record


def config_from_lock(lock):
    from .repair_checks import RepairNumerics
    names={f.name for f in fields(RepairNumerics)}
    data={k:v for k,v in lock['repair_numerics'].items() if k in names}
    data['factors']=tuple(data['factors'])
    config=RepairNumerics(**data)
    assert json.loads(json.dumps(config.to_dict()))==lock['repair_numerics'],'NUMERICAL_LOCK_DRIFT'
    return config


def validate_repair_pass(lock):
    """No model loading or fresh science until exact saved-episode PASS exists."""
    p=Path(lock['repair_pass_path']);receipt=json.loads(p.read_text())
    assert receipt['status']=='SAVED_EPISODE_TECHNICAL_PASS_NOT_G0'
    assert receipt['scientific_lock']==identity(lock['_lock_path'])
    for field in ('source_head','source_tree','source_archive','model_revision','teacher_manifest','repair_numerics'):
        assert receipt[field]==lock[field],('TECHNICAL_REUSE_IDENTITY',field)
    assert receipt['checks']['status']=='MODEL_TECHNICAL_CHECKS_PASS_NOT_G0'
    assert receipt['native_target_calls']==receipt['native_solves']==receipt['commits']==0
    for obj in ('E','D'):
        assert receipt['checks'][obj]['direct']['status']=='PASS'
        assert receipt['checks'][obj]['fd']['status']=='PASS'
        assert all(x['status']=='PASS' for x in receipt['checks'][obj]['fd']['directions'].values())
    return receipt,identity(p)


def endpoint_checks(adapter,records,sweeps,lock,output):
    """E direct+full grid, then D direct+full grid; bounded, no fit/solve."""
    import torch
    from .repair_checks import (_save_json,make_directions,check_direct_route,run_objective_fd)
    from .technical import functional_materialized
    from .model_adapter import tensor_sha
    from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng,restore_rng
    root=Path(output);root.mkdir(parents=True,exist_ok=False,mode=0o700)
    config=config_from_lock(lock);record=recorder(root/'early')
    saved_rng=capture_rng();record('diagnostic_rng',saved_rng)
    raw_sha=tensor_sha(adapter.weight);A_sha=tensor_sha(adapter.fixed_a)
    # Capture baseline repeat jitter without calling a new target or gradient.
    baseline={};checks={}
    for obj,key in (('E','current'),('D','generic')):
        observations=[sweeps[key]]
        for repeat in range(2):
            restore_rng(saved_rng)
            observation=adapter.current(records) if obj=='E' else adapter.generic('S64')
            _save_json(root/f'{obj}-zero-repeat-{repeat+1}.json',observation)
            observations.append(observation)
        baseline[obj]=observations
    current_exact=all(x['rows']==sweeps['current']['rows'] for x in baseline['E'])
    generic_exact=all(x['rows']==sweeps['generic']['rows'] for x in baseline['D'])
    _save_json(root/'C0-parity.json',dict(current_exact=current_exact,generic_exact=generic_exact,
        E_values=[x['E'] for x in baseline['E']],D_values=[x['D'] for x in baseline['D']]))
    assert current_exact and generic_exact,'C0_ACTUAL_NATIVE_METRICS_PARITY'
    zero=torch.zeros_like(sweeps['gE'])
    parity={}
    for label,c in [('zero',zero),('nonzero',sweeps['gE']*(1e-4*float(adapter.raw.double().norm())/
            max(float((sweeps['gE']@adapter.fixed_a).double().norm()),1e-30)))]:
        restore_rng(saved_rng)
        result=functional_materialized(adapter,c,numerics=lock['technical_numerics'],
            recorder=lambda value:_save_json(root/f'{label}-materialized.json',value))
        parity[label]=result
    for obj in ('E','D'):
        restore_rng(saved_rng)
        direct=adapter.direct_weight_gradients(records,objectives=(obj,),recorder=record)
        gradient=sweeps['g'+obj];gW=direct['gW'+obj]
        dirs=make_directions(gradient,objective=obj,config=config)
        comparison=check_direct_route(gradient,gW,adapter.fixed_a,dirs,config=config)
        _save_json(root/f'{obj}-direct-route.json',dict(comparison,collection=direct['receipt']))
        assert comparison['status']=='PASS',obj+'_DIRECT_ROUTE_UNRESOLVED'
        assert direct['current' if obj=='E' else 'generic']['rows']==sweeps['current' if obj=='E' else 'generic']['rows'],'DIRECT_ROUTE_FORWARD_METRICS'
        del direct,gW,dirs
        kwargs={'E_receipt':checks['E']['fd']} if obj=='D' else {}
        fd=run_objective_fd(adapter,records,gradient,objective=obj,baseline_observations=baseline[obj],
            output=root/(obj+'-FD'),config=config,**kwargs)
        checks[obj]=dict(direct=comparison,fd=fd)
        _save_json(root/f'{obj}-stage.json',checks[obj])
        assert fd['status']=='PASS',obj+'_FD_UNRESOLVED'
    restore_rng(saved_rng)
    assert tensor_sha(adapter.weight)==raw_sha and tensor_sha(adapter.fixed_a)==A_sha
    receipt=dict(status='MODEL_TECHNICAL_CHECKS_PASS_NOT_G0',E=checks['E'],D=checks['D'],
        C0_current_metrics_exact=current_exact,C0_generic_metrics_exact=generic_exact,
        functional_materialized=parity,raw_sha256=raw_sha,A_sha256=A_sha,
        gE_sha256=tensor_sha(sweeps['gE']),gD_sha256=tensor_sha(sweeps['gD']),
        native_target_calls=0,native_solves=0,history_append=0,endpoint_restore_exact=True,
        RNG_exact=capture_rng()==saved_rng,model_nonselected_hooks_guard='PER_OBSERVATION_POINTER_VERSION',
        direct_route_is_not_independent_proof_of_all_neural_backward=True)
    _save_json(root/'receipt.json',receipt)
    return receipt


def scientific_checks(adapter,records,sweeps,lock,output,prior):
    """Reuse exact state/gradient evidence, otherwise run only endpoint checks."""
    from .model_adapter import tensor_sha
    from .technical import functional_materialized
    import torch
    checks=prior['checks']
    actual=dict(raw_sha256=tensor_sha(adapter.raw),A_sha256=tensor_sha(adapter.fixed_a),
        gE_sha256=tensor_sha(sweeps['gE']),gD_sha256=tensor_sha(sweeps['gD']))
    equal=all(checks[k]==v for k,v in actual.items())
    if not equal:
        result=endpoint_checks(adapter,records,sweeps,lock,output)
        result.update(reuse='NOT_SAME_EPISODE_OR_GRADIENT;NEW_ENDPOINT_CHECKS',prior=identity(lock['repair_pass_path']))
        return result
    # Actual new process has its own byte/materialization/restore evidence.
    c0=adapter.current(records);d0=adapter.generic('S64')
    assert c0['rows']==sweeps['current']['rows'] and d0['rows']==sweeps['generic']['rows']
    parity=functional_materialized(adapter,torch.zeros_like(sweeps['gE']),numerics=lock['technical_numerics'])
    return dict(status='MODEL_TECHNICAL_CHECKS_PASS_NOT_G0',reuse='EXACT_VP_A_GE_GD_IDENTITY',
        actual=actual,prior=identity(lock['repair_pass_path']),C0_metrics_exact=True,
        live_materialization=parity,extra_native_fit=0,full_GPU_continuation_replay=False)


def run(lock_path,output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from .model_adapter import EpisodeAdapter,TeacherStore,tensor_sha
    from .repair_checks import _save_json
    from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng
    root=Path(output);root.mkdir(parents=True,exist_ok=False,mode=0o700)
    started=time.monotonic();stage='INPUT_VERIFY';model=None
    try:
        lock=json.loads(Path(lock_path).read_text());assert lock['mode']=='SAVED_NATIVE_EPISODE_TECHNICAL_ONLY'
        assert os.environ.get('SLURMD_NODENAME')=='server4'
        assert lock['native_target_calls']==lock['native_solves']==0
        verification=verify_members(lock);config_from_lock(lock)
        assert torch.__version__==lock['torch'] and transformers.__version__==lock['transformers']
        assert identity(lock['saved_episode']['path'])==lock['saved_episode']
        assert identity(lock['old_cpu_receipt']['path'])==lock['old_cpu_receipt']
        old_cpu=json.loads(Path(lock['old_cpu_receipt']['path']).read_text())
        _save_json(root/'preserved-old-coarse.json',dict(old_receipt=lock['old_cpu_receipt'],
            original_fd=old_cpu['fd_exact_saved_receipt'],old_failure_permanently_preserved=True,
            new_gradient_byte_identity='NOT_CLAIMED_OLD_TENSOR_NOT_SAVED',
            new_direction_and_h='NEW_MEASUREMENTS_IN_checks/E-FD/self_gradient'))
        records=load_prefix(lock['dataset_root'],1000)[:100]
        assert [r['case_id'] for r in records]==lock['batches'][0]['case_ids']
        episode=torch.load(lock['saved_episode']['path'],map_location='cpu',weights_only=True,mmap=True)
        assert [r['case_id'] for r in episode['target_observations']]==[r['case_id'] for r in records]
        assert tensor_sha(episode['native_proposal'])=='124e7a3d6a6ead73785703075dd88022ac1c0e7900dc99c7c1e872bee946810f'
        assert episode['native_proposal'].shape==(4096,14336) and episode['A'].shape==(100,14336)
        stage='MODEL_W0_LOAD';load_start=time.monotonic()
        torch.set_num_threads(8);transformers.set_seed(lock['diagnostic_seed'])
        torch.backends.cuda.matmul.allow_tf32=lock['tf32_matmul'];torch.backends.cudnn.allow_tf32=lock['tf32_cudnn']
        model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,torch_dtype=torch.float32,
            low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        assert model.config.vocab_size==128256
        for p in model.parameters():p.requires_grad_(False)
        tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);tok.pad_token_id=tok.eos_token_id
        assert tok.padding_side=='right'
        name='model.layers.4.mlp.down_proj.weight';weight=dict(model.named_parameters())[name]
        teacher=TeacherStore(lock['reference_root'],lock['teacher_manifest']['path'],
            expected_manifest_sha=lock['teacher_manifest']['sha256'],verify_payload_hashes=False)
        adapter=EpisodeAdapter(model,tok,name,teacher)
        w0=weight.detach().cpu().clone();load_seconds=time.monotonic()-load_start
        with torch.no_grad():weight.copy_(episode['native_proposal'].to(weight.device))
        adapter.set_episode(weight,episode['A'].to(weight.device))
        early=recorder(root/'early')
        early('entry',dict(diagnostic_rng=capture_rng(),diagnostic_seed=lock['diagnostic_seed'],
            raw_sha256=tensor_sha(weight),A_sha256=tensor_sha(adapter.fixed_a),
            original_saved_episode=lock['saved_episode'],original_postfit_rng='NOT_SAVED',
            prior_input_verification=verification,teacher=teacher.receipt,load_seconds=load_seconds,
            native_target_calls=0,native_solves=0,W0_nonselected='PINNED_BASE_MODEL_ASSET_REUSE',
            model_revision=lock['model_revision'],source_archive=lock['source_archive']))
        stage='RESIDUAL_E_THEN_D_GRADIENT_EARLY_SAVE'
        sweeps=adapter.gradient_sweeps(records,recorder=early)
        # Observer-only true-target loss and desired margin at the SAME Vp.
        true_records=copy.deepcopy(records)
        for row in true_records:row['requested_rewrite']['target_new']=row['requested_rewrite']['target_true']
        truth=adapter.current(true_records)
        _save_json(root/'current-true-and-margin.json',dict(observation=truth,
            margins=[t['nll']-n['nll'] for t,n in zip(truth['rows'],sweeps['current']['rows'])],
            extra_current_forward_groups=7,controller_access=False))
        stage='DIRECT_AND_CONVERGED_WINDOW_CHECKS'
        checks=endpoint_checks(adapter,records,sweeps,lock,root/'checks')
        stage='RESTORE_W0_GUARD'
        with torch.no_grad():weight.copy_(w0.to(weight.device))
        assert tensor_sha(weight)==tensor_sha(w0)
        receipt=dict(status='SAVED_EPISODE_TECHNICAL_PASS_NOT_G0',checks=checks,
            lock=identity(lock_path),scientific_lock=lock['scientific_lock'],
            **{k:lock[k] for k in ('source_head','source_tree','source_archive','model_revision','teacher_manifest','repair_numerics')},
            saved_episode=lock['saved_episode'],W0_selected_restore_exact=True,
            native_target_calls=0,native_solves=0,commits=0,history_append=0,
            seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved=torch.cuda.max_memory_reserved(),host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cost='TECHNICAL_ONLY_NOT_SCIENTIFIC_COUNTS',scientific_gate='NOT_RUN')
        save(root/'technical-PASS.json',receipt)
        print('EP_TW1_SAVED_EPISODE_TECHNICAL_PASS_NOT_G0',flush=True)
        return receipt
    except BaseException as error:
        _save_json(root/'failure.json',dict(status='TERMINAL_TECHNICAL_FAIL_OR_UNRESOLVED',stage=stage,
            error=str(error),exception_type=type(error).__name__,traceback=traceback.format_exc(),
            seconds=time.monotonic()-started,native_target_calls=0,native_solves=0,scientific_started=False,
            next_agent_state='WAITING_USER_RESUME',automatic_repair=False,
            peak_cuda_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run(args.lock,args.output)

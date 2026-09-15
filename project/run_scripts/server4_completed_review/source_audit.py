"""Frozen source, input and saved native-log audit, CPU/read-only."""
import ast
import hashlib
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from .begin import ROOT, WT, LOCAL, TASK
from .cake import ATTEMPT, RAW
from project.run_scripts.bg_tw_reference.ep_tw import review_nogate as r

SWEEP=ROOT/'local/ep-tw1-alpha-cap-sweep/20260915-v1'

def source_ref(p,needle):
    p=Path(p);text=p.read_text();line=next(i for i,s in enumerate(text.splitlines(),1) if needle in s)
    return dict(r.ref(p),line=line)

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cl=r.read(ATTEMPT/'execution.lock.json');src=Path(cl['cake_root']);up=ATTEMPT/'upstream/CAKE'
    old=(up/'Cake/Cake_main.py').read_text();new=(src/'Cake/Cake_main.py').read_text()
    assert old.replace('from notebooks.util import hparams\n','').rstrip()==new.rstrip()
    assert ast.dump(ast.parse(old.replace('from notebooks.util import hparams\n','')))==ast.dump(ast.parse(new))
    assert r.sha(src/'Cake/compute_z.py')==r.sha(up/'Cake/compute_z.py')
    assert r.sha(src/'Cake/compute_ks.py')==r.sha(up/'Cake/compute_ks.py')
    imports=r.read(RAW/'runtime.json')['actual_imports'];files=[]
    for name,x in imports.items():
        p=Path(x['path']);actual=r.ref(p);assert actual['sha256']==x['sha256']
        files.append(dict(actual,role='CAKE_ACTUAL_IMPORTED_SOURCE',module=name))
    caps=[];source_seen=set();archive_seen=set();oldsource=None
    for arm in ('CAP10','CAP100','NORM_ONLY'):
        a=SWEEP/arm/'attempt-v1';lock=r.read(a/'execution.lock.json');entry=r.read(a/'scientific-v1/entry.json')
        assert entry['M_zero'] and entry['seed']==lock['seed']==20260915
        assert entry['model_revision']==lock['model_revision'] and entry['P_mapping']==lock['projector_mapping']
        assert lock['numerical_validation']=='NOT_ESTABLISHED'
        assert entry['source_lock']['sha256']==r.sha(a/'execution.lock.json')
        for m in lock['members']:
            p=Path(m['path'])
            if p.is_relative_to(lock['source_root']) and str(p) not in source_seen:
                actual=r.ref(p);assert actual['sha256']==m['sha256'] and actual['bytes']==m['bytes']
                files.append(dict(actual,role='FROZEN_EXECUTION_SOURCE'));source_seen.add(str(p))
        ar=Path(lock['source_archive']['path'])
        if str(ar) not in archive_seen:
            actual=r.ref(ar);assert actual['sha256']==lock['source_archive']['sha256'];files.append(dict(actual,role='EXECUTION_ARCHIVE'));archive_seen.add(str(ar))
        # These small immutable locks/manifests, not all model/teacher tensors.
        for name in ('sample_lock','teacher_manifest','config4','contexts'):
            item=lock[name]
            if isinstance(item,dict) and 'path' in item:
                actual=r.ref(item['path']);assert actual['sha256']==item['sha256'];files.append(dict(actual,role='SMALL_INPUT_'+name))
        caps.append(dict(arm=arm,source=lock['source_head'],lock=r.ref(a/'execution.lock.json'),
            numerical_policy=lock['numerical_policy'],seed=lock['seed'],base_W4=entry['base_W4_sha256'],state=entry['state'],
            old_numerical_rationale_alpha=lock['numerical_rationale'].get('alpha_cap'),
            rationale_status='LEGACY_CAP1_DESCRIPTIVE_METADATA; actual numerical_policy and executed correction receipt govern',
            monitoring_metadata='HISTORICAL_EXECUTION_LOCK_RUN_THROUGH; latest user pause and this review-only recall govern agent'))
    assert caps[0]['state']==caps[1]['state']==caps[2]['state']
    ns=SWEEP/'source-v1/project/run_scripts/bg_tw_reference/ep_tw'
    oldlock=r.read(ROOT/'local/ep-tw1-c4/20260915-v1/gate-skip-r1/execution.lock.json')
    osrc=Path(oldlock['source_root'])/'project/run_scripts/bg_tw_reference/ep_tw'
    for name in ('model_adapter.py','ledger.py'):
        assert r.sha(ns/name)==r.sha(osrc/name)
    diff=subprocess.check_output(['git','diff','6d317bdb2660d7e9919bc3a9fb878564e9729e37','8c64366c2314f188e034e5f4403fe89f0eaad873','--',
        'project/run_scripts/bg_tw_reference/ep_tw/policy.py','project/run_scripts/bg_tw_reference/ep_tw/runner.py','project/run_scripts/bg_tw_reference/ep_tw/gate_skip.py'],cwd=WT,text=True)
    with (out/'CAP1-to-sweep-execution.patch').open('x') as f:f.write(diff)
    mappings=[]
    def add(family,rule,file,needle,artifact,level):
        x=source_ref(file,needle)
        mappings.append(dict(family=family,requirement=rule,source=x['path'],line=x['line'],source_sha=x['sha256'],artifact=artifact,verification=level))
    cmain=src/'Cake/Cake_main.py';crt=Path(cl['source_root'])/'cake_native_lifelong/runtime.py'
    frozen_runtime=subprocess.check_output(['git','show','7884aeb6000f8343139172825ec6c4ca24357fc0:project/run_scripts/cake_native_lifelong/runtime.py'],cwd=WT)
    assert r.sha(crt)==hashlib.sha256(frozen_runtime).hexdigest()
    for rule,file,needle,artifact,level in [
        ('original apply once B100',crt,'module.apply_Cake_to_model','100 native-observation/commit','SOURCE_AND_STORED_TELEMETRY'),
        ('last L8 target once per request',cmain,'cur_z = compute_z(','z[100] per batch, layer8 and case order','SOURCE_AND_STORED_TELEMETRY'),
        ('current-state residual',cmain,'targets = zs - cur_zs','keys[4..8], solves5; residual tensor NOT_SAVED','SOURCE_CONFIRMED_PARTIAL_TELEMETRY'),
        ('remaining causal-weight allocation',cmain,'effective_ratio = current_weight / remaining_weight_sum','layer-weights CPU source formula; log six decimals','SOURCE_AND_LOG_AGGREGATE'),
        ('native projected direct solve',cmain,'upd_matrix = torch.linalg.solve','500 solve shape/dtype records','SOURCE_AND_STORED_TELEMETRY'),
        ('history only original apply final pass',cmain,'cache_c[i,:,:] +=','100 final passes /500 layer histories','SOURCE_AND_STORED_TELEMETRY'),
        ('observer/current+allseen; no controller feedback',crt,'current=evaluate','current and fullseen state hashes','SOURCE_AND_REDUCED_ROWS'),
        ('no saved W/M',crt,"saved_weight_tensor_count=0",'0 pt files, metadata links99','USER_DIRECTED_NO_CP_NOT_REPLAY')]:add('CAKE',rule,file,needle,artifact,level)
    for rule,file,needle,artifact,level in [
        ('actual RAW Vp; own native fit1',ns/'runner.py','fit=capture_native_fit','30 native-targets-map; commit native100/solve1','SOURCE_AND_CPU_TENSOR_BINDING'),
        ('C0 copies Vp; gC=gW A^T',ns/'model_adapter.py','return weight_gradient @ fixed_a.T','gE/gD/RAW header bridge; derivative FD skipped','SOURCE_CONFIRMED_NUMERICAL_NOT_ESTABLISHED'),
        ('E desired token mean then request mean',ns/'model_adapter.py','scalar = losses.sum() / len(records)','current rows denominator100;7 microbatches','SOURCE_AND_STORED_ROWS'),
        ('D W0 full-vocab KL S64',ns/'model_adapter.py','by_position = (logp0.exp()','64doc128positions; S64 teacher ID','SOURCE_AND_STORED_ROWS'),
        ('E/D separate sweep, only S64 gradient',ns/'model_adapter.py',"if backward and role != 'S64'",'70current+640S64 backward/arm','SOURCE_AND_COUNTERS'),
        ('halfspace d and zero branch',ns/'policy.py','coefficient = min(q, 0.0)','per-batch-actions CPU q/dot vs stored','SOURCE_AND_STORED_TENSORS'),
        ('bounded/disabled cap only',ns/'policy.py','alpha = min(config.alpha_cap','alpha norm/used/mode/cap receipt','SOURCE_AND_CPU_SCALARS'),
        ('ball then C-only trust',ns/'policy.py','correction = projected - Zp','ball hit/trust/inner products','SOURCE_AND_CPU_SCALARS'),
        ('RAW/C1/C05/C025 actual materialization',ns/'policy.py','value = Vp.detach().clone()','160 candidate receipts incl reused10batch','SOURCE_AND_CPU_TENSORS'),
        ('exact E and strict ID set; minD tieRAW',ns/'policy.py','e_ok = finite and obs.mean_nll','candidate-details arithmetic; no tolerance relaxation','SOURCE_AND_STORED_ROWS'),
        ('final history1 and accepted-only ledger',ns/'runner.py','finalization=fitter.finalize','30CP/27links/ledger summaries','SOURCE_AND_CPU_TENSORS'),
        ('FD/ULP/selfKL/direct diagnostic skip',ns/'runner.py',"tech=diagnostic(lock,'VALIDATE_EPISODE",'technical JSON SKIPPED_USER_DIRECTED','SKIPPED_USER_DIRECTED')]:add('EP_SWEEP',rule,file,needle,artifact,level)
    r.table(out/'design-conformance.csv',mappings);r.table(out/'source-inventory.csv',files)
    # CAKE log-derived actual update counters. No prompt lines are published.
    log=ATTEMPT/'logs/48101.out';counts=[];n=None
    for line in log.open(errors='replace'):
        if line.strip()=='Computing right vector (v)':assert n is None;n=0
        elif n is not None and line.startswith('loss '):n+=1
        elif n is not None and line.startswith('Init norm ') and ' | Target norm ' in line:
            assert 1<=n<=25;counts.append(n);n=None
    assert n is None and len(counts)==10000
    batchcounts=[dict(batch=i+1,requests=100,loss_evaluations=sum(counts[i*100:(i+1)*100]),
        Adam_updates=sum(x-1 for x in counts[i*100:(i+1)*100]),early_stop=sum(x<25 for x in counts[i*100:(i+1)*100]),
        zero_updates=sum(x==1 for x in counts[i*100:(i+1)*100]),origin='LOG_RECONSTRUCTED_FROM_NATIVE_PRINT_BOUNDARIES_NOT_NATIVE_COUNTER') for i in range(100)]
    r.table(out/'CAKE-log-target-counters.csv',batchcounts)
    hp=cl['hparams'];scores=np.array([hp['causal_scores'][str(i)] for i in range(5)])
    vals=np.exp((scores-scores.max())/hp['temperature']);weights=(vals/vals.sum()).astype(np.float32)
    layers=[dict(physical_layer=4+i,score_index=i,score=float(scores[i]),weight_FP32=float(w),
        remaining_ratio_FP32=float(w/weights[i:].sum(dtype=np.float32)),source='CPU_FORMULA_DERIVED_NOT_GPU_REPLAY') for i,w in enumerate(weights)]
    r.table(out/'CAKE-layer-weights.csv',layers)
    return r.save(out/'source-audit.json',dict(task=TASK,caps=caps,model_adapter_ledger_bytes_unchanged=True,
        cake_native_patch='UNUSED_NOTEBOOKS_IMPORT_REMOVAL_AND_EOF_LF_ONLY',cake_same_native_AST=True,
        CAKE_actual_log_losses=sum(counts),CAKE_log_Adam=sum(x-1 for x in counts),CAKE_log_early_stop=sum(x<25 for x in counts),
        CAKE_log_zero_updates=sum(x==1 for x in counts),log_ref=r.ref(log),
        full_model_teacher_P_rehash='REUSED_PRIOR_IMMUTABLE_IDENTITY_NOT_NEW_FULL_REHASH',
        source_files=len(files),new_model_forwards=0,numerical_validation='NOT_ESTABLISHED'))

if __name__=='__main__':print(json.dumps(run(LOCAL/'analysis/source-audit-v2')))

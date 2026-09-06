"""Budget-locked post-common-primary refinement/audit; no D10A replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import torch
from .runtime import ObservedFamily,raw_requests
from .provenance import save,ARM_ORDER,git
from .algebra import FrozenNormalization
from .native_binding import NativeDictionary
from .trajectory import run_joint
from project.run_scripts.ordered_response_barrier_ode import runtime as old
from project.run_scripts.ordered_response_barrier_ode import preflight as assets
from project.run_scripts.ordered_response_barrier_ode.contracts import canonical_arm_configs,assert_full_fp32
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256


def verify_warm_member(path, expected_sha):
    if path.is_symlink() or not path.is_file():raise RuntimeError('WARM_MEMBER_TYPE')
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected_sha:raise RuntimeError('WARM_MEMBER_BYTES_DRIFT')


def run(repo,primary_root,output_root,cell_id,budget_seconds):
    def terminate(signum,frame):raise RuntimeError(f'EXTERNAL_RESOURCE_TERMINATION_SIGNAL_{signum}')
    signal.signal(signal.SIGTERM,terminate)
    stage='PREMODEL';f=None;started=time.perf_counter()
    root=output_root/f'cell-{cell_id}';root.mkdir(parents=True,mode=0o700)
    try:
        for i in range(4):
            terminal=json.loads((primary_root/f'cell-{i}'/'primary-terminal.json').read_text())
            if terminal['status']!='TERMINAL_VALID' or terminal['primary_arm_count']!=4 or terminal['primary_request_endpoints']!=40:
                raise RuntimeError('FOUR_CELL_PRIMARY_GATE_BOUNDARY')
            gate=json.loads((primary_root/f'cell-{i}'/'gpu_fidelity_checks.json').read_text())
            if gate['status']!='PASS':raise RuntimeError('PRIOR_FIDELITY_BOUNDARY')
        lock=json.loads((output_root/'followup.lock.json').read_text())
        if lock['source_head']!=git(repo,'rev-parse','HEAD') or lock['budget_seconds_per_cell']!=budget_seconds or git(repo,'status','--porcelain','--untracked-files=no'):
            raise RuntimeError('FOLLOWUP_SOURCE_RESOURCE_DRIFT')
        warm_path=primary_root/f'cell-{cell_id}'/'warm-state.pt'
        verify_warm_member(warm_path,lock['warm_members'][str(cell_id)]['sha256'])
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        random.seed(20260906);torch.manual_seed(20260906);torch.cuda.manual_seed_all(20260906)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        if torch.cuda.device_count()!=1:raise RuntimeError('SINGLE_GPU_BOUNDARY')
        old._bootstrap_easyedit(assets.OFFICIAL_EASYEDIT_ROOT)
        cell=assets.cell_spec(cell_id);snapshot=old._model_snapshot(assets.HF_HUB_CACHE_ROOT,cell.model_alias)
        from transformers import AutoModelForCausalLM,AutoTokenizer
        stage='MODEL_RELOAD';load_start=time.perf_counter()
        model=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float32,low_cpu_mem_usage=True,device_map={'':'cuda:0'},attn_implementation='eager')
        tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True)
        if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
        tok.padding_side='right';model.config.pad_token_id=tok.pad_token_id;model.config.use_cache=False;model.eval()
        assert_full_fp32(model);old.seal_eager_attention(model);old._install_model_forward_counter(model)
        hp,_=old._load_hparams(repo,cell.writer_family,cell.model_alias);hp.device=0
        hp.stats_dir=str(assets.EASYEDIT_ARTIFACT_ROOT/'examples/data/stats')
        if cell.writer_family=='AlphaEdit':hp.P_loc=str(assets.EASYEDIT_ARTIFACT_ROOT/assets.MODEL_BINDINGS[cell.model_alias]['projector'][0])
        module=old._method_module(cell.writer_family)
        # This is our own sealed tensor artifact, never an untrusted pickle input.
        warm=torch.load(warm_path,map_location='cpu',weights_only=False)
        contexts=warm['contexts'];module.CONTEXT_TEMPLATES_CACHE=contexts
        sample=json.loads((primary_root/'sample.lock.json').read_text())
        if warm['sample_root']!=sample['ordered_root']:raise RuntimeError('WARM_SAMPLE_BOUNDARY')
        raw=old._raw_records(assets.EASYEDIT_ARTIFACT_ROOT/assets.DATASET_RELATIVE,{int(r['case_id']) for r in sample['records']})
        def make(fixture,cold=False):
            selected=[r for r in sample['records'] if r['fixture']==fixture]
            rows=[raw[int(r['case_id'])] for r in selected]
            for meta,row in zip(selected,rows,strict=True):
                if assets.canonical_hash(row)!=meta['raw_record_sha256']:raise RuntimeError('SAMPLE_BYTES_DRIFT')
            value=ObservedFamily(model=model,tokenizer=tok,family=cell.writer_family,hparams=hp,module=module,
                requests=old._official_requests(raw_requests(rows)),endpoint_records=rows,
                request_order_sha256=assets.canonical_hash([m['request_sha256'] for m in selected]),contexts=contexts)
            if cold:value.prepare_method_state()
            else:value.bind_existing_method_state()
            return value
        stage='D2_COLD_REFINEMENT';f=make('D2',True)
        cold_weights=f.w0;cold_sha=f.w0_sha256
        cold_cache=module.cache_c.detach().clone() if cell.writer_family=='AlphaEdit' else None
        cold_cache_flag=getattr(module,'cache_c_new',None)
        f.compute_fixed_z()
        with torch.no_grad():
            entry=f.terminal();norm=FrozenNormalization.capture(f.fixed_z.values,entry,f.w0_sha256)
            dictionary=NativeDictionary(f);dictionary.capture_reference(dictionary.build(entry,0))
        save(root/'runtime.lock.json',dict(source_head=lock['source_head'],phase='POST_PRIMARY',
            model_reload_count=1,model_reload_seconds=time.perf_counter()-load_start,
            D2_entry='COLD_DEV_FIXTURE',D2_z_compute_count=1,D10A_replay_count=0,
            warm_entry_sha=warm['commit']['committed_weight_sha256'],sample_root=sample['ordered_root'],
            budget_seconds=budget_seconds,scientific_promotion=False))
        for n in (2,4,8):
            stage=f'D2_N{n}'
            result=run_joint(f,'JV_NATIVE',dictionary,norm,n=n,output=root/'D2'/f'N{n}',fixture='D2')
            save(root/f'D2-N{n}.json',result)
        # H10 independent from D10B, using saved exact common warm snapshot.
        elapsed=time.perf_counter()-started
        old_primary=[json.loads((primary_root/f'cell-{cell_id}'/f'D10B-{arm}.json').read_text()) for arm in ARM_ORDER]
        primary_entry=json.loads((primary_root/f'cell-{cell_id}'/'D10B-entry.json').read_text())
        estimate=1.5*(sum(r['total_seconds'] for r in old_primary)+primary_entry['target_seconds'])
        remaining=budget_seconds-elapsed-120
        if remaining<=estimate:
            save(root/'audit-status.json',dict(status='NOT_RUN_BUDGET',estimate_seconds=estimate,remaining_seconds=remaining,
                rule='1.5*(primary4arm wall + target wall), budget-only, cleanup reserve120s'))
        else:
            stage='H10_WARM_RESTORE'
            f._apply_shadow(warm['weights'])
            if tensor_set_sha256(f.parameters)!=warm['commit']['committed_weight_sha256']:
                raise RuntimeError('WARM_WEIGHT_SHA_BOUNDARY')
            if cell.writer_family=='AlphaEdit':module.cache_c.copy_(warm['alpha_cache']);module.cache_c_new=True
            f=make('H10');f.old_records=[raw[int(r['case_id'])] for r in sample['records'] if r['fixture']=='D10A']
            if f.method_state_identity()!=warm['commit']['committed_method_state_sha256']:
                raise RuntimeError('WARM_METHOD_STATE_BOUNDARY')
            f.compute_fixed_z()
            with torch.no_grad():
                entry=f.terminal();norm=FrozenNormalization.capture(f.fixed_z.values,entry,f.w0_sha256)
                dictionary=NativeDictionary(f);dictionary.capture_reference(dictionary.build(entry,0))
                f.metric_observer=dictionary;pre=f.evaluate_endpoint();old_before=f.last_old_evaluation
            save(root/'H10-entry.json',dict(evaluation=pre,old_before=old_before,entry_sha=f.w0_sha256,fixed_z_sha=f.fixed_z.identity_sha256))
            configs={c.arm.value:c for c in canonical_arm_configs()}
            for arm in ARM_ORDER:
                stage=f'H10_{arm}'
                if arm=='O_NATIVE':result=f.run_official(fixed_z=f.fixed_z)
                elif arm=='ORBFH_HIST':result=old._arm_run(f,configs['ORBFH'])
                else:result=run_joint(f,arm,dictionary,norm,output=root/'H10'/arm,fixture='H10')
                save(root/f'H10-{arm}.json',dict(result=result,old_before=old_before,old_after=f.last_old_evaluation,
                    actual_physical_action=f.last_physical_action))
        f.reset_entry();old._restore_selected(f.parameters,cold_weights)
        if cell.writer_family=='AlphaEdit':module.cache_c.copy_(cold_cache);module.cache_c_new=cold_cache_flag
        save(root/'terminal.json',dict(status='TERMINAL_VALID',D2_refinements=3,cold_restore=tensor_set_sha256(f.parameters)==cold_sha,
            allocated_seconds=time.perf_counter()-started,scientific_promotion=False))
    except BaseException as exc:
        restore=None
        if f is not None:
            try:f.reset_entry();restore=tensor_set_sha256(f.parameters)==f.w0_sha256
            except BaseException as err:restore=str(err)
        save(root/'failure-boundary.json',dict(stage=stage,exception=str(exc),type=type(exc).__name__,traceback=traceback.format_exc(),
            restore=restore,allocated_seconds=time.perf_counter()-started,science_change_count=0,scientific_promotion=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['repo','primary-root','output-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cell',type=int,required=True);p.add_argument('--budget-seconds',type=int,required=True)
    a=p.parse_args();run(a.repo,a.primary_root,a.output_root,a.cell,a.budget_seconds)

"""One isolated model/family process, cold fidelity then common warm comparisons."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
import traceback
import torch
from project.run_scripts.ordered_response_barrier_ode import runtime as old
from project.run_scripts.ordered_response_barrier_ode import preflight as assets
from project.run_scripts.ordered_response_barrier_ode.contracts import canonical_arm_configs, assert_full_fp32
from project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator import evaluate_counterfact_with_canonical_ns
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256
from .algebra import FrozenNormalization
from .native_binding import NativeDictionary
from .fidelity import run_fidelity
from .trajectory import run_joint
from .provenance import ARM_ORDER, SCIENCE, save, git


class ObservedFamily(old.FamilyRuntime):
    """Old prompts are exposed only within the endpoint evaluator callback."""
    old_records=()
    last_old_evaluation=None
    last_terminal=None
    evaluation_seconds=0.

    def evaluate_endpoint(self):
        old._sync(); started=time.perf_counter()
        result=super().evaluate_endpoint()
        self.last_terminal=self.terminal()
        if self.old_records:
            self.last_old_evaluation=evaluate_counterfact_with_canonical_ns(self.model,self.tokenizer,
                self.old_records,device=self.device,microbatch_size=16)
        old._sync();self.evaluation_seconds+=time.perf_counter()-started
        return result


def raw_requests(rows):
    return [dict(case_id=int(r['case_id']),prompt=r['requested_rewrite']['prompt'],
                 subject=r['requested_rewrite']['subject'],target_new=r['requested_rewrite']['target_new']['str']) for r in rows]


def run_cell(repo, root, cell_id):
    cell=assets.cell_spec(cell_id)
    output=root/f'cell-{cell_id}'
    if output.exists():
        raise RuntimeError('CREATE_ONCE_CELL_EXISTS')
    output.mkdir(mode=0o700)
    started=time.perf_counter();f=None;stage='PREMODEL';completed=[]
    cold=None;cold_state=None;module=None;model=None
    try:
        source=json.loads((root/'source.lock.json').read_text())
        if source['head']!=git(repo,'rev-parse','HEAD') or source['tree']!=git(repo,'rev-parse','HEAD^{tree}'):
            raise RuntimeError('QUEUED_SOURCE_DRIFT')
        if git(repo,'status','--porcelain','--untracked-files=no'):
            raise RuntimeError('EXECUTION_TRACKED_DIRTY')
        if json.loads((root/'science.lock.json').read_text()) != SCIENCE:
            raise RuntimeError('SCIENCE_LOCK_DRIFT')
        sample=json.loads((root/'sample.lock.json').read_text())
        if assets.canonical_hash(sample['records'])!=sample['ordered_root']:
            raise RuntimeError('SAMPLE_LOCK_DRIFT')
        raw=old._raw_records(assets.EASYEDIT_ARTIFACT_ROOT/assets.DATASET_RELATIVE,
                             {int(r['case_id']) for r in sample['records']})
        for r in sample['records']:
            if assets.canonical_hash(raw[int(r['case_id'])])!=r['raw_record_sha256']:
                raise RuntimeError('SAMPLE_BYTES_DRIFT')
        official=old._bootstrap_easyedit(assets.OFFICIAL_EASYEDIT_ROOT)
        if official['head']!=assets.OFFICIAL_EASYEDIT_HEAD or not official['tracked_clean']:
            raise RuntimeError('EASYEDIT_SOURCE_DRIFT')
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
        random.seed(20260906);torch.manual_seed(20260906);torch.cuda.manual_seed_all(20260906)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')
        if torch.cuda.device_count()!=1:
            raise RuntimeError('SINGLE_GPU_BOUNDARY')
        torch.cuda.set_device(0)
        from transformers import AutoModelForCausalLM,AutoTokenizer
        snapshot=old._model_snapshot(assets.HF_HUB_CACHE_ROOT,cell.model_alias)
        stage='MODEL_LOAD';load_start=time.perf_counter()
        model=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float32,low_cpu_mem_usage=True,device_map={'':'cuda:0'},attn_implementation='eager')
        tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True)
        if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
        tok.padding_side='right';model.config.pad_token_id=tok.pad_token_id;model.config.use_cache=False;model.eval()
        old.seal_eager_attention(model);old._install_model_forward_counter(model)
        dtype=assert_full_fp32(model);old._sync();load_seconds=time.perf_counter()-load_start
        hp,hp_path=old._load_hparams(repo,cell.writer_family,cell.model_alias)
        hp.device=0;hp.stats_dir=str(assets.EASYEDIT_ARTIFACT_ROOT/'examples/data/stats')
        if cell.writer_family=='AlphaEdit':hp.P_loc=str(assets.EASYEDIT_ARTIFACT_ROOT/assets.MODEL_BINDINGS[cell.model_alias]['projector'][0])
        module=old._method_module(cell.writer_family);module.CONTEXT_TEMPLATES_CACHE=None
        with old._model_name(model,str(hp.model_name)):
            contexts=module.get_context_templates(model,tok)
        save(output/'runtime.lock.json',dict(cell=asdict(cell),source=source,official=official,
            hf_snapshot=str(snapshot),hparams=str(hp_path),dtype=dtype,parameter_inventory=old._parameter_inventory(model),
            target_context_identity=assets.canonical_hash(contexts),sample_root=sample['ordered_root'],
            cuda_device=torch.cuda.get_device_name(0),cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
            slurm_job=os.environ.get('SLURM_JOB_ID'),load_seconds=load_seconds,
            timing='CUDA synchronize before/after; synchronization wait included, leading sync excluded',
            model_load_count=1,autocast=False,tf32=False,controller_dtype='float64',
            memit_ephemeral_solve_dtype='float64',dynamic_model_forward_dtype='float32'))

        def make(fixture,cold_prepare=False):
            rows=[raw[int(r['case_id'])] for r in sample['records'] if r['fixture']==fixture]
            value=ObservedFamily(model=model,tokenizer=tok,family=cell.writer_family,hparams=hp,module=module,
                requests=old._official_requests(raw_requests(rows)),endpoint_records=rows,
                request_order_sha256=assets.canonical_hash([assets.canonical_hash(r['requested_rewrite']) for r in rows]),
                contexts=contexts,capture_persistent_endpoint=(fixture=='D10A'))
            if cold_prepare:value.prepare_method_state()
            else:value.bind_existing_method_state()
            return value

        stage='G0A';f=make('G0A',True)
        cold=f.w0;cold_sha=f.w0_sha256
        cold_state=module.cache_c.detach().clone() if cell.writer_family=='AlphaEdit' else None
        cold_flag=getattr(module,'cache_c_new',None)
        gates=[]
        for fixture in ('G0A','G0B'):
            stage=fixture
            if fixture!='G0A':f=make(fixture)
            f.compute_fixed_z()
            gate=run_fidelity(f,output/fixture);gates.append(gate)
            if gate['status']!='PASS':
                raise RuntimeError('NONZERO_FIDELITY_GATE_NOT_OBSERVED')
            save(output/f'first-valid-{fixture}.json',dict(status='PASS',cell_id=cell_id,fixture=fixture,
                fixed_z_compute_count=f.fixed_z.request_count,fixed_z_recompute_count=0,w0_restore=True))
        save(output/'gpu_fidelity_checks.json',dict(status='PASS',fixtures=gates))

        stage='D10A_WARM_CREATE';f=make('D10A');f.compute_fixed_z()
        native=f.run_official(fixed_z=f.fixed_z)
        save(output/'D10A-native-warm-creation.json',native)
        old_records=f.endpoint_records
        commit=f.commit_captured_endpoint(expected_sha256=native['selected_weight_endpoint_sha256'])
        save(output/'warm-entry.json',dict(**commit,common_D10A_entry=cold_sha,
            short_history_batches=1,fixed_target_shared_within_fixture=True))
        # D10B and H10 bind this same unchanged warm entry; no comparator carry.
        stage='D10B';f=make('D10B');f.old_records=old_records
        target_start=time.perf_counter();f.compute_fixed_z();old._sync();target_seconds=time.perf_counter()-target_start
        with torch.no_grad():
            entry=f.terminal();norm=FrozenNormalization.capture(f.fixed_z.values,entry,f.w0_sha256)
            dictionary=NativeDictionary(f);builds=dictionary.build(entry,0);metric=dictionary.capture_reference(builds)
            pre=f.evaluate_endpoint();old_before=f.last_old_evaluation
        save(output/'D10B-entry.json',dict(evaluation=pre,old_evaluation=old_before,metric=metric,
            target_seconds=target_seconds,fixed_z_sha=f.fixed_z.identity_sha256,
            entry_weight_sha=f.w0_sha256,method_state_sha=f.method_state_identity(),
            normalization_constants=norm.scales.tolist(),active=norm.active.tolist()))
        v0=float(norm.weight(f.fixed_z.values-entry).square().sum()/2)
        configs={c.arm.value:c for c in canonical_arm_configs()}
        primary=[]
        for arm in ARM_ORDER:
            stage=f'D10B/{arm}';arm_start=time.perf_counter();forward_before=old._model_forward_count(model)
            eval_before=f.evaluation_seconds
            if arm=='O_NATIVE':
                result=f.run_official(fixed_z=f.fixed_z)
                result=dict(status='TERMINAL_VALID',arm=arm,fixture='D10B',endpoint=result,
                            main_jvp_count=0,diagnostic_jvp_count=0)
            elif arm=='ORBFH_HIST':
                result=old._arm_run(f,configs['ORBFH'])
                result=dict(status='TERMINAL_VALID',arm=arm,fixture='D10B',historical=result,
                    main_jvp_count=result['jvp_ledger']['jvp_call_count'],diagnostic_jvp_count=0)
            else:
                result=run_joint(f,arm,dictionary,norm,output=output/'D10B'/arm,fixture='D10B')
            old._sync();elapsed=time.perf_counter()-arm_start
            vt=float(norm.weight(f.fixed_z.values-f.last_terminal).square().sum()/2)
            result.update(old_before=old_before,old_after=f.last_old_evaluation,
                V0=v0,VT=vt,V_ratio=vt/v0 if v0 else None,
                evaluation_seconds=f.evaluation_seconds-eval_before,
                total_seconds=elapsed,forward_count=old._model_forward_count(model)-forward_before,
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
                w0_restore=tensor_set_sha256(f.parameters)==f.w0_sha256,
                fixed_z_sha=f.fixed_z.identity_sha256,source_entry_sha=f.w0_sha256)
            save(output/f'D10B-{arm}.json',result);primary.append(result);completed.append(arm)
            save(output/f'progress-{len(completed)}.json',dict(completed_primary=completed,stage=stage,
                 allocated_seconds=time.perf_counter()-started))
        save(output/'primary-terminal.json',dict(status='TERMINAL_VALID',cell_id=cell_id,primary_arm_count=4,
            primary_request_endpoints=40,completed=completed,scientific_promotion=False,
            source_head=source['head'],sample_root=sample['ordered_root']))
        # Fixed resource-only estimate, never endpoint/efficacy-based selection.
        # All four cells use this identical budget rule and predeclared D2 fixture.
        primary_seconds=time.perf_counter()-started
        remaining=6900-primary_seconds
        estimate=max(60.,sum(r.get('total_seconds',0) for r in primary if r['arm']=='JV_NATIVE')*3.5)
        if remaining > estimate:
            stage='D2_REFINEMENT';f=make('D2');f.compute_fixed_z()
            with torch.no_grad():
                entry=f.terminal();norm=FrozenNormalization.capture(f.fixed_z.values,entry,f.w0_sha256)
                dic=NativeDictionary(f);dic.capture_reference(dic.build(entry,0))
            for n in (2,4,8):
                result=run_joint(f,'JV_NATIVE',dic,norm,n=n,output=output/'D2'/f'N{n}',fixture='D2')
                save(output/f'D2-N{n}.json',result)
        else:
            save(output/'refinement-status.json',dict(status='NOT_RUN_BUDGET',remaining_seconds=remaining,
                locked_estimate_seconds=estimate,estimate_rule='3.5*observed_primary_JV_wall; min60; budget only'))
        # H10 audit is never started unless the observed primary cost fits.
        estimate=sum(r.get('total_seconds',0) for r in primary)*1.5+target_seconds*1.5
        if 6900-(time.perf_counter()-started)>estimate:
            stage='H10_AUDIT';f=make('H10');f.old_records=old_records;f.compute_fixed_z()
            with torch.no_grad():
                entry=f.terminal();norm=FrozenNormalization.capture(f.fixed_z.values,entry,f.w0_sha256)
                dic=NativeDictionary(f);dic.capture_reference(dic.build(entry,0))
            for arm in ARM_ORDER:
                if arm=='O_NATIVE':result=f.run_official(fixed_z=f.fixed_z)
                elif arm=='ORBFH_HIST':result=old._arm_run(f,configs['ORBFH'])
                else:result=run_joint(f,arm,dic,norm,output=output/'H10'/arm,fixture='H10')
                save(output/f'H10-{arm}.json',dict(result=result,old_after=f.last_old_evaluation))
        else:
            save(output/'audit-status.json',dict(status='NOT_RUN_BUDGET',remaining_seconds=6900-(time.perf_counter()-started),
                estimate_seconds=estimate))
        f.reset_entry()
        old._restore_selected(f.parameters,cold)
        if cell.writer_family=='AlphaEdit':module.cache_c.copy_(cold_state);module.cache_c_new=cold_flag
        save(output/'terminal.json',dict(status='TERMINAL_VALID',cell=asdict(cell),primary_arm_count=4,
            primary_request_endpoints=40,source_head=source['head'],source_tree=source['tree'],
            cold_w0_restore=tensor_set_sha256(f.parameters)==cold_sha,
            gpu_allocated_seconds=time.perf_counter()-started,scientific_promotion=False,
            historical_states='HISTORICAL_STATE_UNAVAILABLE_NO_REPLAY',
            server4_rerun_mutation_count=0))
    except BaseException as exc:
        restore_error=None
        if f is not None:
            try:f.reset_entry()
            except BaseException as restore:restore_error=repr(restore)
        save(output/'failure-boundary.json',dict(stage=stage,type=type(exc).__name__,exception=str(exc),
            traceback=traceback.format_exc(),completed_primary=completed,
            restore_error=restore_error,w0_restore=(tensor_set_sha256(f.parameters)==f.w0_sha256) if f is not None else None,
            allocated_seconds=time.perf_counter()-started,science_change_count=0,tolerance_change_count=0,
            automatic_retry_count=0,scientific_promotion=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--cell',type=int,required=True)
    args=p.parse_args();run_cell(args.repo,args.run_root,args.cell)

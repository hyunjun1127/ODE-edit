"""Cold-start sequential harness; pinned kernels and captured-endpoint commits."""
import argparse,gc,json,os,random,resource,signal,sys,time,traceback
from pathlib import Path
import torch
from project.run_scripts.ordered_response_barrier_ode import runtime as old,preflight as assets
from project.run_scripts.ordered_response_barrier_ode.contracts import assert_full_fp32
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256,tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator import evaluate_counterfact_with_canonical_ns
from project.run_scripts.native_response_ode_v31.runtime import ObservedFamily,raw_requests
from project.run_scripts.native_response_ode_v31.native_binding import NativeDictionary
from project.run_scripts.native_response_ode_v31.algebra import FrozenNormalization
from project.run_scripts.native_response_ode_v31.provenance import save,git
from .contracts import SCIENCE,CHAINS,ALIASES,CHECKPOINTS,LAYERS
from .trajectory import run_joint
from .state import commit_checked,check_entry,checkpoint
from .evaluation import public_full,public_rewrite,rewrite_only,join_panels
from .accounting import Accounting


def validate_inputs(repo,root):
    source=json.loads((root/'source.lock.json').read_text())
    if source['head']!=git(repo,'rev-parse','HEAD') or source['tree']!=git(repo,'rev-parse','HEAD^{tree}'):
        raise RuntimeError('QUEUED_SOURCE_DRIFT')
    if git(repo,'status','--porcelain','--untracked-files=no'):raise RuntimeError('TRACKED_SOURCE_DIRTY')
    if json.loads((root/'science.lock.json').read_text())!=SCIENCE:raise RuntimeError('SCIENCE_LOCK_DRIFT')
    sample=json.loads((root/'sample.lock.json').read_text())
    if assets.canonical_hash(sample['records'])!=sample['ordered_root']:raise RuntimeError('SAMPLE_ROOT_DRIFT')
    data=old._raw_records(assets.EASYEDIT_ARTIFACT_ROOT/assets.DATASET_RELATIVE,
        {int(r['case_id']) for r in sample['records']}|set(sample['smoke_ids']))
    for r in sample['records']:
        if assets.canonical_hash(data[int(r['case_id'])])!=r['raw_record_sha256']:raise RuntimeError('SAMPLE_RECORD_DRIFT')
    return source,sample,data


def verify_prerequisites(root,mode,index):
    if (root/'server4-takeover.lock.json').is_file():
        from .server4_takeover import validate_handoff
        validate_handoff(root,mode,index)
        return
    if mode=='smoke':return
    binding=json.loads((root/'smoke-gates.lock.json').read_text())
    for alias in ALIASES:
        path=Path(binding['root'])/f'smoke-{alias}'/'terminal-receipt.json'
        if assets.sha256_file(path)!=binding['terminal_receipt_sha256'][alias]:raise RuntimeError('SMOKE_GATE_RECEIPT_DRIFT')
        gate=json.loads(path.read_text())
        if gate['status']!='TERMINAL_VALID' or gate['completed_batches']!=2 or gate['W0_restored'] is not True:
            raise RuntimeError('TWO_MODEL_TWO_BATCH_SMOKE_REQUIRED')
    if index>=4:
        # Secondary execution follows publication of the four-chain main table.
        gate=json.loads((root/'main-table-ready.json').read_text())
        if gate['completed_chains']!=[0,1,2,3]:raise RuntimeError('MAIN_TABLE_PRIORITY')


def run(repo,root,mode,index):
    verify_prerequisites(root,mode,index)
    source,sample,data=validate_inputs(repo,root)
    alias,arm=(ALIASES[index],'JV_NATIVE') if mode=='smoke' else CHAINS[index]
    output=root/(f'smoke-{alias}' if mode=='smoke' else f'chain-{index}-{alias}-{arm}')
    output.mkdir(mode=0o700) # create-once; existing attempt is never reused
    stage='SOURCE';completed=[];f=model=module=account=None;cold=None;coldM=None;coldsha=None
    started=time.perf_counter();restored=False
    def terminate(signum,frame):raise RuntimeError(f'EXTERNAL_TERMINATION_{signum}')
    signal.signal(signal.SIGTERM,terminate)
    try:
        sys.dont_write_bytecode=True
        official=old._bootstrap_easyedit(Path(source['official']['root']))
        if official['head']!=assets.OFFICIAL_EASYEDIT_HEAD or not official['tracked_clean']:raise RuntimeError('OFFICIAL_SOURCE_DRIFT')
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true',PYTHONDONTWRITEBYTECODE='1')
        random.seed(20260906);torch.manual_seed(20260906);torch.cuda.manual_seed_all(20260906)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')
        torch.set_num_threads(8)
        if torch.cuda.device_count()!=1:raise RuntimeError('ONE_GPU_PROCESS')
        torch.cuda.set_device(0)
        from transformers import AutoModelForCausalLM,AutoTokenizer
        snapshot=old._model_snapshot(assets.HF_HUB_CACHE_ROOT,alias)
        stage='MODEL_LOAD';model_start=time.perf_counter()
        model=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float32,low_cpu_mem_usage=True,device_map={'':'cuda:0'},attn_implementation='eager')
        tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True)
        if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
        tok.padding_side='right';model.config.pad_token_id=tok.pad_token_id;model.config.use_cache=False;model.eval()
        old.seal_eager_attention(model);dtype=assert_full_fp32(model);account=Accounting(model)
        hp,hppath=old._load_hparams(repo,'AlphaEdit',alias)
        if tuple(hp.layers)!=LAYERS:raise RuntimeError('FULL_LAYER_INVENTORY')
        hp.device=0;hp.stats_dir=str(assets.EASYEDIT_ARTIFACT_ROOT/'examples/data/stats')
        hp.P_loc=str(assets.EASYEDIT_ARTIFACT_ROOT/assets.MODEL_BINDINGS[alias]['projector'][0])
        module=old._method_module('AlphaEdit');module.CONTEXT_TEMPLATES_CACHE=None
        account.bind_native(module)
        with old._model_name(model,str(hp.model_name)):contexts=module.get_context_templates(model,tok)
        save(output/'runtime.lock.json',dict(source=source,alias=alias,arm=arm,mode=mode,
            official=official,snapshot=str(snapshot),hparams=str(hppath),dtype=dtype,
            parameter_inventory=old._parameter_inventory(model),contexts_sha256=assets.canonical_hash(contexts),
            sample_root=sample['ordered_root'],slurm_job=os.environ.get('SLURM_JOB_ID'),
            cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_device=torch.cuda.get_device_name(0),
            model_load_seconds=time.perf_counter()-model_start,model_load_count=1,
            autocast=False,tf32=False,precision='model/storage FP32; native metric/controller FP64'))
        batches=[[data[x]] for x in sample['smoke_ids']] if mode=='smoke' else [
            [data[r['case_id']] for r in sample['records'] if r['batch_index']==k] for k in range(1,11)]
        previous=None;seen=[]
        for k,rows in enumerate(batches,1):
            stage=f'B{k}_ENTRY';batch_start=account.snapshot();batchdir=output/f'batch-{k:02}'
            f=ObservedFamily(model=model,tokenizer=tok,family='AlphaEdit',hparams=hp,module=module,
                requests=old._official_requests(raw_requests(rows)),endpoint_records=rows,
                request_order_sha256=assets.canonical_hash([assets.canonical_hash(r['requested_rewrite']) for r in rows]),
                contexts=contexts,capture_persistent_endpoint=True)
            if k==1:
                f.prepare_method_state()
                if bool(torch.count_nonzero(module.cache_c)):raise RuntimeError('COLD_M0_NOT_ZERO')
                cold={n:p.detach().cpu().clone() for n,p in f.parameters.items()};coldsha=tensor_set_sha256(cold)
                cold_pointers={n:p.data_ptr() for n,p in f.parameters.items()}
                coldM=module.cache_c.detach().clone()
                if mode=='main':
                    if (root/'server4-takeover.lock.json').is_file():
                        from .server4_takeover import w0_reference
                        reference=w0_reference(root,alias)
                    else:
                        gate_binding=json.loads((root/'smoke-gates.lock.json').read_text())
                        reference=json.loads((Path(gate_binding['root'])/f'smoke-{alias}'/'W0-full.json').read_text())
                    if reference['W0_sha256']!=coldsha or reference['sample_root']!=sample['ordered_root']:
                        raise RuntimeError('COMMON_ORIGINAL_W0_REFERENCE_IDENTITY')
            else:f.bind_existing_method_state()
            check_entry(f,previous)
            entry=dict(batch_index=k,alias=alias,arm=arm,W_entry=f.w0_sha256,
                M_entry=tensor_sha256(module.cache_c),method_state_entry=f.method_state_identity(),
                previous_commit=previous,source_head=source['head'],sample_root=sample['ordered_root'],
                case_ids=[int(r['case_id']) for r in rows],request_order_sha256=f.request_order_sha256,
                cold_initialization_count=int(k==1),new_family_and_dictionary=True)
            save(batchdir/'entry.json',entry)
            stage=f'B{k}_TARGET';before=account.snapshot();f.compute_fixed_z();old._sync();target_compute=account.difference(before)
            with torch.no_grad():
                terminal=f.terminal();normalization=FrozenNormalization.capture(f.fixed_z.values,terminal,f.w0_sha256)
                dictionary=NativeDictionary(f)
                # O_NATIVE has no JV controller/reference inserted in its write path.
                metric=None
                if arm!='O_NATIVE':metric=dictionary.capture_reference(dictionary.build(terminal,0))
                f.metric_observer=dictionary
            save(batchdir/'target-reference.json',dict(fixed_z_sha256=f.fixed_z.identity_sha256,
                fixed_z_request_count=f.fixed_z.request_count,compute_z=target_compute,metric=metric,
                normalization_scales=normalization.scales.tolist(),normalization_active=normalization.active.tolist(),
                W_entry=f.w0_sha256,M_entry=entry['M_entry'],inner_z_reoptimization_count=0))
            stage=f'B{k}_WRITE';before=account.snapshot()
            if arm=='O_NATIVE':result=dict(arm=arm,endpoint=f.run_official(fixed_z=f.fixed_z),nodes=[],main_jvp_count=0)
            else:result=run_joint(f,arm,dictionary,normalization,batch_index=k,output=batchdir/'nodes')
            old._sync();write_compute=account.difference(before)
            result['history_finalization_compute']=account.finish_history(f.evaluation_seconds if arm=='O_NATIVE' else 0.)
            if write_compute['history_key_captures']!=5:raise RuntimeError('POST_HISTORY_FULL_INVENTORY_COUNT')
            endpoint=result['endpoint'];current_raw=endpoint['evaluation']
            save(batchdir/'current-full-raw.json',current_raw)
            endpoint['evaluation']=public_full(current_raw,rows)
            result['actual_physical_action']=f.last_physical_action
            if arm=='O_NATIVE':
                result['actual_physical_action']['native_net_normalized']=None
                result['actual_physical_action']['normalized_status']='NOT_APPLICABLE_NO_JV_REFERENCE_IN_OFFICIAL_PATH'
            result.update(compute_z=target_compute,write_including_endpoint=write_compute,
                endpoint_evaluation_seconds=f.evaluation_seconds,fixed_z_sha256=f.fixed_z.identity_sha256)
            save(batchdir/'writer.json',result)
            stage=f'B{k}_COMMIT';before=account.snapshot();commit=commit_checked(f,endpoint)
            commit.update(batch_index=k,case_ids=entry['case_ids'],sample_root=sample['ordered_root'],
                W_entry=entry['W_entry'],M_entry=entry['M_entry'],fixed_z_sha256=f.fixed_z.identity_sha256)
            commit['compute']=account.difference(before);save(batchdir/'commit.json',commit);previous=commit
            stage=f'B{k}_SEEN_EVAL';before=account.snapshot()
            if seen:
                if k in CHECKPOINTS:
                    older=evaluate_counterfact_with_canonical_ns(model,tok,seen,device=f.device,microbatch_size=16)
                    combined=join_panels([older,current_raw]);save(batchdir/'seen-full-raw.json',combined)
                    save(batchdir/'seen-full.json',public_full(combined,seen+rows))
                else:older=rewrite_only(f,seen)
                rewrite=join_panels([{x:older[x] for x in ('rewrite_target_new','rewrite_target_true')},
                    {x:current_raw[x] for x in ('rewrite_target_new','rewrite_target_true')}])
            else:
                rewrite={x:current_raw[x] for x in ('rewrite_target_new','rewrite_target_true')}
                save(batchdir/'seen-full.json',endpoint['evaluation'])
            save(batchdir/'rewrite-retention.json',public_rewrite(rewrite,commit['committed_weight_sha256'],k))
            old._sync();seen_compute=account.difference(before);seen+=rows
            stage=f'B{k}_CHECKPOINT';before=account.snapshot();cp=None
            if k in CHECKPOINTS or mode=='smoke':
                cp=checkpoint(output/'checkpoints'/f'W{k}-M{k}.pt',f,commit,
                    dict(source=source,sample_root=sample['ordered_root'],arm=arm,alias=alias,batch=k))
            summary=dict(status='BATCH_COMMITTED',batch_index=k,alias=alias,arm=arm,
                requested=len(rows),seen_requested=len(seen),current={key:endpoint['evaluation'][key]
                    for key in ('preference','locality','kind_summaries')},
                commit=commit,checkpoint=cp,seen_compute=seen_compute,checkpoint_compute=account.difference(before),
                full_batch_compute=account.difference(batch_start),technical_failure_count=0,
                semantic_success_filtering_count=0,peak_gpu_allocated=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved=torch.cuda.max_memory_reserved(),peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save(batchdir/'complete.json',summary);completed.append(summary)
            # Release the previous family snapshot and metric before the next full dense cache clone.
            f.metric_observer=None;del dictionary,result,terminal,current_raw,endpoint,normalization
            f=None;gc.collect();torch.cuda.empty_cache()
        stage='RESTORE_W0'
        parameters={n:model.get_parameter(n) for n in cold}
        with torch.no_grad():
            for n,p in parameters.items():p.copy_(cold[n].to(p.device))
            module.cache_c.copy_(coldM);module.cache_c_new=True
        restored=tensor_set_sha256(parameters)==coldsha and torch.equal(module.cache_c,coldM)
        restored=restored and {n:p.data_ptr() for n,p in parameters.items()}==cold_pointers
        if not restored:raise RuntimeError('FINAL_W0_M0_RESTORE')
        if mode=='smoke':
            # One common original-W0 reference per model, after smoke state is discarded.
            stage='W0_REFERENCE';rows=[data[r['case_id']] for r in sample['records']];before=account.snapshot()
            raw=evaluate_counterfact_with_canonical_ns(model,tok,rows,device=torch.device('cuda:0'),microbatch_size=16)
            save(output/'W0-full-raw.json',raw)
            save(output/'W0-full.json',dict(W0_sha256=coldsha,sample_root=sample['ordered_root'],
                evaluation=public_full(raw,rows),source_head=source['head'],compute=account.difference(before),
                reused_for_all_main_arms=True,controller_influence_count=0))
        save(output/'terminal-receipt.json',dict(status='TERMINAL_VALID',mode=mode,alias=alias,arm=arm,
            completed_batches=len(completed),requested=len(seen),source=source,sample_root=sample['ordered_root'],
            W0_sha256=coldsha,W0_restored=restored,cold_init_count=1,history_appends=len(completed),
            final_commit=previous,technical_failure_count=0,imputation_count=0,
            total_seconds=time.perf_counter()-started,compute=account.snapshot(),scientific_promotion=False))
    except BaseException as exc:
        save(output/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(exc),
            traceback=traceback.format_exc(),completed_batches=len(completed),source=source,
            sample_root=sample['ordered_root'],prefix_preserved=True,imputation_count=0))
        raise
    finally:
        if account is not None:account.close()
        # Failure keeps immutable prefix receipts/checkpoints. No implicit retry.
        if model is not None and cold is not None and module is not None and not restored:
            with torch.no_grad():
                for n,v in cold.items():model.get_parameter(n).copy_(v.to('cuda:0'))
                module.cache_c.copy_(coldM);module.cache_c_new=True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=('smoke','main'),required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();run(a.repo,a.root,a.mode,a.index)

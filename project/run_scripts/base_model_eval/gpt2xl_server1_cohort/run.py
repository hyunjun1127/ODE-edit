"""One actual cold W0 score pass, followed only by saved-row cohort reduction."""
import os
import resource
import time
from .common import *
from .curves import attach_occurrence_ordinals,build_curves,curve_coverage,reduce_rows,validate_rows

def tracking_start(c,lock,output):
    from .tracking import init
    cfg=dict(server='server1',task_id=TASK,arm='W0_BASE_MODEL',attempt=c['run_instance']['attempt'],
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],model='gpt2xl',model_family='gpt2',
        metric_schema=SCHEMA,job_id=os.environ['SLURM_JOB_ID'],**c['reference_config'])
    tracker=init(env_file=c['tracking']['env_file'],spool=output/'tracking',config=cfg,run_id=c['run_instance']['run_id'])
    write(output/'tracking-identity.json',dict(run_id=tracker.run_id,url=tracker.startup.get('url'),
        config=tracker.config_values,startup=tracker.startup,immutable_identity=True,
        startup_before_model_load=True,scientific_completion=False))
    return tracker

def observe(view,bench,records,expected,output,tracker=None):
    from project.run_scripts.jlz_price_gpt2xl.scores import scores
    before,selected,hooks=view.guard(),view.selected_state(),view.hooks()
    require(all(not keys for groups in hooks.values() for keys in groups),'COHORT_NO_EXTRA_HOOKS')
    raw=[];progress_acceptance=[];started=time.monotonic()
    try:
        for start in range(0,len(records),50):
            capacity(output);specs,pairs=pair_specs(records[start:start+50],bench)
            values=scores(view,bench,pairs,2)
            require(len(values)==2*len(specs),'COHORT_ACTUAL_CANDIDATE_COUNT')
            for index,row in enumerate(specs):
                for label,value in zip(('new','true'),values[2*index:2*index+2]):
                    row.update({label+'_'+key:value for key,value in value.items()})
                row.update(occurrence_ordinal=start+index//13,**{key:0 for key in
                    ('actual_model_edits','actual_applied_edits','pre_state_edits','post_state_edits')},
                    margin_true_minus_new=row['true_nll']-row['new_nll'],
                    margin_new_minus_true=row['new_nll']-row['true_nll'])
            validate_rows(specs,expected[start*13:start*13+len(specs)])
            check_guard(before,view.guard());require(hooks==view.hooks(),'COHORT_HOOK_MUTATION')
            write(output/f'chunk-{start:04d}.json',dict(schema='w0-occurrence-rows-v1',rows=specs,selected_W=selected,
                optimizer_feedback=False,fresh_W0=True,reference_only=True,actual_model_edits=0,
                actual_applied_edits=0,pre_state_edits=0,post_state_edits=0,history_present=False))
            raw.extend(specs)
            if tracker is not None:
                accepted=tracker.log(dict(step=start+len(records[start:start+50]),phase_id=2,
                    **{'time/elapsed_seconds':time.monotonic()-started,
                       'memory/host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
                progress_acceptance.append(dict(requests=start+len(records[start:start+50]),accepted=accepted,
                    rejection=getattr(tracker,'last_rejection',None) if not accepted else None))
    finally:
        check_guard(before,view.guard())
        require(hooks==view.hooks() and selected==view.selected_state(),'COHORT_SELECTED_BYTE_MUTATION')
    validate_rows(raw,expected,require_full=True)
    groups=reduce_rows(raw)
    require({kind:value['denominator'] for kind,value in groups.items()}==DENOMINATORS and
        view.physical_candidate_rows==52000,'COHORT_FULL_DENOMINATORS')
    summary=dict(endpoint='W0',requests=2000,summary=groups,prompt_pairs=len(raw),candidate_rows=2*len(raw),
        ordered_occurrence_rows=digest([[row['occurrence_ordinal'],row['identity']] for row in raw]),selected_W=selected,
        no_mutation=True,nonselected_byte_certificate='NOT_ESTABLISHED; pointer/version/scope guard only',
        fresh_W0=True,previous_raw_used_as_measurement=False,physical_forward_calls=view.calls,
        physical_candidate_rows=view.physical_candidate_rows,seconds=time.monotonic()-started,
        actual_model_edits=0,edit_calls=0,target_fits=0,solves=0,history_appends=0,C0_P_loads=0,checkpoint_saved=False)
    write(output/'summary.json',summary);write(output/'progress-acceptance.json',progress_acceptance)
    return raw,summary

def emit_curves(raw,expected,c,output,tracker):
    """Only CPU arithmetic, including the x0 baseline before monotonically rising x."""
    payloads=build_curves(raw,expected)
    config=dict(c['reference_config'],metric_schema=SCHEMA)
    coverage=curve_coverage(payloads,config)
    write(output/'curve-payloads.json',dict(config=config,payloads=payloads,coverage=coverage,
        new_model_forward_calls=0,fresh_source='same attempt/W0/chunk-*',previous_raw_reuse=False))
    acceptances=[]
    for payload in payloads:
        accepted=tracker.log(payload)
        acceptances.append(dict(edits=payload['edits'],accepted=accepted,
            delivery='SDK_ASYNC_NOT_REMOTE_ACK' if accepted else 'LOGGING_DEGRADED_REJECTED_POINT',
            rejection=getattr(tracker,'last_rejection',None) if not accepted else None))
    value=dict(coverage=coverage,acceptances=acceptances,
        status='SDK_ASYNC_NOT_REMOTE_ACK' if all(item['accepted'] for item in acceptances) else 'LOGGING_DEGRADED',
        scientific_actual_edits=0,new_model_forward_calls=0)
    write(output/'curve-acceptance.json',value);return value

def run(config_path,lock_path):
    started=time.monotonic();c,lock=verify_lock(config_path,lock_path)
    require(os.environ.get('ODEEDIT_W0_COHORT_SOURCE')==lock['source_commit'],'COHORT_EXPLICIT_RUNTIME_SOURCE')
    output=Path(c['attempt'])/'W0';require(not output.exists(),'COHORT_NO_REEVALUATION_OVERWRITE');output.mkdir()
    tracker=None;terminal={};stage='TRACKING_STARTUP'
    try:
        tracker=tracking_start(c,lock,output)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
        require((str(torch.__version__),transformers.__version__)==(c['runtime']['torch'],c['runtime']['transformers']),'COHORT_PINNED_RUNTIME')
        torch.set_num_threads(8);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(not torch.is_autocast_enabled(),'COHORT_AUTOCAST_OFF')
        stage='COLD_MODEL_LOAD';load=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
            attn_implementation='eager',use_safetensors=True).cuda().eval();model.requires_grad_(False)
        require(all(bool(torch.isfinite(weight).all()) for weight in model.parameters()),'COHORT_MODEL_NONFINITE')
        view=W0View(model);require(view.selected_state()==c['cold_selected_W'],'COHORT_COLD_WEIGHT_IDENTITY')
        tokenizer=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
        tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
        bench=CounterFactAdapter(tokenizer,read(c['contexts']['path']))
        records=load_prefix(Path(c['stream']).parent,2000)
        require(digest([row['case_id'] for row in records])==c['ordered_ids_sha256'],'COHORT_RUNTIME_ORDER')
        expected=read(c['observer_identity']['path'])['rows']
        require(attach_occurrence_ordinals(token_identity(records,bench))==expected,'COHORT_RUNTIME_TOKEN_ORDINAL_IDENTITY')
        write(output/'runtime.json',dict(torch=str(torch.__version__),transformers=transformers.__version__,
            device=torch.cuda.get_device_name(),model=c['model'],revision=c['model_revision'],source=lock['source_commit'],
            config=lock['config_sha256'],FP32=True,eager=True,autocast=False,TF32=False,model_eval=True,
            fresh_model_load=True,load_seconds=time.monotonic()-load,selected_W=view.selected_state(),C0_P_H_loaded=False))
        stage='FRESH_W0_SINGLE_OBSERVATION';raw,summary=observe(view,bench,records,expected,output,tracker)
        stage='SAVED_ROWS_CPU_CURVES';logging=emit_curves(raw,expected,c,output,tracker)
        terminal=dict(status='COMPLETED',requests=2000,prompt_pairs=26000,candidate_rows=52000,
            source=lock['source_commit'],config=lock['config_sha256'],fresh_W0=True,
            summary=member(output/'summary.json'),curves=member(output/'curve-payloads.json'),
            curve_delivery=logging['status'],curve_coverage=logging['coverage'])
    except BaseException as error:
        terminal=dict(status='FAILED',stage=stage,error_type=type(error).__name__,error=str(error)[:1200],
            partial_chunks=len(list(output.glob('chunk-*.json'))),scientific_complete=False,automatic_retry=False)
        write(output/'failure.json',terminal)
    finally:
        if tracker is not None:
            try:write(output/'tracking-finish.json',tracker.finish(exit_code=0 if terminal.get('status')=='COMPLETED' else 1,timeout=45))
            except Exception as error:write(output/'tracking-finish-error.json',dict(error_type=type(error).__name__,logger_failure_does_not_repeat_science=True))
        import torch
        terminal.update(source=lock['source_commit'],config=lock['config_sha256'],program_seconds=time.monotonic()-started,
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE',actual_model_edits=0,actual_applied_edits=0,
            edit_calls=0,target_fits=0,solves=0,history_appends=0,stats_loads=0,projector_loads=0)
        write(output/'terminal.json',terminal)
    return terminal

"""Conditional selected final-CF W20 generation only; no writer or job submission.

NativeGenerationObserver/scoring/profile are reused unchanged. Checkpoint file
SHA is authoritative; mmap exposes only selected weights to the restore path.
cache_c, contexts and RNG are never restored or cloned for evaluation.
"""
import argparse
from pathlib import Path
import torch
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,verify,member,require
from official.runners.server1 import flucon_eval as common
from official.runners.fe_cake_flucon import bind_weights
from official.diagnostics.fe_qwen_current import benchmark
from official.diagnostics.fe_qwen_current.contract import NONCE
from official.diagnostics.fe_qwen_current.rescue import SPEC,validated_selection
from official.diagnostics.fe_qwen_current.storage import tensor_sha,finite_tensor
from official.evaluation.factual import validate_retained_observation
from official.evaluation.reduce import counterfact
from official.evaluation.generation.assets import load_assets,_manifest,_identity
from official.evaluation.generation.common import digest as generation_digest
from official.evaluation.generation.native_observer import NativeGenerationObserver,read_observed
from official.evaluation.generation.native_profile import PROFILE,runtime_identity
from official.evaluation.generation.observer import _record_identity,occurrence
from official.evaluation.generation.metrics import generation_payload
from official.evaluation.generation.paper_display import paper_generation
from official.tracking import init,official_generation_progress
from official.tracking.schema import config as validate_tracking,bind_job_identity,job_identity

TASK='fe-qwen-current-z-final-cf-flucon-20261012'
REFERENCE='75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6'
ORDER='0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
STREAM='66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37'
HP=dict(layers=[4,5,6,7,8],rewrite_module_tmp='model.layers.{}.mlp.down_proj')
SELECTED_SHAPE=(3584,18944)
COUNT=2000


def load_reference(path):
    manifest,_=_manifest(path)
    require(manifest['identity_sha256']==REFERENCE and generation_digest(_identity(manifest))==REFERENCE,
        'ORIGINAL_REFERENCE_CONTENT_IDENTITY')
    return load_assets(path)


def generation_config(c,lock):
    return dict(model_identity=c['model_identity'],profile=PROFILE,eval_seed=20261007,
        generation_source_sha=dict(code_commit=lock['source_commit'],official_tree=lock['official_tree']))


def endpoint_binding(c,lock):
    return dict(checkpoint=c['original']['checkpoint'],original_identity=c['original']['identity'],
        final_result=c['final_result'],source=lock['source_commit'],config_sha256=c['config_sha256'])


def verify_generation(result,records,c,lock):
    """Bind shared independently scored raw to the exact original ordered prompts."""
    runtime=generation_digest(runtime_identity(generation_config(c,lock),REFERENCE))
    record_ids=[_record_identity(r,occurrence(r)) for r in records]
    stream=generation_digest(dict(runtime=runtime,eval_seed=20261007,ordered_record_identities=record_ids))
    binding=endpoint_binding(c,lock)
    expected=dict(runtime=runtime,state_sha256=generation_digest(binding),endpoint='W20',cohort='first2000',
        ordered_occurrences=[occurrence(r) for r in records],sampling_stream_sha256=stream,
        observation_identities=[generation_digest(dict(runtime=runtime,state_identity=binding,
            record_identity=r,sampling_stream_sha256=stream)) for r in record_ids])
    require(result['identity']==expected,'GENERATION_EXACT_QUERY_RUNTIME_BINDING')
    require(len(result['rows'])==result['summary']['planned_count']==COUNT,'GENERATION_COUNT')


def audit_final(final_member):
    """Read-only re-reduction of both final runs; never calls the writing collector."""
    final=read(verify(final_member));root=Path(final_member['path']).parent
    require(final.get('both_W20_complete') is True,'BOTH_FINAL_W20_REQUIRED')
    lock=read(root/'lock.json');rows=final['rows']
    final_contract=read(verify(lock['selection_lock']))
    require(final_contract['stage']=='final_benchmark' and final_contract['spec']==SPEC
        and final_contract['spec_sha256']==digest(SPEC),'RESCUE_FINAL_SOURCE_CONTRACT')
    for m in lock['source_members']+lock['author_members']:verify(m)
    require(len(rows)==2 and sorted(r['dataset'] for r in rows)==['cf','zsre'],'FINAL_PAIR')
    require(sorted(r['config']['sha256'] for r in rows)==sorted(m['sha256'] for m in lock['configs']),'FINAL_CONFIG_MEMBERS')
    chosen=None;cf=None
    for row in rows:
        c=read(verify(row['config']));out=Path(c['output'])
        require(c['config_sha256']==digest({k:v for k,v in c.items() if k!='config_sha256'}),'FINAL_CONFIG_SHA')
        proof=validated_selection(c['selected_result'])
        require(c.get('stage')=='final_benchmark' and c.get('cold_start') is True
            and c['dataset']==row['dataset'] and c['model']=='qwen25'
            and c['L2']==row['L2']==proof['selection']['selected_L2']
            and row['status']=='W20_CPU_RAW_VERIFIED' and row['requests']==2000
            and row['committed_batches']==20 and row['source']==c['source']==lock['source_commit'],'FINAL_SCOPE')
        require(c['selection_contract_sha256']==digest(SPEC) and c['L2']==100,'RESCUE_FINAL_CONTRACT')
        if chosen is None:chosen=c['selected_result']
        require(c['selected_result']==chosen==final_contract['prior_selected_result'],'FINAL_PAIR_SELECTION')
        done=read(out/'COMPLETE.json');ident=done['identity']
        require(done['status']=='W20_COMPLETE' and (done['edits'],done['batches'])==(2000,20)
            and ident['config_sha256']==c['config_sha256'] and ident['source']==c['source']
            and ident['stage']=='final_benchmark' and ident['dataset']==c['dataset']
            and ident['selected_result']==chosen and ident['L2']==c['L2']
            and ident['selection_contract_sha256']==digest(SPEC)
            and ident['stream_sha256']==c['stream']['sha256']
            and ident['split']=='original_'+c['dataset']+'_first2000'
            and ident['z_reference']=='batch_entry' and ident['dtype']=='bfloat16','FINAL_COMPLETE_IDENTITY')
        verify(row['checkpoint']);require(done['checkpoint']==row['checkpoint'],'FINAL_CHECKPOINT')
        inputs=read(verify(c['stream']));full_queries=None;seen=set()
        require(len(inputs)==2000,'FINAL_INPUT_COUNT')
        for b in range(21):
            commit=read(out/'commits'/f'B{b:02d}.json')
            require(commit['identity']==ident and (commit['batch'],commit['edits'])==(b,b*100),'FINAL_COMMIT')
            labels=['W0'] if b==0 else [f'B{b:02d}-current']
            if b in SPEC['milestones']:labels.append(f'B{b:02d}-all-seen')
            require(len(labels)==len(commit['raw']),'FINAL_ENDPOINT_COUNT')
            for label,rm in zip(labels,commit['raw']):
                seen.add(label);saved=row['endpoints'][label];raw=read(verify(rm))
                ei=raw['identity'].get('external_identity',raw['identity'])
                require(all(ei[k]==v for k,v in ident.items()),'FINAL_RAW_IDENTITY')
                receipt=read(out/'evaluations'/f'{label}.json');plan=read(verify(receipt['queries']))
                require(receipt['identity']==ident and receipt['raw']==rm==saved['raw'],'FINAL_RAW_RECEIPT')
                expected=inputs if b==0 else inputs[:b*100] if label.endswith('all-seen') else inputs[(b-1)*100:b*100]
                require([r['case_id'] for r in raw['cases']]==[r['case_id'] for r in expected],'FINAL_RAW_ORDER')
                require(raw['summary']['requests']==len(expected),'FINAL_REQUEST_DENOMINATOR')
                if c['dataset']=='zsre':
                    if b==0:
                        full_queries=plan
                        require(plan['query_sha256']==read(verify(c['query_parity']))['query_sha256'],'FINAL_QUERY_PARITY')
                    benchmark.verify_query_subset(full_queries,plan,expected)
                    reduced=benchmark.reduce_zsre(raw,plan)
                    require(all(saved[k]==v for k,v in reduced.items()),'FINAL_ZSRE_REDUCTION')
                else:
                    audit=validate_retained_observation(raw,'cf',input_signature=plan)
                    values=counterfact(raw['cases'])
                    require(saved['metrics']==values and saved['validation']==audit,'FINAL_CF_REDUCTION')
                    require(all(values[k]==raw['summary'][k] for k in ('Efficacy','Generalization','Specificity','Score','requests')),'FINAL_CF_SUMMARY')
        require(set(row['endpoints'])==seen and row['W20']==row['endpoints']['B20-all-seen']
            and row['W0']==row['endpoints']['W0'] and done['endpoint']==row['W20']['raw'],'FINAL_REPORT_BINDING')
        if c['dataset']=='cf':
            cf=dict(row=row,config=c,identity=ident,terminal=member(out/'COMPLETE.json'),
                pointer=member(out/'checkpoints/latest.json'),commit=member(out/'commits/B20.json'))
    return cf


def payload_weights(original):
    """Trusted owned pickle: full SHA before mmap; no traversal/copy of H storage."""
    cp=original['checkpoint'];pointer=read(verify(original['pointer']))
    require(pointer['sha256']==cp['sha256'] and pointer['bytes']==cp['bytes']
        and pointer['path']==cp['path']
        and pointer['batch']==20 and pointer['final_W20'] is True
        and pointer['identity']==original['identity'],'CURRENT_W20_POINTER')
    p=torch.load(verify(cp),map_location='cpu',mmap=True,weights_only=False)
    weights,integrity=validate_payload(p,original)
    del p
    verify(cp)
    return weights,integrity


def validate_payload(p,original):
    require(set(p)=={'weights','cache_c','contexts','z_member','batch','cursor','identity','rng','integrity_sha256'},'CURRENT_PAYLOAD_SCHEMA')
    require(p['batch']==20 and p['identity']==original['identity'],'CURRENT_PAYLOAD_IDENTITY')
    require(p['identity']['stage']=='final_benchmark' and p['identity']['dataset']=='cf'
        and p['identity']['split']=='original_cf_first2000'
        and p['identity']['selection_contract_sha256']==digest(SPEC) and p['identity']['L2']==100
        and p['identity']['z_reference']=='batch_entry' and p['identity']['dtype']=='bfloat16','FINAL_CF_NOT_HELDOUT')
    cursor=p['cursor'];native=cursor['native_config']
    require(cursor['edits']==2000 and digest(cursor['case_ids'])==ORDER
        and cursor['raw']==read(verify(original['commit']))['raw'],'CURRENT_W20_CURSOR')
    require(native['bs']==100 and native['num_edits']==2000 and native['seed']==0
        and native['model_dtype']=='bfloat16' and native['algs']['add_old_keys'] is True
        and native['algs']['L2']==p['identity']['L2']
        and {k:native['llms'][k] for k in HP}==HP,'CURRENT_NATIVE_CONFIG')
    required=dict(clamp_norm_factor=1,v_num_grad_steps=35,v_lr=.5,v_weight_decay=.001,
        kl_factor=.0625,v_loss_layer=27,mom2_update_weight=[15000]*5)
    require(all(native['llms'][k]==v for k,v in required.items()),'CURRENT_AUTHOR_HPARAMS')
    require(p['z_member']==dict(policy='batch_entry_current100',durable_target_cache=False),'CURRENT_Z_POLICY')
    integrity=p['integrity_sha256']
    require(type(integrity) is str and len(integrity)==64 and all(x in '0123456789abcdef' for x in integrity),'PAYLOAD_INTEGRITY_FIELD')
    if 'payload_integrity_sha256' in original:require(integrity==original['payload_integrity_sha256'],'PAYLOAD_INTEGRITY_BINDING')
    names={HP['rewrite_module_tmp'].format(l)+'.weight' for l in HP['layers']}
    require(set(p['weights'])==names,'CURRENT_SELECTED_NAMES')
    for value in p['weights'].values():
        require(value.dtype==torch.bfloat16 and value.device.type=='cpu' and value.is_contiguous()
            and tuple(value.shape)==SELECTED_SHAPE and finite_tensor(value),'CURRENT_SELECTED_BF16')
    # The file SHA authenticates all payload bytes. Do NOT state_digest(cache_c)
    # again or restore checkpoint RNG/contexts/history into evaluation.
    return p['weights'],integrity


def restore(model,original):
    weights,integrity=payload_weights(original)
    restored=bind_weights(model,weights,original,torch.bfloat16)
    del weights
    return dict(checkpoint=original['checkpoint'],identity=original['identity'],
        payload_integrity_sha256=integrity,restored_selected_weights_fp32_sha256=restored,
        cast='EXACT_BF16_TO_FP32_ROUNDTRIP',evaluation='FP32_EAGER_TF32_OFF',
        nonselected_unchanged=True,history_contexts_rng_consumed=False,edit_calls=0)


def tracking_values(c,source):
    # Reuse the standing shared eval-only schema authority; actual task authority
    # is separately immutable as registration_authority, never impersonated.
    proxy=dict(c,original=dict(c['original'],identity=c['model_identity']))
    values=common.tracking_values(proxy,source)
    values.update(server=c['server'],task_id=TASK,baseline='FE-selected-current-z-final-CF',writer='fe_native_batch_entry_z')
    return validate_tracking(values)


def run(config_path,lock_path):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from official.runners.server1.fe_history_prepare import runtime
    c=read(config_path);lock=read(lock_path)
    require(c['registration_authority']==NONCE and c['instruction']==common.INSTRUCTION
        and c['config_sha256']==digest({k:v for k,v in c.items() if k!='config_sha256'})
        and member(config_path) in lock['configs'],'FLUCON_CONFIG')
    require(Path(__file__).resolve().is_relative_to(Path(lock['source_directory']).resolve()),'FROZEN_FLUCON_SOURCE')
    for m in lock['source_members']:verify(m)
    require(runtime()==c['runtime'],'FLUCON_RUNTIME')
    audited=audit_final(c['final_result']);original=c['original']
    require(audited['row']['checkpoint']==original['checkpoint'] and audited['identity']==original['identity'],'FLUCON_FINAL_BINDING')
    records=read(verify(c['stream']))
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDER
        and digest(records)==digest(read(verify(audited['config']['stream']))),'GENERATION_ORIGINAL_QUERY_ORDER')
    assets=read(verify(c['assets']))
    require(c['assets']==audited['config']['model_assets'] and c['model_identity']==dict(model='qwen25',
        model_revision=assets['model']['identity']['revision'],tokenizer_sha256=assets['tokenizer_sha256'],
        model_assets_sha256=c['assets']['sha256']),'ORIGINAL_BASE_MODEL_TOKENIZER')
    for m in assets['model']['members']:verify(m)
    refs=load_reference(verify(c['reference']));require(refs.sha==REFERENCE,'REFERENCE_IDENTITY')
    from official.diagnostics.fe_qwen_current.flucon_prepare import storage_admission
    storage_admission(c['output'],c['storage']['reserve_bytes'],c['storage']['raw_margin_bytes'])
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    values=bind_job_identity(tracking_values(c,lock['source_commit']))
    require(job_identity(values)['execution_backend']!='local','ACTUAL_EVAL_JOB_REQUIRED')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'EVAL_GPU1')
    tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=values);code=1
    def log(values):require(tracker.log(values) is not False,'FLUCON_TRACKING_REJECTED')
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        snap=assets['model']['snapshot'];tok=AutoTokenizer.from_pretrained(snap,local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='right'
        model=AutoModelForCausalLM.from_pretrained(snap,local_files_only=True,torch_dtype=torch.float32,
            low_cpu_mem_usage=True,attn_implementation='eager').to('cuda:0').eval()
        model.config.use_cache=False
        for p in model.parameters():p.requires_grad_(False)
        restored=restore(model,original);write_new(out/'restore.json',restored)
        parameters=dict(model.named_parameters());names=set(original['payload_tensor_sha256'])
        binding=endpoint_binding(c,lock)
        def state():return dict(binding=binding,selected_weights={n:tensor_sha(parameters[n]) for n in names})
        before=state()
        observer=NativeGenerationObserver(model,tok,refs,generation_config(c,lock),
            out/'generation',state_callback=state,progress_callback=lambda v:log(official_generation_progress(v,endpoint='W20')))
        result=observer.observe(records,'W20',cohort='first2000',state_identity=binding)
        require(state()==before and result['summary']['planned_count']==len(result['rows'])==2000,'GENERATION_COVERAGE_STATE')
        checked=read_observed(result['rows_path'],assets=refs)
        verify_generation(checked,records,c,lock)
        require(checked['summary']==result['summary'],'GENERATION_CPU_REDUCTION')
        verify(original['checkpoint']);verify(original['pointer'])
        payload=generation_payload('all_seen/post',result['summary']);payload.update(edits=2000,post_state_edits=2000);log(payload)
        write_new(out/'COMPLETE.json',dict(config=member(config_path),source=lock['source_commit'],
            endpoint=member(result['rows_path']),summary=result['summary'],paper=paper_generation(result['summary']),
            binding=binding,state=before,requests=2000,edit_calls=0,W0_generation=0,checkpoint_keep=True))
        code=0
    except BaseException as error:
        write_new(out/'FAILURE.json',dict(type=type(error).__name__,error=str(error),checkpoint_keep=True));raise
    finally:tracker.finish(exit_code=code,timeout=45)


def collect(config_path,lock_path,output):
    c=read(config_path);lock=read(lock_path);done=read(Path(c['output'])/'COMPLETE.json')
    require(member(config_path) in lock['configs'] and done['config']==member(config_path)
        and done['source']==lock['source_commit']==c['source']
        and c['config_sha256']==digest({k:v for k,v in c.items() if k!='config_sha256'})
        and c['registration_authority']==NONCE and c['instruction']==common.INSTRUCTION,'GENERATION_RESULT_SOURCE')
    audited=audit_final(c['final_result'])
    require(audited['identity']==c['original']['identity'] and audited['row']['checkpoint']==c['original']['checkpoint'],'COLLECT_FINAL_BINDING')
    refs=load_reference(verify(c['reference']));require(refs.sha==REFERENCE,'REFERENCE_IDENTITY')
    result=read_observed(verify(done['endpoint']),assets=refs)
    verify_generation(result,read(verify(c['stream'])),c,lock)
    require(result['identity']['endpoint']=='W20' and result['identity']['cohort']=='first2000'
        and result['identity']['state_sha256']==generation_digest(done['binding'])
        and result['summary']==done['summary'] and len(result['rows'])==result['summary']['planned_count']==2000,'GENERATION_RESULT_IDENTITY')
    require(done['binding']==endpoint_binding(c,lock),'GENERATION_CHECKPOINT_BINDING')
    write_new(output,dict(status='CPU_RAW_VERIFIED',source=lock['source_commit'],config=member(config_path),
        checkpoint=c['original']['checkpoint'],endpoint=done['endpoint'],summary=result['summary'],
        paper=paper_generation(result['summary']),raw_units=dict(Flu='bits',Con='cosine_0_to_1'),GPU_parity_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','collect']);p.add_argument('config');p.add_argument('lock');p.add_argument('--output');a=p.parse_args()
    if a.mode=='run':run(a.config,a.lock)
    else:collect(a.config,a.lock,a.output)

"""One immutable cold GPT-J W0 generation endpoint; no edits or checkpoints."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from official.experiments.prepare import digest, file_sha, write_new
from official.evaluation.generation.native_profile import PROFILE
from official.evaluation.generation.metrics import generation_payload
from official.tracking.schema import config as validate_tracking

INSTRUCTION='USER-GH-GPTJ-W0-FLUCON-SERVER4-20261011-R1'
TASK='gptj-w0-flucon-20261011'
REVISION='47e169305d2e8376be1d31e765533382721b2cc1'
TOKENIZER_FILES=('added_tokens.json','merges.txt','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json')

def read(p): return json.loads(Path(p).read_text())
def member(p):
    p=Path(p).absolute()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))
def verify(m):
    p=Path(m['path'])
    if p.stat().st_size!=m['bytes'] or file_sha(p)!=m['sha256']: raise ValueError('INPUT_SHA_MISMATCH')
    return p
def runtime():
    return {n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','numpy','nltk','scipy','scikit-learn')}

def tracking_values(c):
    return dict(server='server4',task_id=TASK,arm='GPTJ_W0_FLUCON',attempt=c['attempt'],
        source_sha=c['source'],config_sha=c['config_sha256'],model='gptj',model_family='gptj',
        writer='none',baseline='W0',role='eval_only',metric_schema='official-baselines-scalar-v1',
        dataset='cf',instruction_id=INSTRUCTION,evaluation_profile='cf-native-generation-W0-only-v1',
        base_model_sha256=digest([m for m in c['model_members'] if Path(m['path']).suffix in ('.bin','.safetensors')]),
        evaluator_sha256=c['evaluator_sha256'],stream_sha256=c['stream']['sha256'],
        tokenizer_sha256=c['tokenizer_sha256'],source_run_id='61723',
        generation_metric_schema='counterfact-cake-generation-metrics-v1',generation_profile=PROFILE,
        generation_eval_seed=20261007,reference_assets_sha256=c['reference_identity'],
        generation_source_sha=c['source'],generation_repair_instruction=INSTRUCTION,
        generation_schedule='W0_ONLY_FIRST2000')

def verify_records(c):
    records=read(verify(c['stream']))
    if len(records)!=2000 or [r['occurrence_index'] for r in records]!=list(range(1,2001)):
        raise ValueError('FULL_FIRST2000_OCCURRENCE_REQUIRED')
    if digest([r['case_id'] for r in records])!=c['ordered_case_ids_sha256']:
        raise ValueError('ORDERED_CASE_ID_MISMATCH')
    if any(not isinstance(r.get('generation_prompts'),list) for r in records):
        raise ValueError('GENERATION_PROMPTS_MISSING')
    return records

def preflight(c, source_root):
    if c['instruction']!=INSTRUCTION or c['revision']!=REVISION: raise ValueError('AUTHORITY_MODEL')
    if digest({k:v for k,v in c.items() if k!='config_sha256'})!=c['config_sha256']:
        raise ValueError('CONFIG_HASH')
    if runtime()!=c['runtime']: raise ValueError('RUNTIME_IDENTITY')
    for relative,expected in c['source_members'].items():
        if file_sha(Path(source_root)/relative)!=expected: raise ValueError('SOURCE_CHANGED')
    verify_records(c)
    return validate_tracking(tracking_values(c))

def run(path):
    c=read(path);source_root=Path(__file__).resolve().parents[3]
    cfg=preflight(c,source_root)
    import shutil
    if shutil.disk_usage(Path(c['output']).parent).free<c['storage_reserve_bytes']:
        raise ValueError('DISK_RESERVE_REQUIRED_KEEP_SOURCE')
    for m in c['model_members']: verify(m)
    verify(c['reference_manifest'])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from official.evaluation.generation.assets import load_assets
    from official.evaluation.generation.native_observer import NativeGenerationObserver, read_observed
    from official.evaluation.generation.paper_display import paper_cell
    from official.tracking import init, official_generation_progress
    if torch.cuda.device_count()!=1: raise ValueError('SINGLE_GPU_REQUIRED')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    refs=load_assets(dict(generation_assets=c['reference_manifest'],asset_paths=c['reference_paths']))
    if refs.sha!=c['reference_identity']: raise ValueError('REFERENCE_IDENTITY')
    records=verify_records(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    tracker=init(env_file=c['tracking_env_file'],spool=out/'tracking',config=cfg)
    code=1
    def log(values):
        if tracker.log(values) is False: raise ValueError('TRACKING_REJECTED')
    try:
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='right'
        model=AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,dtype=torch.float32,
            attn_implementation='eager',low_cpu_mem_usage=True).to('cuda:0').eval()
        model.config.use_cache=False
        for p in model.parameters(): p.requires_grad_(False)
        def state():
            return {n:(p.data_ptr(),p._version,str(p.dtype),list(p.shape)) for n,p in model.named_parameters()}
        before=state()
        binding=dict(model_revision=REVISION,model_members_sha256=digest(c['model_members']),
            tokenizer_sha256=c['tokenizer_sha256'],actual_model_edits=0,completed_batch=0,model_state='COLD_W0')
        observer=NativeGenerationObserver(model,tok,refs,dict(model_identity=binding,profile=PROFILE,
            eval_seed=20261007,generation_source_sha=c['source']),out/'generation',state_callback=state,
            progress_callback=lambda v:log(official_generation_progress(v,endpoint='W0')))
        result=observer.observe(records,'W0',cohort='OFFICIAL_CF_FIRST2000',state_identity=binding)
        if state()!=before or not result['RNG_restored'] or not result['observer_no_mutation']:
            raise ValueError('MODEL_OR_RNG_MUTATION')
        actual=read_observed(result['rows_path'],assets=refs)
        if actual['summary']!=result['summary'] or len(actual['rows'])!=2000:
            raise ValueError('INCOMPLETE_RAW')
        payload=generation_payload('W0_first2000',actual['summary'])
        payload.update(edits=0,pre_state_edits=0,post_state_edits=0);log(payload)
        summary=actual['summary']
        write_new(out/'COMPLETE.json',dict(instruction=INSTRUCTION,job_id=os.environ.get('SLURM_JOB_ID'),
            source=c['source'],config=member(path),endpoint='W0',edits=0,requests=2000,
            raw=member(result['rows_path']),summary=summary,identity=actual['identity'],
            paper_Flu=paper_cell(summary.get('ngram_entropy'),metric='Flu',raw_unit='bits'),
            paper_Con=paper_cell(summary.get('reference_score'),metric='Con',raw_unit='cosine_0_to_1'),
            checkpoint_saved=False,weights_unchanged=True,RNG_restored=True))
        code=0
    except BaseException as e:
        write_new(out/'FAILURE.json',dict(type=type(e).__name__,error=str(e),automatic_retry=False));raise
    finally:
        receipt=tracker.finish(exit_code=code,timeout=45)
        write_new(out/'transport-final.json',dict(receipt=receipt,science_exit_code=code))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    if a.preflight: preflight(read(a.config),Path(__file__).resolve().parents[3])
    else: run(a.config)

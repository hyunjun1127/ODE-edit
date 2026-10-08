"""Fail-closed non-sensitive config and scalar schemas; no shell evaluation."""
import configparser
import math
import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit
from .method import (COMPARISON_SCHEMA, OFFICIAL_SCHEMA, METHOD_CONFIG,
                     METHOD_METRICS, GENERATION_METRICS, OFFICIAL_METRICS,
                     validate as validate_method)

ENTITY = 'wkdguswns2256'
PROJECT = 'layer allocation'
SDK_VERSION = '0.30.0'
OFFICIAL_INSTRUCTION = 'USER-OFFICIAL-BASELINES-20261008-R1'
OFFICIAL_GENERATION_SCHEDULE = 'W0_AND_W20_FIRST2000'
OFFICIAL_DEFERRED_W20_SCHEDULE = 'W0_ONLY_W20_DEFERRED_CHECKPOINT'
NATIVE_GENERATION_PROFILE = 'cf-cake-native-casebatch-kv-total100-globalrng-v1'
JOB_FIELDS = {'job_id','array_job_id','array_task_id','step_id','job_display_id','execution_backend','identity_source'}
SLURM_ENV = {'job_id':'SLURM_JOB_ID','array_job_id':'SLURM_ARRAY_JOB_ID',
             'array_task_id':'SLURM_ARRAY_TASK_ID','step_id':'SLURM_STEP_ID'}
CONFIG_KEYS = {'server', 'task_id', 'arm', 'attempt', 'source_sha', 'config_sha', 'parent_run_id',
               'source_run_id','source_run_url','observation_identity','baseline',
               'generation_metric_schema','generation_profile','generation_eval_seed',
               'reference_assets_sha256','generation_source_sha',
               'generation_qualification_plan_sha256','generation_repair_instruction',
               'generation_schedule','instruction_id','dataset'} | JOB_FIELDS | METHOD_CONFIG
METRICS = {
    'setup_ok','step','batch','edits','candidate','phase_id','status_code',
    'fit/loss','fit/nll','fit/kl','fit/norm','fit/gradient_norm',
    'optimizer/calls','optimizer/accepted','optimizer/rejected','optimizer/stop_code',
    'eval/RS','eval/PS','eval/NS','eval/harmonic',
    'eval/R_numerator','eval/R_denominator','eval/P_numerator','eval/P_denominator',
    'eval/N_numerator','eval/N_denominator','eval/true_nll','eval/new_nll','eval/desired_nll',
    'eval/TF_token_micro','eval/TF_prompt_macro','eval/TF_strict',
    'eval/TF_token_correct','eval/TF_token_count','eval/TF_strict_numerator','eval/TF_strict_denominator',
    'retention/lost','retention/gained','retention/denominator','retention/cohort_id',
    'time/elapsed_seconds','time/phase_seconds','time/allocated_gpu_seconds',
    'memory/gpu_allocated_bytes','memory/gpu_reserved_bytes','memory/host_rss_bytes',
    'resource/gpus','resource/cpus','logging/dropped_points'
}
ENV_KEYS = {'WANDB_ENTITY','WANDB_PROJECT','WANDB_MODE','WANDB_CONSOLE','WANDB_SAVE_CODE',
            'WANDB_BASE_URL','ODEEDIT_WANDB_PYTHON'}
METRICS |= METHOD_METRICS
GENERATION_PROGRESS_FIELDS=('completed_cases','total_cases','completed_prompts','total_prompts',
    'generated_tokens','new_cases','reused_cases','elapsed_sec','cases_per_sec','prompts_per_sec',
    'tokens_per_sec','physical_forward_calls','prefill_query_tokens','decode_query_tokens','step')
GENERATION_PROGRESS_METRICS={'generation_progress/'+key for key in GENERATION_PROGRESS_FIELDS}
METRICS |= GENERATION_PROGRESS_METRICS | {'phase'}
GENERATION_PHASES = ('W0_generation', 'generation_evaluation', 'W20_generation')


def official_generation_progress(values, *, endpoint):
    """Map the native observer's W20 phase; preserve counts and separate axis.

    NativeGenerationObserver uses generation_evaluation for W20. This adapter
    changes only observational phase naming, not generation, RNG, or scores.
    """
    require(endpoint in ('W0','W20') and type(values) is dict,
            'OFFICIAL_GENERATION_CALLBACK_ENDPOINT')
    expected='W0_generation' if endpoint=='W0' else 'W20_generation'
    allowed=('W0_generation',) if endpoint=='W0' else ('generation_evaluation','W20_generation')
    require(values.get('phase') in allowed,'OFFICIAL_GENERATION_CALLBACK_PHASE')
    result=dict(values);result['phase']=expected
    metrics(result,scientific=True)
    return result


def require(ok, code):
    if not ok:
        raise ValueError(code)


def identifier(value):
    require(type(value) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value), 'INVALID_IDENTIFIER')
    return value


def config(values):
    require(type(values) is dict and set(values) <= CONFIG_KEYS, 'CONFIG_NOT_ALLOWLISTED')
    require({'server','task_id','arm','attempt','source_sha'} <= set(values), 'MISSING_CONFIG')
    result = {}
    for key,value in values.items():
        if key in ('source_sha','config_sha','observation_identity','reference_assets_sha256','generation_source_sha',
                   'generation_qualification_plan_sha256'):
            require(type(value) is str and re.fullmatch(r'[a-f0-9]{40}|[a-f0-9]{64}',value), 'INVALID_SHA')
        elif key=='generation_eval_seed':
            require(type(value) is int and value==20261007,'GENERATION_EVAL_SEED')
        elif key=='generation_repair_instruction':
            require(type(value) is str and value in (
                'USER-GH-SH1-SH2-BASELINE-GENERATION-KV-BATCH-REPAIR-20261008-R1',
                'USER-GH-SH1-GPT2XL-BLUE-PRUNE-RECT-W20-GENERATION-20261008-R1',
                'USER-GH-SH1-GPT2XL-MEMIT-ALPHAEDIT-CAKE-W20-GENERATION-20261008-R1',
                'USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1',
                OFFICIAL_INSTRUCTION),
                'GENERATION_REPAIR_INSTRUCTION')
        elif key=='generation_schedule':
            require(type(value) is str and value in (
                'W20_ONLY_FIRST2000', OFFICIAL_GENERATION_SCHEDULE, OFFICIAL_DEFERRED_W20_SCHEDULE), 'GENERATION_SCHEDULE')
        elif key == 'step_id':
            step_identifier(value)
        elif key == 'source_run_url':
            parsed=urlsplit(value)
            require(parsed.scheme=='https' and parsed.hostname in ('wandb.ai','forge.coreweave.com')
                    and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
                    and '/runs/' in parsed.path, 'INVALID_SOURCE_RUN_URL')
        else:
            identifier(value)
        result[key] = value
    require(result['server'] in ('server1','server2','server3','server4'), 'INVALID_SERVER')
    if METHOD_CONFIG & result.keys():
        require(METHOD_CONFIG|{'config_sha'} <= result.keys(),'METHOD_CONFIG_REQUIRED')
        require(result['metric_schema'] in (COMPARISON_SCHEMA, OFFICIAL_SCHEMA),'METHOD_SCHEMA_UNREGISTERED')
        require(result['model'] in ('llama3','gptj','qwen','qwen25','gpt2xl'),'MODEL_ALIAS_UNREGISTERED')
        require(result['role'] in ('scientific','derived_comparison_snapshot'),'METHOD_ROLE')
    official=result.get('metric_schema')==OFFICIAL_SCHEMA
    if official:
        require(result.get('instruction_id')==OFFICIAL_INSTRUCTION
                and result.get('dataset') in ('cf','zsre'), 'OFFICIAL_AUTHORITY_DATASET_REQUIRED')
        require(result['model'] in ('llama3','gptj','qwen25'), 'OFFICIAL_MODEL_ALIAS')
    elif 'instruction_id' in result or 'dataset' in result:
        require(False,'OFFICIAL_CONFIG_REQUIRES_OFFICIAL_SCHEMA')
    generation={'generation_metric_schema','generation_profile','generation_eval_seed',
                'reference_assets_sha256','generation_source_sha'}
    if official and result['dataset']=='cf':
        require(generation <= result.keys(),'OFFICIAL_CF_GENERATION_CONFIG_REQUIRED')
    if generation & result.keys():
        require(generation|METHOD_CONFIG|{'baseline'}<=result.keys(),'GENERATION_CONFIG_REQUIRED')
        require(result['generation_metric_schema']=='counterfact-cake-generation-metrics-v1'
                and result['generation_profile'] in (
                    'cf-cake-prompt-inclusive-total100-eos-corrected-v1',
                    'cf-cake-native-casebatch-kv-total100-globalrng-v1'),
                'GENERATION_SCHEMA_PROFILE')
        if result['generation_profile']==NATIVE_GENERATION_PROFILE:
            if official:
                require(result['dataset']=='cf'
                        and result.get('generation_repair_instruction')==OFFICIAL_INSTRUCTION
                        and result.get('generation_schedule') in (OFFICIAL_GENERATION_SCHEDULE, OFFICIAL_DEFERRED_W20_SCHEDULE),
                        'OFFICIAL_NATIVE_GENERATION_AUTHORITY_SCHEDULE')
                if result.get('generation_schedule')==OFFICIAL_DEFERRED_W20_SCHEDULE:
                    require(result['server']=='server1' and result['model']=='llama3'
                            and result['writer'] in ('ft','memit','memit_fe','none'),
                            'OFFICIAL_DEFERRED_W20_SERVER1_SCOPE')
            else:
                require(result.get('generation_repair_instruction')==
                    'USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1'
                    and result.get('generation_schedule')=='W20_ONLY_FIRST2000',
                    'NATIVE_GENERATION_EXPLICIT_REPAIR_AUTHORITY')
        elif official:
            require(False,'OFFICIAL_GENERATION_PROFILE_CHANGED')
    elif 'generation_schedule' in result or 'generation_repair_instruction' in result:
        require(False,'GENERATION_CONFIG_REQUIRED')
    if official and result['dataset']=='zsre':
        require(not any(k.startswith('generation_') or k=='reference_assets_sha256'
                        for k in result),'ZSRE_GENERATION_FORBIDDEN')
    return result


def step_identifier(value):
    # Slurm step sentinels may be signed; they are metadata, not job IDs.
    require(type(value) is str and re.fullmatch(r'(?:-?[0-9]+|[A-Za-z][A-Za-z0-9_.-]{0,63})',value), 'INVALID_SLURM_STEP_ID')
    return value


def job_identity(cfg):
    result = {k:cfg[k] for k in JOB_FIELDS if k in cfg}
    if result.get('execution_backend') == 'local':
        require(result == dict(execution_backend='local',identity_source='NOT_APPLICABLE'), 'LOCAL_JOB_ID_FORBIDDEN')
        return result
    require(result.get('execution_backend')=='slurm' and result.get('identity_source')=='SLURM_ENV', 'JOB_IDENTITY_MISSING')
    require(bool(re.fullmatch(r'[1-9][0-9]*',result.get('job_id',''))), 'SLURM_JOB_ID_REQUIRED')
    require(('array_job_id' in result)==('array_task_id' in result), 'INCOMPLETE_ARRAY_IDENTITY')
    display = result['job_id']
    if 'array_job_id' in result:
        require(bool(re.fullmatch(r'[1-9][0-9]*',result['array_job_id'])) and
                bool(re.fullmatch(r'0|[1-9][0-9]*',result['array_task_id'])), 'INVALID_ARRAY_IDENTITY')
        display = result['array_job_id']+'_'+result['array_task_id']
    if 'step_id' in result: step_identifier(result['step_id'])
    require(result.get('job_display_id')==display,'JOB_DISPLAY_MISMATCH')
    return result


def bind_job_identity(values, environ=None):
    """Capture only four allowlisted keys in the parent, before env isolation."""
    cfg=config(values); env=os.environ if environ is None else environ
    raw={k:env.get(v) for k,v in SLURM_ENV.items()}
    # Empty optional step exports do not identify a step; never fabricate one.
    if raw['step_id']=='':raw['step_id']=None
    if any(v is not None for v in raw.values()):
        identity={k:v for k,v in raw.items() if v is not None}
        identity.update(execution_backend='slurm',identity_source='SLURM_ENV')
        identity['job_display_id']=(str(raw['array_job_id'])+'_'+str(raw['array_task_id'])
            if raw['array_job_id'] is not None else raw['job_id'])
        job_identity(identity)
    else:
        identity=dict(execution_backend='local',identity_source='NOT_APPLICABLE')
    for key in JOB_FIELDS & cfg.keys():
        require(cfg[key]==identity.get(key),'JOB_IDENTITY_CALLER_ENV_MISMATCH:'+key)
    cfg.update(identity)
    job_identity(cfg)
    return cfg


def run_name(cfg):
    identity=job_identity(cfg)
    suffix='job'+identity['job_display_id'] if identity['execution_backend']=='slurm' else 'local'
    return cfg['server']+'-'+cfg['arm']+'-'+cfg['attempt']+'-'+suffix


def metrics(values,*,scientific=False,config_values=None):
    require(type(values) is dict and 0 < len(values) <= len(METRICS), 'METRIC_MAPPING')
    require(set(values) <= METRICS, 'METRIC_NOT_ALLOWLISTED')
    # No float(tensor), .item(), .cpu(), arbitrary __float__, or GPU sync.
    require(all((key=='phase' and type(x) is str and x in GENERATION_PHASES) or
                (key!='phase' and type(x) in (int,float,bool) and math.isfinite(x))
                for key,x in values.items()), 'BUILTIN_FINITE_SCALARS_ONLY')
    if GENERATION_PROGRESS_METRICS & values.keys():
        require(type(values.get('phase')) is str and values['phase'] in GENERATION_PHASES
            and 'generation_progress/step' in values,'GENERATION_PROGRESS_PHASE_AXIS_REQUIRED')
        require(not ({'edits','pre_state_edits','post_state_edits'} & values.keys())
            and not any(k.startswith(('W0_first2000/','current/','all_seen/','w0/','fit/','optimizer/'))
            for k in values),'GENERATION_PROGRESS_NOT_ENDPOINT_OR_FIT')
        for key in GENERATION_PROGRESS_METRICS & values.keys():
            value=values[key]
            require(type(value) in (int,float) and value>=0,'GENERATION_PROGRESS_NONNEGATIVE')
            if key.rsplit('/',1)[1] not in ('elapsed_sec','cases_per_sec','prompts_per_sec','tokens_per_sec'):
                require(type(value) is int,'GENERATION_PROGRESS_INTEGER')
        for completed,total in (('completed_cases','total_cases'),('completed_prompts','total_prompts')):
            if {'generation_progress/'+completed,'generation_progress/'+total}<=values.keys():
                require(values['generation_progress/'+completed]<=values['generation_progress/'+total],
                    'GENERATION_PROGRESS_COVERAGE')
    if OFFICIAL_METRICS & values.keys():
        require(config_values is not None and config_values.get('metric_schema')==OFFICIAL_SCHEMA,
                'OFFICIAL_METRIC_CONFIG_REQUIRED')
    if config_values is not None and config_values.get('metric_schema')==OFFICIAL_SCHEMA:
        cfg=config(config_values)
        generation_keys=GENERATION_METRICS & values.keys()
        progress=GENERATION_PROGRESS_METRICS & values.keys()
        if cfg['dataset']=='zsre':
            require(not generation_keys and not progress and 'phase' not in values,
                    'ZSRE_GENERATION_FORBIDDEN')
        if generation_keys or progress:
            require(cfg.get('generation_profile')==NATIVE_GENERATION_PROFILE
                    and cfg.get('generation_schedule') in (OFFICIAL_GENERATION_SCHEDULE, OFFICIAL_DEFERRED_W20_SCHEDULE),
                    'OFFICIAL_GENERATION_CONFIG_REQUIRED')
            if cfg.get('generation_schedule')==OFFICIAL_DEFERRED_W20_SCHEDULE:
                require(not any(key.startswith('all_seen/post/') for key in generation_keys)
                        and (not progress or values.get('phase')=='W0_generation'),
                        'OFFICIAL_W20_GENERATION_DEFERRED')
        if progress:
            require(values['phase'] in ('W0_generation','W20_generation'),
                    'OFFICIAL_GENERATION_PHASE')
        if generation_keys:
            prefixes={key.rsplit('/generation/',1)[0] if '/generation/' in key else
                key.rsplit('/fluency/',1)[0] if '/fluency/' in key else
                key.rsplit('/consistency/',1)[0] for key in generation_keys}
            require(len(prefixes)==1 and prefixes<= {'W0_first2000','all_seen/post'},
                    'OFFICIAL_GENERATION_ENDPOINT_ONLY')
            prefix=next(iter(prefixes));expected=0 if prefix=='W0_first2000' else 2000
            require(values.get('edits')==expected
                    and values.get(prefix+'/generation/planned_count')==2000,
                    'OFFICIAL_GENERATION_EXACT_ENDPOINT')
    return validate_method(dict(values),scientific=scientific,
        official=config_values is not None and config_values.get('metric_schema')==OFFICIAL_SCHEMA)


def endpoint(value):
    parsed = urlsplit(value)
    require(parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment and parsed.path in ('','/'), 'UNVERIFIED_API_ENDPOINT')
    # In particular, do not derive a base_url from the user's project UI URL.
    return value.rstrip('/')


def load_env(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        require('=' in line, 'CONFIG_SYNTAX')
        key,value = line.split('=',1)
        require(key in ENV_KEYS and key not in result, 'CONFIG_KEY_NOT_ALLOWED')
        parts = shlex.split(value)
        require(len(parts)==1 and not any(x in parts[0] for x in ('$','`','\n')), 'CONFIG_VALUE_NOT_ALLOWED')
        result[key] = parts[0]
    require(result.get('WANDB_ENTITY')==ENTITY and result.get('WANDB_PROJECT')==PROJECT, 'PROJECT_IDENTITY')
    require(result.get('WANDB_MODE')=='online' and result.get('WANDB_CONSOLE')=='off'
            and result.get('WANDB_SAVE_CODE')=='false', 'PRIVACY_CONFIG')
    require(Path(result.get('ODEEDIT_WANDB_PYTHON','')).is_file(), 'SDK_PYTHON_MISSING')
    result['WANDB_BASE_URL'] = endpoint(result.get('WANDB_BASE_URL','https://api.wandb.ai'))
    return result


def existing_endpoint():
    """Existing local config only; never inspect or return credential fields."""
    value = os.environ.get('WANDB_BASE_URL')
    if value:
        return endpoint(value), 'EXISTING_ENVIRONMENT'
    settings = Path(os.environ.get('WANDB_CONFIG_DIR',str(Path.home()/'.config/wandb'))) / 'settings'
    if settings.is_file():
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(settings)
        value = parser.get('default','base_url',fallback=None)
        if value:
            return endpoint(value), 'EXISTING_SETTINGS'
    return 'https://api.wandb.ai', 'SDK_CLOUD_DEFAULT'

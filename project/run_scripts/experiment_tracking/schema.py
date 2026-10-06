"""Fail-closed non-sensitive config and scalar schemas; no shell evaluation."""
import configparser
import math
import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit

ENTITY = 'wkdguswns2256'
PROJECT = 'layer allocation'
SDK_VERSION = '0.30.0'
JOB_FIELDS = {'job_id','array_job_id','array_task_id','step_id','job_display_id','execution_backend','identity_source'}
SLURM_ENV = {'job_id':'SLURM_JOB_ID','array_job_id':'SLURM_ARRAY_JOB_ID',
             'array_task_id':'SLURM_ARRAY_TASK_ID','step_id':'SLURM_STEP_ID'}
CONFIG_KEYS = {'server', 'task_id', 'arm', 'attempt', 'source_sha', 'config_sha', 'parent_run_id'} | JOB_FIELDS
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
        if key in ('source_sha','config_sha'):
            require(type(value) is str and re.fullmatch(r'[a-f0-9]{40}|[a-f0-9]{64}',value), 'INVALID_SHA')
        else:
            identifier(value)
        result[key] = value
    require(result['server'] in ('server1','server2','server3','server4'), 'INVALID_SERVER')
    return result


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
    if 'step_id' in result: identifier(result['step_id'])
    require(result.get('job_display_id')==display,'JOB_DISPLAY_MISMATCH')
    return result


def bind_job_identity(values, environ=None):
    """Capture only four allowlisted keys in the parent, before env isolation."""
    cfg=config(values); env=os.environ if environ is None else environ
    raw={k:env.get(v) for k,v in SLURM_ENV.items()}
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


def metrics(values):
    require(type(values) is dict and 0 < len(values) <= len(METRICS), 'METRIC_MAPPING')
    require(set(values) <= METRICS, 'METRIC_NOT_ALLOWLISTED')
    # No float(tensor), .item(), .cpu(), arbitrary __float__, or GPU sync.
    require(all(type(x) in (int,float,bool) and math.isfinite(x) for x in values.values()), 'BUILTIN_FINITE_SCALARS_ONLY')
    return dict(values)


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

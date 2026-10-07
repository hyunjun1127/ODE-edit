"""Task-private scalar extension; SH1 shared logger source is not modified."""
import math
from project.run_scripts.experiment_tracking import schema as base
from project.run_scripts.experiment_tracking.schema import (
    ENTITY, PROJECT, SDK_VERSION, identifier, endpoint, load_env, require, job_identity, run_name)
PREFIXES=('current/pre','current/post','all_seen/post','W0_first2000','w0/current','w0/all_seen')
REASONS=('missing_generation_prompts','missing_reference','zero_generated_vector',
         'zero_reference_vector','nonfinite_score','length_cap_no_continuation')
FIELDS=('fluency/ngram_entropy','consistency/reference_score','generation/planned_count',
        'generation/fluency_count','generation/consistency_count','generation/generation_prompt_count',
        'generation/generated_token_count')+tuple('generation/missing_'+x+'_count' for x in REASONS)
GEN_KEYS={p+'/'+f for p in PREFIXES for f in FIELDS}
EXTRA_CONFIG={'generation_metric_schema','generation_profile','generation_eval_seed',
              'reference_assets_sha256','generation_source_sha','baseline'}
def config(values):
    require(type(values) is dict and set(values)<=base.CONFIG_KEYS|EXTRA_CONFIG,'GEN_CONFIG_ALLOWLIST')
    cfg=base.config({k:v for k,v in values.items() if k not in EXTRA_CONFIG})
    extra={k:values[k] for k in EXTRA_CONFIG if k in values}
    require(set(extra)==EXTRA_CONFIG,'GENERATION_CONFIG_REQUIRED')
    require(extra['generation_eval_seed']==20261007 and type(extra['generation_eval_seed']) is int,'GEN_SEED')
    require(extra['generation_metric_schema']=='counterfact-cake-generation-metrics-v1'
            and extra['generation_profile']=='cf-cake-prompt-inclusive-total100-eos-corrected-v1','GEN_SCHEMA')
    for k in ('reference_assets_sha256','generation_source_sha'):
        require(type(extra[k]) is str and len(extra[k]) in (40,64)
                and all(c in '0123456789abcdef' for c in extra[k]),'GEN_SOURCE_ASSET_SHA')
    identifier(extra['baseline'])
    require(cfg['model']=='gptj' and cfg['model_family']=='gptj' and cfg['role']=='scientific','GEN_GPTJ_ROLE')
    return dict(cfg,**extra)
def bind_job_identity(values,environ=None):
    cfg=config(values)
    plain={k:v for k,v in cfg.items() if k not in EXTRA_CONFIG}
    bound=base.bind_job_identity(plain,environ)
    return config(dict(bound,**{k:cfg[k] for k in EXTRA_CONFIG}))
def metrics(values,*,scientific=False):
    require(type(values) is dict and bool(values) and set(values)<=base.METRICS|GEN_KEYS,'GEN_METRICS_ALLOWLIST')
    require(all(type(v) in (int,float,bool) and math.isfinite(v) for v in values.values()),'GEN_FINITE_SCALARS')
    generation={k:v for k,v in values.items() if k in GEN_KEYS}
    plain={k:v for k,v in values.items() if k not in GEN_KEYS}
    if plain:base.metrics(plain,scientific=scientific)
    if generation:
        require(scientific and type(values.get('edits')) is int and values['edits']>=0,'GEN_AXIS')
        for p in PREFIXES:
            selected={k:v for k,v in generation.items() if k.startswith(p+'/')}
            if not selected:continue
            counts={key:values.get(p+'/generation/'+key) for key in ('planned_count','fluency_count','consistency_count','generation_prompt_count','generated_token_count')}
            require(all(type(v) is int and v>=0 for v in counts.values()),'GEN_COUNTS_REQUIRED')
            require(counts['fluency_count']<=counts['planned_count'] and counts['consistency_count']<=counts['planned_count'],'GEN_VALID_COUNT_RANGE')
            for k,v in selected.items():
                if '/generation/' in k:require(type(v) is int and v>=0,'GEN_COUNT_INTEGER')
            entropy=p+'/fluency/ngram_entropy';score=p+'/consistency/reference_score'
            require((entropy in selected)==(counts['fluency_count']>0),'GEN_ENTROPY_MISSING_NOT_ZERO')
            require((score in selected)==(counts['consistency_count']>0),'GEN_COSINE_MISSING_NOT_ZERO')
            if entropy in selected:require(type(selected[entropy]) in (int,float) and selected[entropy]>=0,'GEN_ENTROPY_BITS')
            if score in selected:require(type(selected[score]) in (int,float) and 0<=selected[score]<=1,'GEN_COSINE_RANGE')
            if p=='W0_first2000':require(values['edits']==0 and counts['planned_count']==2000,'GEN_W0_FIRST2K')
            if p.startswith('current/'):
                require(counts['planned_count']==100 and values.get('post_state_edits')==values['edits']
                        and values.get('pre_state_edits')==values['edits']-100,'GEN_CURRENT_STATE_COUNTS')
            if p=='all_seen/post':
                require(values['edits'] in (500,1000,1500,2000) and counts['planned_count']==values['edits']
                        and values.get('post_state_edits')==values['edits'],'GEN_SEEN_STATE_COUNTS')
    return dict(values)

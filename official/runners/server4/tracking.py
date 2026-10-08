"""Caller mapping only. Transport is the readonly common official distribution."""
from official.tracking import init, official_generation_progress
from official.tracking.schema import config as validate_config
from official.evaluation.generation.metrics import generation_payload


def config(config, assets, ready, attempt):
    result=dict(server='server4',task_id='official-baselines-20261008',
        arm=config['run_id'],attempt=attempt,source_sha=ready['code_commit'],
        config_sha=ready['config_sha256'],model='llama3',model_family='llama',
        writer=config['method'],baseline=config['method'],role='scientific',
        metric_schema='official-baselines-scalar-v1',
        instruction_id='USER-OFFICIAL-BASELINES-20261008-R1',dataset=config['dataset'])
    if config['dataset']=='cf':
        result.update(generation_metric_schema='counterfact-cake-generation-metrics-v1',
            generation_profile='cf-cake-native-casebatch-kv-total100-globalrng-v1',
            generation_eval_seed=20261007,reference_assets_sha256=assets['generation_identity_sha256'],
            generation_source_sha=ready['code_commit'],
            generation_repair_instruction='USER-OFFICIAL-BASELINES-20261008-R1',
            generation_schedule='W0_AND_W20_FIRST2000')
    return validate_config(result)


def start(configuration, assets, ready, output, attempt):
    return init(env_file=assets['tracking_env'],spool=output/'tracking'/attempt,
                config=config(configuration,assets,ready,attempt))


def generation(result, endpoint):
    prefix='W0_first2000' if endpoint=='W0' else 'all_seen/post'
    edits=0 if endpoint=='W0' else 2000
    return dict(edits=edits,post_state_edits=edits,**generation_payload(prefix,result['summary']))


def progress(tracker, endpoint):
    return lambda raw:tracker.log(official_generation_progress(raw,endpoint=endpoint))

"""Pure metadata for the original CAKE case-batched generation profile.

This deliberately does not share the EOS-corrected/per-prompt-seed namespace.
No model, GPU, tokenizer or native repository is accessed by these helpers.
"""
import copy

from .common import SCHEMA, EVAL_SEED, require

PROFILE = 'cf-cake-native-casebatch-kv-total100-globalrng-v1'
ROUTE = 'NATIVE_CASE_PADDED_KV_GLOBAL_RNG'
SOURCE = dict(
    primary_repository='CAKE',
    primary_commit='0b378234862bd76c69f58404ef84c27d5f4bf9ef',
    primary_file='util/generate.py',
    primary_file_sha256='43d219ecc1955a4e4f82ea22edc286b4774e52e917d4b375021b8d7329b0adf3',
    primary_function='generate_fast',
    comparison_repository='BLUE',
    comparison_commit='311b076a92e4ed0f14f5c8b4909732da781bc5f7',
    comparison_file='util/generate.py',
    comparison_file_sha256='51f41871d750a1fe86f39d4142fe8a5c12a41086c6d4a2cee98b72f809ff0366',
    attention_mask='CAKE_CUMULATIVE_PREFIX',
    decode='CAKE_NO_SKIP_SPECIAL_TOKENS_NFKD_DOUBLE_NEWLINE_REMOVE_ENDOFTEXT',
    compatibility_changes=['typed_input_cache_logit_finite_checks',
        'FP32_eval_no_autocast_TF32_guard', 'local_token_work_and_observation_metadata',
        'endpoint_RNG_and_model_native_state_restoration_guards'],
    bitwise_BLUE_equivalence_claim=False,
    native_source_modified=False)


def source_identity():
    return copy.deepcopy(SOURCE)


def runtime_identity(config, assets_sha=None):
    """A source-bound identity, without the old qualification/fallback route."""
    require(config.get('model_identity'), 'NATIVE_GENERATION_MODEL_IDENTITY_REQUIRED')
    require(config.get('profile', config.get('generation_profile', PROFILE)) == PROFILE,
            'NATIVE_GENERATION_PROFILE_CHANGED')
    require(config.get('eval_seed', config.get('generation_eval_seed', EVAL_SEED)) == EVAL_SEED,
            'NATIVE_GENERATION_SEED_CHANGED')
    require(config.get('generation_route', ROUTE) == ROUTE, 'NATIVE_GENERATION_ROUTE_CHANGED')
    require(not any(key in config for key in ('qualification_receipt_member',
        'qualification_plan_member', 'old_w0_reuse', 'generation_microbatch')),
        'NATIVE_GENERATION_OLD_QUALIFICATION_OR_REUSE_FORBIDDEN')
    source = config.get('generation_source_sha', config.get('source_identity'))
    require(source, 'NATIVE_GENERATION_SOURCE_IDENTITY_REQUIRED')
    return dict(schema=SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
        model_identity=config['model_identity'], generation_source_sha=source,
        reference_assets_sha256=assets_sha if assets_sha is not None else config['reference_assets_sha256'],
        route=ROUTE, original_generator=source_identity(),
        sampling_scope='ENDPOINT_GLOBAL_BATCH_STREAM', case_batching='ALL_GENERATION_PROMPTS',
        n_gen_per_prompt=1, top_k=5, max_total_tokens=100, EOS_stop=False)

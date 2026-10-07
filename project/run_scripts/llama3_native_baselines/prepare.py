"""CPU preparation only; do not submit or silently bypass shared generation READY.

Existing native/input/model assets are rebound using SHA receipts and stat.
Only the small native Python/YAML source closure is copied for source freezing.
"""
import argparse
import copy
import importlib.metadata
import json
import os
import shutil
import subprocess
import unicodedata
from pathlib import Path

from scripts.fixed_counterfact import load_prefix
from .common import (APP_ROOT, AUTHORITY_FILES, LOCAL, MODEL_REVISION, NONCE,
    ORDERED_SHA, PROFILE, ROOT, TASK, authority, batches, digest, member, require,
    stat_seal, verify, write)
from .native_binding import build_bindings, verify as verify_native
from .producer import transport_support

PRIOR = APP_ROOT / 'local/jlz-price-cap-base-repair-2k/method-metrics-20261007/config.json'
SHARED_NAMESPACE = ROOT / 'project/run_scripts/experiment_generation_eval'
REFERENCE_ROOT = APP_ROOT / 'local/baseline-generation-eval-assets/20261007'
W0_SOURCE = APP_ROOT / 'local/jlz-interference-priced-l1-2k/repair-59721/source'
W0_EVALUATOR = ('project/run_scripts/jlz_realization/observe.py',
    'project/run_scripts/jlz_realization/inputs.py',
    'project/run_scripts/jlz_interference_l1/observer_io.py')


def w0_evaluator_binding():
    rows = []
    for relative in W0_EVALUATOR:
        historical, current = member(W0_SOURCE / relative), member(ROOT / relative)
        require((historical['bytes'], historical['sha256']) ==
                (current['bytes'], current['sha256']), 'W0_CURRENT_EVALUATOR_CHANGED')
        rows.append(dict(relative=relative, historical=historical, current=current))
    return rows


def copy_native(bindings, output):
    """Immutable exact source copies, not a model/context/tensor transfer."""
    result = copy.deepcopy(bindings)
    for method, binding in result.items():
        target = Path(output) / method
        target.mkdir(parents=True, exist_ok=False)
        binding['origin_root'] = binding['root']
        binding['origin_files'] = copy.deepcopy(binding['files'])
        binding['private_source_copy'] = True
        for row in binding['files']:
            source = verify_native(row)
            destination = target / row['relative']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            new = member(destination)
            require((new['bytes'], new['sha256']) == (row['bytes'], row['sha256']),
                    'PRIVATE_NATIVE_COPY_BYTES')
            row.update(new)
        hp = binding['hparams']; source = verify_native(hp)
        destination = target / ('hparams' + source.suffix)
        shutil.copyfile(source, destination)
        binding['origin_hparams'] = hp
        binding['hparams'] = member(destination)
        require(binding['hparams']['sha256'] == hp['sha256'], 'PRIVATE_HPARAMS_BYTES')
        binding['root'] = str(target.resolve())
    return result


def generation_availability():
    """Presence is not software/asset READY, API qualification or model PASS."""
    required = ('attribute_snippets.json', 'idf.npy', 'tfidf_vocab.json')
    return dict(owner='SH1', profile=PROFILE,
        shared_namespace=str(SHARED_NAMESPACE),
        shared_source_present=SHARED_NAMESPACE.is_dir(),
        reference_root=str(REFERENCE_ROOT),
        reference_present={name: (REFERENCE_ROOT / name).is_file() for name in required},
        exact_ready_binding=None, source_review='NOT_BOUND',
        actual_model_qualification='NOT_OBSERVED',
        state='SHARED_EVALUATOR_REFERENCE_NOT_BOUND',
        transport=transport_support())


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to(LOCAL) and not output.exists(), 'NEW_TASK_LOCAL_PREPARATION')
    authority_members = authority()
    previous = json.loads(PRIOR.read_text())
    old = previous['models']['LLAMA']
    for row in old['assets'] + previous['runtime']['source_members']:
        stat_seal(row)
    records = load_prefix(Path(previous['stream']).parent, 2000)
    packs = [dict(batch=number, count=len(current), seen=len(seen),
                  occurrence_order=digest([r['case_id'] for r in current]))
             for number, current, seen in batches(records)]
    # This only inventories canonical generation_prompts. No prompt text is in
    # the compact receipt, and no model is loaded to fill missing observations.
    counts = [len(row.get('generation_prompts', [])) for row in records]
    raw_bindings = build_bindings()
    bindings = copy_native(raw_bindings, output / 'native-source')
    runtime = copy.deepcopy(previous['runtime'])
    runtime['torch'] = importlib.metadata.version('torch')
    runtime['transformers'] = importlib.metadata.version('transformers')
    runtime['GPU_runtime_qualification'] = 'NOT_OBSERVED_FOR_THIS_NATIVE_ATTEMPT'
    generation = generation_availability()
    config = dict(instruction_id=NONCE, task_id=TASK, model=old['model'],
        model_revision=MODEL_REVISION, seed=20261002, ordered_ids_sha256=ORDERED_SHA,
        stream=previous['stream'], authority_members=authority_members,
        assets=old['assets'], prior_input_runtime=member(PRIOR), runtime=runtime,
        cold_W=old['cold_W0_H0']['W'], W0_source_zero_H=old['cold_W0_H0']['H'],
        W0_reuse=old['W0_reuse'], W0_evaluator=w0_evaluator_binding(),
        observer_identity=old['observer_identity'],
        observation_identity=old['observation_identity'], packs=packs, native=bindings,
        generation=generation, tracking=dict(env_file=previous['tracking']['env_file']),
        noCP=True, z_disk_cache=False, exact_resume='NOT_AVAILABLE',
        resources=dict(project_cap=3, task_cap=3, per_job_gpu=1, cpu=8,
            host_mib=59392, host_hard_mib=60416, wall='2-00:00:00',
            collector_gpu=0, collector_cpu=8, collector_host_mib=24576,
            collector_wall='04:00:00', node='server4', partition='gpu',
            export='NONE', requeue=0, wall_is_eta=False, actual_admission='NOT_SUBMITTED'),
        stage='IMPLEMENTING_SHARED_GENERATION_NOT_BOUND', job_ids=[])
    write(output / 'config.prepared.json', config)
    summary = dict(task_id=TASK, stage=config['stage'], job_ids=[],
        output=str(output), config=member(output / 'config.prepared.json'),
        native_methods={method: dict(layers=binding['layers'],
            hparams=binding['scientific_fields'], source_commit=binding['source_commit'],
            source_files=len(binding['files']), private_copy=True)
            for method, binding in bindings.items()},
        generation_prompt_count=sum(counts), missing_generation_prompts=counts.count(0),
        W0_RPN_reuse='HASH_BOUND_SCALAR_ONLY; current native H recorded separately',
        generation_W0='NOT_OBSERVED', generation=generation,
        model_loads=0, GPU=0, scientific_fits=0, Slurm_writes=0,
        no_broadcast='NO_BROADCAST_NOT_REQUIRED: exact same-host small source preparation')
    write(output / 'preparation.json', summary)
    return summary


def bind_ready(prepared, output, attempt, shared_commit):
    """CPU-only receiver source/reference validation and serialization budget.

    The original prepared receipt remains untouched. No LM forward, stock fit,
    generation, toy, projection or Slurm action is made by this preparation.
    """
    from project.run_scripts.experiment_generation_eval.assets import load_assets
    from project.run_scripts.experiment_generation_eval.common import PROFILE as shared_profile
    from transformers import AutoTokenizer
    output, attempt = Path(output).resolve(), Path(attempt).resolve()
    require(output.is_relative_to(LOCAL) and not output.exists() and attempt.parent == LOCAL
            and not attempt.exists(), 'NEW_READY_PREPARATION')
    c = json.loads(verify(member(prepared)).read_text())
    authority()
    require(shared_profile == PROFILE and len(shared_commit) == 40, 'SH1_PROFILE_SOURCE')
    common_files = sorted(SHARED_NAMESPACE.glob('*.py'))
    require(bool(common_files), 'SH1_SOURCE_NOT_DELIVERED')
    sources = []
    for path in common_files:
        row = member(path)
        relative = str(path.relative_to(ROOT))
        original = subprocess.run(['git', 'show', shared_commit+':'+relative], cwd=ROOT,
            capture_output=True, check=True).stdout
        import hashlib
        require(hashlib.sha256(original).hexdigest() == row['sha256'], 'SH1_SEALED_SOURCE_BYTES')
        sources.append(row)
    manifest_path = REFERENCE_ROOT/'reference-ready-r1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    overrides = {name: str(REFERENCE_ROOT/'download-r1'/name) for name in manifest['files']}
    nltk_path = REFERENCE_ROOT/'nltk_data'
    require(Path(os.environ.get('NLTK_DATA', '')).resolve() == nltk_path,
            'EXACT_PRIVATE_NLTK_ENV_REQUIRED')
    assets = load_assets(dict(reference_assets=member(manifest_path), asset_paths=overrides))
    require(assets.sha == manifest['identity_sha256'] and manifest['status'] == 'READY',
            'SH1_REFERENCE_IDENTITY')
    records = load_prefix(Path(c['stream']).parent,2000)
    list(batches(records))
    missing = sum(not assets.snippets_for(r['requested_rewrite']['relation_id'],
        r['requested_rewrite']['target_new']['id']) for r in records)
    require(missing == 0, 'CANONICAL_FIRST2000_REFERENCE_COVERAGE')
    # Software/output upper bound from the actual immutable input+tokenizer,
    # without generating samples or recording text/token arrays in Git.
    tokenizer = AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
    prompts = [p for r in records for p in r.get('generation_prompts', [])]
    maximum_input = max(len(tokenizer(p,add_special_tokens=True)['input_ids']) for p in prompts)
    max_token_json = max(len(json.dumps(unicodedata.normalize('NFKD',
        tokenizer.decode([index],skip_special_tokens=False)), ensure_ascii=False).encode())-2
        for index in tokenizer.get_vocab().values())
    # Double token bound permits boundary decoding/replacement; JSON record
    # overhead has a separate 4KiB allowance. Per-case endpoint identities/sums
    # are 16KiB apart from their prompt observations. No duplicated text stream.
    token_bound = max(100,maximum_input)
    prompt_bound = 4096 + token_bound*(2*max_token_json+24)
    per_case = 16384 + max(len(r.get('generation_prompts',[])) for r in records)*prompt_bound
    # W0 is produced once, birth pre/post=4000 occurrences/arm and milestones
    # 500+1000+1500+2000=5000; overlaps deliberately overcounted for reserve.
    observed_cases_each = 9000
    generation_bytes = (2000+6*observed_cases_each)*per_case
    # RPN, authoritative native scalar JSONL, stdout/W&B/temp/error collector
    # reserves are fixed independent allowances, not quality-driven omissions.
    other_bytes = 6*(512<<20) + (2<<30)
    total = generation_bytes + other_bytes
    free = shutil.disk_usage(LOCAL).free
    require(free >= total, 'RESOURCE_BLOCKED_STORAGE_FULL_SIX_RUN_BOUND')
    output.mkdir()
    ready_path = output/'reference-receiver-ready.json'
    write(ready_path, dict(status='READY', identity_sha256=assets.sha,
        manifest=member(manifest_path), reference_files={name:member(path) for name,path in overrides.items()},
        ordered_ids_sha256=ORDERED_SHA, reference_coverage=2000, missing_reference=0,
        source_commit=shared_commit, source_members=sources, CPU_only=True,
        actual_model_generation='NOT_OBSERVED', transfer='EXACT_ONE_BUNDLE_PULL_NO_DELETE'))
    c['generation'] = dict(status='SOURCE_REFERENCE_BOUND',profile=PROFILE,
        source_sha=shared_commit,source_members=sources,assets_sha256=assets.sha,
        reference_manifest=member(manifest_path),asset_paths=overrides,
        READY=member(ready_path),nltk_data=str(nltk_path),W0_producer='MEMIT',
        W0_ready_path=str(attempt/'shared-generation-W0-ready.json'))
    c.update(attempt=str(attempt),run_instance={'attempt':'20261007-r1'},observer_microbatch=2,
        stage='CPU_SOURCE_BINDING_GPU_NOT_OBSERVED',job_ids=[],
        W0_evaluator=w0_evaluator_binding())
    c['resources'].update(qos='lab_gpu_s4',combined_reserve_bytes=total,reserve_bytes=total,
        wall_is_eta=False,wall_rationale='Unpadded full-prefix generation work plus native20fits; 48h request ceiling, measured ETA unavailable')
    c['storage'] = dict(startup_required_bytes=total,W0_required_bytes=total,
        # Largest next batch is W20: pre100 + measured post-allseen2000.
        # Subsets reuse text, but their scalar/identity framing is reserved too.
        next_batch_required_bytes=2100*per_case+(512<<20),
        next_batch_generation_case_bound=2100,free_observed_bytes=free,
        serializer_prompt_max_bytes=prompt_bound,serializer_case_max_bytes=per_case,
        maximum_input_tokens=maximum_input,maximum_token_json_bytes=max_token_json,
        raw_generation_case_bound=2000+6*observed_cases_each,
        six_generation_raw_upper_bytes=generation_bytes,other_reserve_bytes=other_bytes,
        total_reserved_bytes=total, checkpoint_bytes=0,
        reserve_model='input/tokenizer-bound conservative text/tokens+scalar framing; no permanent W/H/cache')
    write(output/'config.ready.json',c)
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--attempt',type=Path)
    parser.add_argument('--shared-commit')
    args = parser.parse_args()
    result = (bind_ready(args.prepared,args.output,args.attempt,args.shared_commit)
              if args.prepared else prepare(args.output))
    if args.prepared:
        print(json.dumps(dict(stage=result['stage'],job_ids=[],output=str(args.output),
            reserve_bytes=result['storage']['total_reserved_bytes'])))
        return
    print(json.dumps({key: result[key] for key in ('stage', 'job_ids', 'output',
        'generation_prompt_count', 'missing_generation_prompts')}, ensure_ascii=False))


if __name__ == '__main__': main()

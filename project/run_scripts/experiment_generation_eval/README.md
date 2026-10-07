# Baseline generation observation API

Owner: SH1. This observes already approved baseline model states; it does not fit, select, commit, or resume a model. Raw case IDs, prompts, text and tokens must remain ignored local.

```python
from project.run_scripts.experiment_generation_eval.assets import load_assets
from project.run_scripts.experiment_generation_eval.observer import GenerationObserver
from project.run_scripts.experiment_generation_eval.metrics import generation_payload

assets = load_assets(reference_manifest)
observer = GenerationObserver(model, tokenizer, assets, {
    'model_identity': sealed_model_runtime_tokenizer_identity,
    'generation_source_sha': sealed_generation_source_sha,
    'profile': 'cf-cake-prompt-inclusive-total100-eos-corrected-v1',
    'eval_seed': 20261007,
}, ignored_raw_directory, state_callback=nonmutation_signature)
# Each record retains CounterFact requested_rewrite and generation_prompts,
# with an explicit ordered occurrence_index (1..2000), not case-id sorting.
endpoint = observer.observe(records, 'W5', cohort='ALL_SEEN', state_identity=actual_weight_hashes)
current = observer.subset(endpoint, current_records, 'W5_CURRENT', cohort='CURRENT')
payload = generation_payload('current/post', current['summary'])
```

`observe` generates once per exact state/runtime/occurrence/profile; `subset` and `read_observed` verify raw identities and never invoke the model. A physical-state identity is JSON scalar/hash metadata, never tensor serialization. Supply a callback binding native H/context/cache as well as W. The callback and module parameter/buffer/hook/training signatures are checked; Python, NumPy, CPU and already initialized CUDA RNG states are restored even on failure.

The declared generator is unpadded single-row full-prefix **no-cache** reference, FP32/eager/autocast off/TF32 off, top-k 5, temperature 1, top-p 1, total prompt-plus-continuation limit 100 and native multi-EOS handling. No prompt truncation/overwrite. It is a repaired CAKE-derived profile, **not** upstream bitwise/paper-performance equivalence and not a qualified accelerated route. Prompts already at/over the limit retain their input and have zero continuation with an explicit reason.

Fluency is native NLTK token 2-/3-gram entropy, H2/3+2H3/3, bits; texts averaged per occurrence then macro over valid occurrences. Consistency is fixed vocabulary/IDF cosine of joined full generated text against all exact relation/target snippets. No TF-IDF fit or subject-only filter. Missing references/prompts/zero/nonfinite vectors are typed missing, not zero; genuine orthogonal cosine and empty n-gram entropy zero remain valid. Public `missing_<reason>_count` keys retain literal reason names, including `missing_missing_reference_count`. W&B payload contains scalar means/counts/reasons only; preserve local sums and actual denominator, separate from R/P/N harmonic.

Asset loader accepts a manifest path/dictionary or `{'generation_assets': manifest_member, 'asset_paths': {filename: local_exact_path}}`. Local overrides must preserve all SHA/size bindings. Scoring runtime is pinned by the manifest (NumPy/SciPy/scikit-learn/NLTK source and active English punkt resource). If a peer differs, use an explicitly isolated CPU scoring environment/adapter; do not upgrade scientific torch/transformers pins or silently substitute tokenizers/IDF.

Same-model W0 generation is fresh once in the first actual cold baseline, atomic shared READY then exact-state reuse/subsets. Prior R/P/N-only W0 is not generation evidence. Generation counts/timing/model forwards/token work and RNG/observer guard are returned separately; preserve returned phase work before proceeding. Source/helper CPU tests and remote W&B delivery are different evidence. No extra model forward for logging, no automatic scientific retry or monitor.

## GPT2 / GPT-J cache-repair API (2026-10-08)

This is a qualified transport/observation repair, not a native baseline method
change. The semantic profile, prompt seed identity, EOS/total100, CAKE metric,
native reference assets, and scoring denominators remain unchanged. Reference,
cached singleton, then equal-token-length KV batching are tested in that order
inside the first replacement allocation. A prelocked plan is not actual PASS.

```python
from project.run_scripts.experiment_generation_eval.kv_qualification import (
    build_qualification_plan, run_qualification, verify_actual_receipt)
from project.run_scripts.experiment_generation_eval.compatibility import (
    runtime_identity, verified_endpoint_row)

# Before submission: freeze this local plan/member/hash/tolerances. Prompts and
# selected local IDs are not W&B/Git payloads. No model forward here.
plan = build_qualification_plan(tokenizer, ordered_records,
    model_identity=sealed_model_identity, microbatch=8)
# First actual cold allocation: 0 fits/commits, one fixed <=8-prompt qualification.
actual = run_qualification(model, tokenizer, assets, plan,
    out=ignored_qualification_directory, state_callback=native_W_H_context_guard,
    source_identity=sealed_generator_source_identity)
generation_config.update(
    generation_route=actual['selected_route'],
    generation_microbatch=actual['fixed_microbatch'],
    qualification_plan_sha256=digest(plan),
    qualification_receipt_member=actual['member'])
observer = GenerationObserver(model, tokenizer, assets, generation_config,
    ignored_raw_directory, state_callback=native_W_H_context_guard,
    progress_callback=approved_scalar_transport)
observed = observer.observe(ordered_records, 'W0', cohort='FIRST2000',
    state_identity={'W': actual_cold_weight_hashes, 'H': {}})
```

`runtime_identity(config)` is a pure metadata builder in `compatibility.py` and
is also exported from `observer.py`. It preserves the old same-source no-cache
identity. Repaired identities explicitly bind the selected route, fixed MB,
source, and actual qualification receipt SHA. Production rejects a CPU-fixture
receipt; `qualification_allow_cpu_fixture=True` is only a local CPU test fixture.
Incomplete actual batch-width coverage is not production MB8 qualification:
the locked fallback is the qualified singleton, then no-cache reference.

### Exact partial old-W0 reuse

Set `generation_config['old_w0_reuse']` to a sealed local-only mapping:

```
observations_root: original atomic complete-case directory
observer_identity_member: original observer-identity.json SHA/bytes/path
config_member: original immutable run config SHA/bytes/path
runtime_member: original runtime receipt SHA/bytes/path (when available)
source_commit: original frozen scientific source commit
allowed_only_state_W: exact original/current cold W hash mapping
cold_observation_guard_member: source-backed original-phase evidence member
```

The guard must bind `source_commit`, `phase='W0_generation'`, `commits=0`,
`history_appends=0`, `model_W`, and `old_generation_runtime`. For a cancelled
partial old endpoint it explicitly says `whole_endpoint_guard_recorded=false`,
`proof_basis='FROZEN_SOURCE_CONTROL_FLOW_COLD_RUNTIME_AND_RPN'`, and
`partial_rows_authorized=true`. This is not a fabricated final RAM/RNG guard or
an old whole-W0 PASS. A new native qualification and completed new endpoint
nonmutation/RNG guard remain required.

`W0Compatibility` looks up only the expected original runtime/state/record key,
then checks exact bytes/payload/schema, ordered occurrence/case/prompt indices,
seed, complete EOS/length state, reference and scoring identity, and original
metric values/counts. Missing/unverified/incomplete rows are explicitly excluded
and only that missing work is generated. Original raw source/runtime/route/path
and bytes are never relabelled or copied. No edited model or RAM trajectory is
resumed. The mixed endpoint binds:

- `qualification_receipt_member` and identity qualification SHA;
- `compatibility_member` and identity compatibility SHA;
- per-row `provenance` with original raw member/runtime/source/route;
- exact ordered observation identities and `provenance_sha256`.

`read_observed` / `subset` retain strict checks for ordinary old same-runtime
endpoints and permit cross-runtime rows only through that explicit compatibility
manifest. `verified_endpoint_row(row, endpoint, expected_record_identity=...,
expected_state=...)` is the independent CPU collector primitive; it returns the
original raw document and imports no observer/model/torch. The collector still
independently checks native cohort/token/denominator/metric reductions. Final
shared READY must verify the actual qualification/compatibility member SHA and
**all planned 2000 complete cases**; a partial-progress file is never READY or a
complete `W0_first2000` score.

### Progress

The callback receives only approved `generation_progress/*` scalar counters,
rates and monotonic `generation_progress/step`, plus fixed public `phase`:
`W0_generation` or `generation_evaluation`. Start/final/error are forced boundary
events; intermediate events are throttled at existing complete case boundaries
to about 15 seconds or 64 prompts. There is no waiting timer, extra forward,
polling agent or heartbeat. This progress axis never substitutes for edits,
pre/post state edits or fit candidate axes. Counts distinguish new/reused cases
and logical generated tokens from actual physical forward/prefill/decode work.
No partial fluency/consistency mean is emitted. Callback execution is included
in RNG isolation and model/native-state guards; transport acceptance is not a
remote-delivery certification.

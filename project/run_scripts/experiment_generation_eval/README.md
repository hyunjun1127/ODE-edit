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

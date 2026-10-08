# Independent original CounterFact factual reference

This adapter is an observation/compatibility check, not a new scientific arm,
fit, performance-promotion gate, generation evaluator, or production factual
replacement. CPU fixture success cannot clear actual pretrained/GPU evidence.

## Source and independence

`cf_native_reference.load_original_counterfact()` verifies the complete original
`experiments/py/eval_utils_counterfact.py` from AlphaEdit commit
`b84624f44dfe8fc6cd9e41df916c44124a0c46dc`: 7,941 bytes, SHA-256
`25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145`.
The adopted file is `official/evaluation/reference/alphaedit_eval_utils_counterfact.py.txt`.
It compiles only the unchanged `test_batch_prediction` AST node. Its execution
globals are `typing`, `np`, `torch`, Python builtins, and the function itself.
The enclosing module is never imported: generation, sklearn, NLTK, dsets,
transformers, and util imports do not execute.

The original makes its own model forward, once per case, using its native
batched tokenizer, literal `.to("cuda")`, Llama target/prefix/logit BOS slices,
FP32 NumPy sequential accumulation/division, and desired-target strict argmax.
The adapter does not derive oracle results from canonical logits, token NLLs,
or production scoring functions. Canonical raw is exclusively the comparison
operand. The proxy only verifies actual native input tokens/masks, output
dtype/shape, input nonmutation, and physical work; it does not score logits.

## API and unchanged canonical reuse

`evaluate_native_counterfact(model, tokenizer, records, *, identity, ...)`
returns original grouped rows, independently aggregated summaries, original
display values, identity, and actual work. It is `OBSERVED_UNCOMPARED`, never a
parity PASS. `compare_native_counterfact(model, tokenizer, records,
canonical_result, *, identity, ...)` validates the supplied canonical raw before
any native forward, then measures the original function independently.

The `identity` argument must equal the canonical result's **existing** external
identity mapping exactly. No existing raw identity/hash is rewritten or
relabelled. Model, tokenizer and state locks may already be in that mapping
under `model_identity`, `tokenizer_identity`, `state_identity`. Alternatively,
supply those three keyword arguments separately; the new native receipt keeps
them under `reference_binding`, preserving the original external mapping.
The owner must certify that these locks describe that actual existing canonical
observation, not attach new state claims to unrelated/stale raw. Every actual
call also requires `state_callback()` to equal `state_identity` before and after
measurement. The supplied callback should cover the owner's weights/native
edit state/context identity, not just an endpoint label.

For example, an authorized owner can reuse its exact first-300 B3 raw:

```python
from official.evaluation.cf_native_reference import (
    compare_native_counterfact, MATCHED_SCOPE,
)

receipt = compare_native_counterfact(
    model, tok, first300_records, existing_B3_canonical_raw,
    identity=existing_B3_canonical_raw["identity"]["external_identity"],
    model_identity=locked_model_identity,
    tokenizer_identity=locked_tokenizer_identity,
    state_identity=frozen_B3_state_identity,
    state_callback=read_current_B3_state_identity,
    locked_cohort=locked_first2000_case_occurrences,
    evidence_scope=MATCHED_SCOPE,
)
```

`locked_cohort` is the immutable ordered list of
`{"case_id": ..., "occurrence_index": ...}` mappings, not a result-selected
sample. All actual scopes require exact prefix identity, ordered occurrences
starting at 1, unchanged token boundaries, correct target BOS behavior, native
right padding, FP32 CUDA parameters/buffers/logits, caller eval on every module,
caller `model.config.use_cache=False`, eager attention, TF32 off, and no autocast.
The adapter does not move/cast the model, change tokenizer settings, rewrite its
model name, or set config fields to satisfy these preconditions.

The existing factual `_observation` guard restores heterogeneous training flags
and Python/NumPy/Torch/already-initialized CUDA RNG, and detects parameter,
buffer, hook, and config mutations. Additional guards check record/tokenizer,
forward-input and external state identity. Unexpected model errors propagate
after restoration, without fallback, retries, fitting, or checkpoint rescue.

## Evidence scopes and fixed comparisons

- `ACTUAL_GPU_SMOKE` (default): preregistered locked first **four** cases, at
  existing authorized W0/native qualification state. This is engineering
  admission only. No edit/fitting is required or performed by the oracle.
- `ACTUAL_MATCHED_SUBSET`: the caller's exact locked prefix, such as first300,
  compared at the same actual frozen B3 state as existing canonical raw. This
  separate receipt does not claim the full2k requirement or relabel its scope as
  smoke. A successful actual matched300 comparison can satisfy the same
  engineering admission as the minimal first4 smoke on that locked model/state;
  no redundant first4 forward is required by this adapter.
- `ACTUAL_FULL_2K`: exact locked first2000, actual Llama AlphaEdit endpoint
  weights, with `method="ALPHAEDIT"` or `method="AlphaEdit"` in the external mapping or separate
  `model_identity` mapping. This alone can supply the separate full2k evidence.
- `TEST_ONLY_CPU_FIXTURE`: both this scope and `test_only_cpu=True` are required.
  A tokenizer-batch transport bridge services the original literal
  `.to("cuda")` on CPU without modifying its AST. Result `CPU_FIXTURE_PASS`
  leaves every actual scope `NOT_OBSERVED`.

Each receipt qualifies only its own scope; absent scopes remain `NOT_OBSERVED`.
Full2k remains unobserved until actual full-cohort original/canonical comparison,
regardless of smoke or fixture success. No actual GPU comparison was performed
to implement or test this adapter.

Tolerances are fixed before actual outputs, not adjustable API arguments:

- Per candidate mean NLL in nats:
  `abs(native-canonical) <= 1e-4 + 1e-5*abs(native)`.
- Desired strict argmax correctness and each strict NLL-preference bit must
  agree **exactly**. Ties fail success; a near-tie bit flip fails even when
  both NLL differences fit the numerical tolerance.
- Request-macro Efficacy/Generalization/Specificity and raw harmonic Score:
  absolute arithmetic tolerance `1e-10` percentage points, not paper 1pp.
- Original display is assessed separately: original NumPy request-level mean
  and std, `np.around(... * 100, 2)`, then SciPy `hmean` of rounded success means.
  Rounded success/strict-accuracy rows must match exactly; displayed Score uses
  `1e-10` arithmetic tolerance. Raw and displayed Scores are never conflated.

Results are `PASS` (actual compared scope only), `CPU_FIXTURE_PASS`, `MISMATCH`,
or typed `NOT_QUALIFIED`. Mismatch rows distinguish NLL, strict desired argmax,
strict NLL preference, request macro, and separately original display failures.
No paper/generation tolerance is borrowed or relaxed after seeing outputs.

All raw cases, tokens, identities, mismatches, and source/state/cohort receipts
remain local-only. `work` reports actual independent forward calls, candidate
sequences, physical/padded input tokens, target tokens and elapsed time. Canonical
forwards are not silently repeated. Compact scalar-only owner reporting remains
a separate policy obligation; do not upload this raw receipt to Git or W&B.

## CPU regression command

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B -m unittest \
official.tests.test_cf_native_reference -v
```

The deterministic fixtures exercise GPT-J no-BOS, genuine Llama paired BOS
slicing, multi-token targets/padding, independent calls, request macros, strict
ties/near ties, mutation/error restoration, source tampering, row/identity
mismatches, exact existing raw reuse, and CPU-versus-actual scope separation.

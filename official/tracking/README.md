# Official scalar W&B transport

This is the **one common `official` transport distribution** for the new official
baseline runners. Server runners import it read-only; they do not copy or fork
the logger. Its seven core modules are adopted from the pinned existing
`project/run_scripts/experiment_tracking` implementation. The old project helper
and already frozen/running sources are unchanged. No `project`, `scripts`, or
external EasyEdit algorithm import is required.

The schema/API/fake SDK tests are CPU evidence only. They do not establish actual
W&B login/project access, remote delivery, native model parity, GPU qualification,
or scientific completion. Production runners still own those distinct receipts.

## API and local setup

```python
from official.tracking import init, LoggingBlocked

tracker = init(env_file=ignored_env_file, spool=new_unique_ignored_spool,
               config=bound_scalar_config)
accepted = tracker.log(measured_scalar_mapping)  # step=None is recommended
result = tracker.finish(exit_code=0)             # bounded finish/readback
```

`init` is keyword-only, creates a new spool (never reuses an old directory), and
starts the SDK in a separate CPU-only Python process. `Tracker` also supports a
context manager; telemetry shutdown never suppresses an original scientific
exception. Startup timeout defaults to 50 seconds; finish timeout to 45 seconds.
`LoggingBlocked` means startup/login/remote identity is not ready. A successful
`log` means the local queue accepted scalar data, **not a remote acknowledgement**.
Check `identity.json`, `receipt.json`, and the bounded startup/finish readback.
`LOGGING_ACCEPTED` is explicitly `SDK_ASYNC_NOT_REMOTE_ACK`; incomplete remote
history is `UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE`, never an invented PASS.

The ignored nonsecret environment file uses the existing `load_env` contract:
`WANDB_ENTITY=wkdguswns2256`, `WANDB_PROJECT='layer allocation'`, online mode,
console off, save-code false, and `ODEEDIT_WANDB_PYTHON` pointing to the isolated
SDK Python with `wandb==0.30.0`. `WANDB_BASE_URL` is an already verified HTTPS API
endpoint (default `https://api.wandb.ai`), **not** a URL derived from Forge's UI.
Login credentials remain in the SDK's secure existing auth locations/environment;
never put them in config, command arguments, Git, or telemetry fields.

## Official config contract

Every official scientific config contains these built-in scalar strings:

| Field | Required value or meaning |
| --- | --- |
| `server`, `task_id`, `arm`, `attempt` | Exact own server/task/cell/new attempt identity |
| `source_sha`, `config_sha` | Actual bound 40/64-character lowercase hex SHA |
| `model` | `llama3`, `gptj`, or `qwen25` |
| `model_family`, `writer`, `baseline`, `role` | Truthful native identity; `role=scientific` |
| `metric_schema` | `official-baselines-scalar-v1` |
| `instruction_id` | `USER-OFFICIAL-BASELINES-20261008-R1` |
| `dataset` | `cf` or `zsre` |

CF configs additionally require `generation_metric_schema` =
`counterfact-cake-generation-metrics-v1`, `generation_profile` =
`cf-cake-native-casebatch-kv-total100-globalrng-v1`, `generation_eval_seed` =
20261007, actual `reference_assets_sha256` and `generation_source_sha`,
`generation_repair_instruction` equal to the official instruction above, and
`generation_schedule=W0_AND_W20_FIRST2000`. Wrong authority/profile/schedule
pairs fail closed. **zsRE omits all CF generation config and metrics**; it does
not invent Fluency or Consistency values.

Only the four allowlisted parent Slurm variables are inspected: `SLURM_JOB_ID`,
`SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`, and `SLURM_STEP_ID`. Raw job/array/step
identity is recorded in config; array index `0` and signed step IDs are valid.
`run.name` ends in `job<displayID>`. Caller/env mismatches fail. Local CPU
execution records `execution_backend=local`, `identity_source=NOT_APPLICABLE`,
and no fake job ID. Each new init creates an immutable UUID/run URL/config/job
receipt, never resumes, renames, or backfills an old W&B run.

## Metrics and denominators

### zsRE dedicated views and mapping API

For new zsRE callers use the dataset-specific namespace
`zsre/{current/pre,current/post,all_seen/post,W0_first2000}/`
with `Efficacy`, `Generalization`, `Specificity`, `Specificity_loc_ans`, `W0_prediction_agreement`,
`Score`, and `requests`. These keys require the official schema and
`dataset=zsre`; CF and legacy configurations reject them. Existing `official/*`
callers remain valid. If both aliases are logged in a row their values must agree.

```python
from official.tracking import official_zsre_metrics

payload = official_zsre_metrics(
    evaluation['summary'], config_values=bound_config,
    endpoint='all_seen/post', edits=2000, post_state_edits=2000)
tracker.log(payload)
```

The input is the measured zsRE reducer summary (not the full raw evaluation).
No model forward, tensor conversion, prompt-pair relabeling or data upload occurs.
Efficacy/Generalization are teacher-forced target token accuracy averaged within
each request, then across requests. **Specificity is neighborhood loc_ans target
accuracy**, also exposed as the compatibility alias `Specificity_loc_ans`.
`W0_prediction_agreement` is a separate auxiliary, not paper Loc. All scores are
percentages. `Score` is the harmonic mean of Efficacy/Generalization/Specificity
(answer accuracy, never W0 agreement); it is derived only when
all three measured components exist. Missing/None fields are omitted, not zero.
Existing denominator/state rules are unchanged: current has 100 requests;
all_seen at W5/10/15/20 has `requests=edits`; W0_first2000 has edits0/requests2000.
Supply both pre/post state axes for current/pre. Evaluation uses `edits`, fit
uses the separate monotone candidate axis. CF generation fields are forbidden.

This supersedes pre-correction W0-agreement Specificity under user instruction
`USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1`. Frozen old runs are not renamed
or hotpatched: re-reduce saved token correctness on CPU, retaining before/after
values and raw/source provenance. The new mapping requires Specificity and
Specificity_loc_ans to agree when both are supplied. Source/config SHA separate
future runs; no historical W&B backfill is implied.

SH2 owns saved-view publication: an index filtered to `config.dataset=zsre`
and model views additionally filtered to `config.model=llama3`, `qwen25`, or
`gptj`. Plot `zsre/*` against `edits`, and keep current and all_seen panels
separate. Code/API availability does not prove that these views or online
history exist. Do not rename, backfill or hotpatch old runs. Report view URLs
and actual new-run startup/readback separately.

Official paper scores are distinct from legacy prompt-pair diagnostics:

```text
official/{current/pre,current/post,all_seen/post,W0_first2000}/
    Efficacy, Generalization, Specificity, requests
    Score, Score_AlphaEdit_display              # CF, if actually measured
    Specificity_loc_ans                        # zsRE answer-accuracy alias
    W0_prediction_agreement                    # zsRE auxiliary, if measured
```

These scores come from `official.evaluation.reduce`: CF first averages prompt
preference success **within each request**, then averages requests; zsRE uses
request-macro teacher-forced token correctness and separate W0 agreement versus
`loc_ans` accuracy. All score values are percent 0..100. CF `Score` is the
harmonic mean of the unrounded E/G/S; `Score_AlphaEdit_display` uses rounded
components solely for the historical display definition. The transport validates
arithmetic, but does not compute, replace, or reinterpret the evaluator.

Legacy `current/pre`, `current/post`, `all_seen/post`, `W0_first2000`,
`w0/current/N`, and `w0/all_seen/N` R/P/N nine-field diagnostics remain supported
when they were actually measured with their documented semantics. They are **not
aliases of the official request-macro E/G/S**. Preserve real prompt-pair/token
counts and NLL units; do not synthesize legacy success counts from a macro score.
Official W0 does not inherit PRICE's hardcoded P4000/N20000 counts: its variable
native CF prompt counts and zsRE token counts must come from the bound actual
observation. The caller verifies raw/order/token/evaluator/source identity.
The helper deliberately does not request extra forwards to fill missing fields.

All evaluation mappings include `edits`. `current/post` always means the current
100-request batch, including W20; official `requests` therefore stays 100.
`all_seen/post` is emitted only at measured 500/1000/1500/2000 endpoints, with
`requests=edits`; official W0 is `edits=0`, `requests=2000`. Actual state is
separate in `pre_state_edits`/`post_state_edits`. Native fit metrics use monotone
`fit/global_candidate` plus `batch` and `candidate`; no batch-reset fit axis.

## CF generation endpoints only

### Deferred FLU/CON evaluation from the final 2K checkpoint

An official CF editing run may explicitly set
`generation_schedule="DEFERRED_CHECKPOINT_EVALUATION"`. Keep the official
instruction, dataset, method, source and config identity; omit generation profile,
metric schema, seed, reference/source SHA, repair instruction and qualification
metadata. The existing Server2 checkpoint-only caller already emits this config.
The common transport accepts it for all official models and servers. Missing or
unknown schedules still fail validation; zsRE and legacy schemas cannot opt in.

This run logs measured factual E/G/L/Score and fit metrics only. FLU/CON scores,
counts, generation progress and generation phases are rejected, including zero
placeholders. W&B records the deferred schedule; successful factual readback
does not imply generation evaluation or scientific completion.

The runner must skip W0/W20 generation, retain the final W20 (2,000-edit)
checkpoint and its model/source/config/sample/RNG identity, and keep the future
evaluation consumer pending. Do not delete the checkpoint before that consumer
finishes. Later FLU/CON measurement must use a separate evaluation run with its
actual generation configuration and checkpoint provenance. Do not relabel the
editing run as generation-enabled or fabricate W0 observations. Reference assets
and metric definitions stay unchanged. These are caller obligations; the W&B
schema does not save checkpoints or launch the deferred evaluator.

The following endpoint rules apply when generation is enabled.

Use `W0_first2000/generation/*`, `W0_first2000/fluency/ngram_entropy` and
`W0_first2000/consistency/reference_score` at `edits=0`. Use the analogous
`all_seen/post/*` keys only at `edits=2000`, `post_state_edits=2000`. Each complete
endpoint requires `generation/planned_count=2000`, actual valid Fluency/
Consistency counts, and only measured finite means (omit means when valid count
is zero). Entropy is raw bits; Consistency is raw TF-IDF cosine, not percent.
Do not relabel full-2k generation as `current/post` or emit generation at W5/10/15.
W0 generation is computed once per model and shared only with exact identity;
W20 is computed once per CF chain. Scheduling and reuse remain runner-owned.

Partial progress uses `phase=W0_generation` or `W20_generation` and the separate
monotone `generation_progress/step` axis. It never mixes `edits`, endpoint scores,
or fit fields into a progress row. Thus a partial W0 is not a completed W0 score.

The existing `NativeGenerationObserver` names W20 progress
`generation_evaluation`. Its raw callback is **not** accepted unchanged by the
official schema. Use the shared, nonmutating observational adapter:

```python
from official.tracking import official_generation_progress

def progress_callback(raw_progress):
    return tracker.log(official_generation_progress(raw_progress, endpoint="W20"))
```

For W0, bind `endpoint="W0"`; do not use this adapter for intermediate endpoints.
It preserves every count/axis scalar, renames only W20's progress phase, and does
not change generation/model state, sampling, scores, or endpoint identity.

Scalar validation permits built-in finite numbers only; it never calls tensor
`.item()`, `.cpu()`, or `float(tensor)`. No raw text, prompts, tokens, arrays,
weights, code, console logs, full environment, machine statistics, or artifacts
are uploaded by this API. File/code/metadata/stat collection is disabled in SDK
settings. Accepted scalar journal and opaque transport receipt remain local.

## Compatibility and CPU checks

`price-first2k-scalar-v1` remains an optional compatibility schema with its old
strict W0 counts and W20-only native authority pair. It is not the official
runner schema. Official instruction + old W20-only schedule, or zsRE + CF
generation metadata, is explicitly rejected. Existing helper callers do not
change merely because this distribution is published.

```bash
python3 -m unittest official.tracking.test_transport -v
```

The 28 fake SDK tests cover schema/schedule authority, variable native W0 counts,
CF/zsRE definitions, separate endpoint/fit/progress axes, array0/signedstep,
immutable identity, sanitized isolated worker, and truthful bounded readback.
No SDK import/login/upload, actual-model forward, or Slurm action is used.
# GPT-J W0 generation-only (2026-10-11)

`USER-GH-GPTJ-W0-FLUCON-SERVER4-20261011-R1`에 한해 server4/gptj/CF/role=eval_only,
`evaluation_profile=cf-native-generation-W0-only-v1`, `generation_schedule=W0_ONLY_FIRST2000`를 사용한다.
`instruction_id`와 `generation_repair_instruction`은 위 nonce다. 공통 native profile/seed20261007/
reference SHA는 그대로이며 저자 baseline fit이나 W20 checkpoint 복원은 하지 않는다.
필수 provenance는 `base_model_sha256`(봉인 base weight manifest의 SHA), `evaluator_sha256`,
`stream_sha256`, `tokenizer_sha256`; edited `checkpoint_sha256`은 금지한다.
runner는 manifest의 실제 member SHA/revision을 검증한다. schema의 SHA 문법 검산은 자산 검증을 대신하지 않는다.
progress는 `official_generation_progress(..., endpoint='W0')`, 최종 지표는
`W0_first2000/{generation,fluency,consistency}/*`, edits=0이다. factual/fit/zsRE/W20 키는 거절한다.
원 bits/cosine과 기존 ×100 표 표시를 분리한다. 다른 authority/기존 frozen 실행은 변경하지 않는다.

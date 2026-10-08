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

Official paper scores are distinct from legacy prompt-pair diagnostics:

```text
official/{current/pre,current/post,all_seen/post,W0_first2000}/
    Efficacy, Generalization, Specificity, requests
    Score, Score_AlphaEdit_display              # CF, if actually measured
    Specificity_loc_ans                        # zsRE, if actually measured
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

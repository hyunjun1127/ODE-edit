# Qwen current-entry FE rescue: L2=100

Authority: `USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1`, latest explicit early-stop/rescue steering.
This is **author-hparam-tuned FE**, not author default or a working-result claim.
Old `ac05e848` four-arm selection and `10843f86` final frozen attempts are untouched.
Prior `janghj-feq-l210-20261012-r2` UID `c3ffd8c8-f465-468c-8da4-64121d4d8d02`
is recorded as `SUPERSEDED_USER_EARLY_STOP`, not W20/numerical failure. The other
old declared arms are not prerequisites and this entrypoint does not submit them.

## Root-only API

Use the same host bindings JSON as `prepare.py` on rent (model/C0/streams/runtime,
author clone, host checkpoint lock, Python, reserve >=32GiB and storage counts).
Use a **new** output root. No source file in an old attempt is modified.

```sh
python -m official.diagnostics.fe_qwen_current.rescue prepare NEW_ROOT BINDINGS.json
PYTHONPATH=NEW_ROOT/source python -m official.diagnostics.fe_qwen_current.rescue launch NEW_ROOT/configs/L2-100.json NEW_ROOT/lock.json
python -m official.diagnostics.fe_qwen_current.rescue collect NEW_ROOT
```

Preparation creates exactly one L2=100 CF selection config, no GPU/submission.
Root still performs actual rent admission/execution with total own GPU<=2;
SH1/SH2 submit0/HOLD. Native author commit478134df, BF16/default backend,
clamp1/steps35/lr.5/decay.001/KL.0625/C0=15000, layers4–8, history and writer
are unchanged. The writer AST is checked against the pinned author source.
Paper current-entry L4 fit and same-entry later-layer firstforward are retained.
No W0-target fallback or added normalization/solve/jitter is present.

The new immutable selection lock uses `protocol=rescue-L2-100-v1` and its own
SPEC hash. Existing preparation/launch functions are reused through scoped
process-local bindings; invoking the old entrypoint retains the old contract.
The generated new native main imports `rescue_observer`, never the old frozen
observer in place. Storage remains latest-only every500, shared host lock,
one atomic tmp; existing checkpoint/raw files count toward admission and remain.
Conservative existing preparation requires planned_latest_count>=4 (>=6 for
final); root binds the actual admitted slots/reserve, with no deletion assumed.

## Quality stop, evidence and diagnostics

After committing current100 (and scheduled all-seen/checkpoint if milestone),
R<80 OR P<70 OR N<65 writes `QUALITY_EARLY_STOP.json`, closes tracking exit2 and
raises `QualityEarlyStop`. No fake numerical failure/COMPLETE is emitted. The
last scheduled CP and all raw/commits remain. Equality to a threshold continues.
This gate is for rescue held-out selection, not benchmark-specific retuning.

Existing tensors yield K/RHS Frobenius, raw delta and BF16-cast delta norm,
pre/post actual W norm, L4/L5 entry target column norms/top10 local indices.
The BF16-cast-delta diagnostic is **not** claimed to be the author's addition
order (author adds native high-precision delta then writes into BF16 W).
The actual solve matrix/RHS/solution are observed while alive: relative residual
uses four fixed RHS columns, row chunks128. It is explicitly a sampled residual,
not a full residual or condition estimate. No additional solve/eigenscan/LM
forward/RNG draw or full H clone. Norm conversion is chunked even for transposes.
Per-case init/delta/clamp/lookup/probability remains existing native stdout;
no extra fit/replay is introduced to obtain it.

Collector requires actual W20/2000, 20 commits plus W0, every current100 and
all-seen500/1000/1500/2000 retained observation/query/summary, exact CP full SHA
and latest pointer. It independently re-reduces request-macro metrics and
rejects a missed quality stop. Only R>=95/P>=85/N>=70 yields SELECTED; no best
failed promotion. Missing/quality-stopped/numeric/technical runs cannot select.

## Conditional final CF and zsRE

After `selection-result.json` is actually SELECTED, use the same final bindings
as previously, `stage=final_benchmark`, `selected_result` exact member and original
zsRE stream. Invoke **rescue prepare/launch**, not the old module entrypoint.
This re-audits the rescue proof and creates exactly cold CF and zsRE L2=100;
no retuning on benchmark. Public-query zsRE and CF evaluators remain unchanged.

```sh
python -m official.diagnostics.fe_qwen_current.rescue prepare FINAL_ROOT FINAL_BINDINGS.json
# root launches each config from FINAL_ROOT/source using rescue launch
python -m official.diagnostics.fe_qwen_current.rescue final-collect FINAL_ROOT
```

Groups are `fe-qwen-current-z-rescue-L2-100-20261012` and
`fe-qwen-current-z-rescue-selected-final-20261012`, separate from historical
selection, main baselines and generation. Standard current/all-seen metric
names/axes, public dataset keys and scalar privacy remain. Generation is
DEFERRED; the unfinished generation adapter is not part of this release.

CPU tests include executing the exact author writer with/without diagnostics:
BF16 selected W, FP32 H and RNG are bitwise equal, sampled residual scalars agree;
this is not a pretrained/GPU, runtime-memory or recovery qualification.

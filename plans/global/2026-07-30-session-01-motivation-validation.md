# Session 01 — ODE-Edit Motivation Validation canonical plan

- 작성/개정일: 2026-07-30
- 소유자: `head-server1-gh` (`global-head`)
- 현재 상태: `mv0_llama_one_shot_preflight`
- 현재 claim: `hypothesis only`
- canonical audit:
  `audits/global/2026-07-30-proposal-easyedit-baseline-audit.md`
- canonical rationale:
  `project/proposals/sections/01-motivation-validation.md`
- raw artifact root:
  `local/results/raw/session01_motivation/`
- compact run metadata:
  `runs/<run_id>/`

## 1. 네 범주

### Proposal에서 온 내용

- fixed direct-z guide 아래 여러 layer low-rank proposal을 local actuator로 보고,
  same-snapshot utility와 editor-native geometry로 write를 재배분할 수 있다는
  가설
- utility heterogeneity, partial update 뒤 non-stationarity, sequential load
  concentration을 먼저 검증하라는 진단 방향
- paraphrase, neighborhood/locality, downstream prompt를 edit-time에서 금지하는
  information firewall

### Repo/protocol에서 확인한 사실

- `PROTOCOL.md`가 canonical 운영 source다.
- remote, server1/server4 record, private resource cap, runtime path가 등록돼 있다.
- 별도 SH Codex session과 completed ODE-Edit run은 아직 없다.
- local EasyEdit는 commit `3488a66`의 dirty worktree이며, core file과 hparams
  hash를 run별로 고정해야 한다.
- 두 고정 model snapshot, CounterFact full file, 두 모델의 precomputed
  Wikipedia covariance와 AlphaEdit projector가 local에 존재한다.

### GH 추정

- 올바른 대비는 “open-loop MEMIT vs feedback”이 아니라
  `Gauss–Seidel-style one-pass construction` 대
  `same-snapshot comparison + all-layer revisit`이다.
- same-snapshot utility CV, rank turnover, Gini만으로는 dynamic routing을
  정당화할 수 없다.
- ODE라는 이름은 step refinement와 refreshed-direction benefit이 확인될 때만
  유지할 수 있다.

### 사용자 확인 필요

- 별도 server-head Codex session ID 등록. 다만 사용자가 지금 Motivation
  kill-test 실행을 명시적으로 지시했으므로, SH가 생기기 전 server1의 최소
  Slurm 제출이 필요하면 GH exception audit에 사유·명령·범위·후속 보고를
  기록한다.

## 2. 고정 연구 범위

### Model

| alias | exact local revision |
| --- | --- |
| `llama3-8b-inst` | `meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| `qwen2.5-7b-inst` | `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28` |

다른 backbone은 canonical Motivation result에 포함하지 않는다.

### Dataset

- source:
  `/mnt/raid5/janghj/EasyEdit/data/counterfact/counterfact.json`
- row count: `21,919`
- SHA256:
  `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`
- selection은 evaluation field나 outcome을 보지 않고 `case_id`의 deterministic
  hash만으로 정한다.
- Git manifest에는 case ID, split/order hash와 source hash만 남긴다.

### Editor

- primary: local EasyEdit MEMIT internal function을 read-only import
- AlphaEdit: MEMIT Motivation 생존 뒤 별도 cache/projector audit 후에만 실행
- EasyEdit source file은 수정, patch, checkout, restore하지 않는다.
- existing EasyEdit example shell과 `BaseEditor.batch_edit`는 사용하지 않는다.

### Precomputed artifact

- covariance:
  `/mnt/raid5/janghj/EasyEdit/examples/data/stats/`
- projector:
  `/mnt/raid5/janghj/EasyEdit/examples/null_space_project_*.pt`
- expected file size와 SHA256을 config에 고정한다.
- missing/mismatch일 때 download/recompute하지 않고 fail-closed한다.
- `layer_stats()` 계산 path와 AlphaEdit `get_project()` 계산 path를 runtime
  guard로 막는다.

## 3. 실행 전 공통 계약

### 3.1 State-edit event

분석 단위는:

```text
(model, order, stream_step, case_id, pre_edit_state_hash)
```

다. 같은 event의 paired arm은 byte-identical pre-state, sanitized request,
frozen contexts, direct-z, covariance, dtype, RNG state를 공유한다. 각 arm은
동일 state에서 독립 branch하고 exact rollback/hash gate를 통과해야 한다.

### 3.2 Information firewall

Edit process가 받을 수 있는 field:

```text
case_id, prompt, subject, target_new
```

와 frozen six allowed rewrite contexts, current model state, covariance뿐이다.

금지:

- paraphrase, neighborhood/locality, generation prompt
- ground-truth locality label
- downstream prompt/label/metric object
- evaluation result를 다음 edit의 layer/step/stopping에 feedback

Evaluator는 별도 process와 artifact namespace를 사용한다. forbidden field가
edit artifact/runtime trace에서 한 건이라도 발견되면 해당 run 전체를
`block`한다.

### 3.3 Context와 direct-z

- base `"{}"` + generated five context 문자열을 model별 fresh process에서
  seed와 함께 freeze하고 SHA256을 기록한다.
- direct-z target construction의 KL row `"{} is a"`는 base-editor 내부
  정보로 별도 기록한다.
- main diagnostic key는 EasyEdit `compute_ks()` aggregation을 유지한다.
- main residual은 canonical MEMIT처럼 raw rewrite prompt에서 측정한다.
- direct-z는 event 시작 state에서 한 번 계산하여 paired branches에서 고정한다.
- cache identity는 model revision, order, step, case ID, state/context/hparams
  hash를 포함한다. 안전한 identity를 구현하지 못하면 cache를 끈다.

### 3.4 Denominator

모든 사전 선택 event는 `intention-to-diagnose`에 포함한다. 다음을 사후
제외하지 않는다.

- direct-z/solve/non-finite failure
- pre-satisfied request
- all non-positive utility
- rejected step
- matched-progress target 미도달
- rollback failure

Actionable denominator는 결과를 보기 전에:

1. pre-edit deficit
   `phi_stop = max(max_non_target_logit - target_logit)`가 양수
   (`exact_top1_stop` helper의 `target-best_other` margin으로는 음수)
2. 모든 지정 layer에서 finite factor 생성
3. paired arm이 동일 direct-z 사용
4. MV-0 fidelity pass

로만 정의한다.

### 3.5 Metric

- exact stop:
  `max_{context,token}(max_non_target_logit - target_logit)`
- differentiable utility:
  별도 normalized smooth surrogate와 signed derivative
- realized progress:
  independent finite-step branch의 exact stop/surrogate 변화
- displacement:
  `C`-weighted state displacement와 non-negative path expenditure를 분리
- raw `CV`, Spearman, Jaccard, Gini는 descriptive only

## 4. Reusable implementation contract

Tracked code는 `project/run_scripts/ode_edit_motivation/`에 둔다.

전체 Motivation ladder에서 순차적으로 갖춰야 할 module:

- immutable request/provenance/context contracts
- EasyEdit read-only import/hash bridge
- native MEMIT canonical-mode adapter
- same-snapshot low-rank factor proposal
- exact stop margin와 differentiable utility
- C-inner-product/norm와 factor-only telemetry
- temporary apply/rollback/hash guard
- MV-0~MV-4 runner mode
- deterministic selection/order manifest
- JSONL raw trace + compact summary
- Slurm render-only envelope

향후 experiment는 같은 request, provenance, factor, metric, rollback, artifact
schema를 import하고 stage-specific controller만 추가한다. 기존 module을 복사해
variant를 만들지 않는다.

현재 구현 완료 범위는 MV-0 fidelity runner와 그 공통 contract/bridge/hook다.
MV-1~4 controller는 각 앞 단계가 생존한 뒤 같은 package에 추가하며, 아직
구현됐다고 간주하지 않는다.

CPU unit test와 no-model preflight가 pass하지 않으면 GPU job을 render하지 않는다.

## 5. Staged kill-test

### MV-0 — Native fidelity and neutrality

#### 실행

- model별 deterministic 3 edit
- base state에서 native singleton `execute_memit` 대 adapter canonical mode
- trace-off 대 trace-on-no-write
- exact solve만 primary
- early/middle sequential sentinel은 MV-4 진입 전에 추가

#### 기록

- layer factor/update C-cosine와 relative C-norm error
- materialized parameter delta relative/max error
- final update cosine/norm error
- allowed-context logits와 rewrite progress 차이
- trace-only 전후 weight/output hash

#### 판정

Self-replay와 dtype-aware numerical floor로 calibration equivalence bound를 먼저
정한다. 양 모델에서 paired equivalence CI가 bound 안에 있어야 pass한다.
한 모델이라도 fail이면 implementation을 고치고 MV-0부터 다시 한다.

#### 독립 분석

run 종료 뒤 실행 담당이 아닌 별도 agent가 raw manifest와 compact summary만
읽고:

`experiment-reports/global/<date>-mv0-fidelity-analysis.md`

를 작성한다.

### MV-1 — Calibrated predictive heterogeneity

#### 진입 전 필수 보강

- model/revision/stats/hparams identity가 없는 EasyEdit `COV_CACHE`를 arm 사이에
  공유하지 않고, pinned file을 process-local로 다시 load한다.
- editable weight뿐 아니라 `requires_grad`, model mode, `use_cache`, context
  cache, RNG 등 non-weight state neutrality를 sentinel로 검증한다.
- generated context cache는 model별 fresh process에서 만들고 paired
  order/arm의 context hash가 byte-identical해야 한다.
- multi-token target의 encode→continuation→tokenize round trip, right padding,
  causal target position, logits tensor layout을 고정 sentinel로 검증한다.

#### 실행

- 사전 선택 100 case를 `calibration 20 / confirmatory 60 / untouched 20`으로
  deterministic split
- calibration에서 finite-step C-budget, replay noise, near-tie equivalence
  envelope 고정
- confirmatory에서 same-snapshot signed utility와 layer별 actual finite-step
  progress 측정
- scale-renormalization, replay, layer-label permutation,
  leave-one-allowed-context-out control

#### Primary event effect

```text
realized_progress(analytic-utility allocation)
  - realized_progress(calibration-fixed static-best)
```

calibration에서 고정한 operational `C`-budget에서 비교하며, winner-vs-median과
analytic utility–realized ordering concordance는 secondary로 둔다. layer를
독립 sample로 세지 않는다. `C`-specific claim에는 norm-min allocation과
cost-shuffled allocation을 negative control로 포함한다.

#### Kill

두 모델 모두에서 winner advantage가 replay/permutation/context null envelope와
구분되지 않거나 utility가 realized progress를 예측하지 못하면 layer-routing
motivation을 kill한다.

단, 이 문장의 “구분되지 않음”은 단일 secondary metric, 한 case, 또는 CI
endpoint 하나의 실패를 뜻하지 않는다. 최신 사용자 지시에 따라
`plans/global/2026-07-30-session01-mv1-implementation-spec.md` 10.6절의
핵심신호 우선 gate를 적용한다. Technical validity는 엄격히 block하되,
scientific kill은 두 model의 primary mean·robust mean·sign과 oracle
opportunity가 함께 null인 경우로 제한한다. 한 model의 강한 신호와 다른
model의 비음수 신호는 architecture-conditional bounded continuation으로
분리하고, 사후 threshold/controller retuning으로 rescue하지 않는다.

#### 기대효과 예측

confirmatory/untouched event에서 다음을 model별 paired CI로 분리한다.

1. `oracle allocation - best static`: routing이 가질 수 있는 empirical upper
   bound
2. `cross-validated controller - best static`: tuning leakage 없이 기대할 수
   있는 achievable gain
3. `refreshed controller - fixed-direction controller`: dynamic refresh만의
   incremental gain
4. 같은 rewrite progress에서의 displacement frontier 차이: retention
   개선으로 연결되기 전의 mechanism-side 예상효과

MV-4에서 displacement–retention intervention slope가 생존한 경우에만 4를
retention 기대효과로 변환한다. 그 전에는 성능 개선 예측이 아니라
`matched-progress displacement reduction forecast`로 보고한다. point estimate,
paired bootstrap CI, model별 sign, compute/NFE 비용을 함께 공개하며 oracle을
실현 가능한 method 성능처럼 쓰지 않는다.

#### Pivot

layer effect는 재현되나 winner가 state와 무관하게 고정이면 static layer
allocation으로 pivot한다.

#### 독립 분석

별도 agent가 MV-1 artifact/report만 읽고 독립 보고서를 작성한다.

### MV-2 — Actionable non-stationarity

#### Paired branches

1. replay/no-op
2. `h=0` sham
3. preregistered partial joint step
4. refreshed direction + refreshed coefficient
5. fixed initial direction + refreshed coefficient
6. fixed direction + fixed coefficient

Primary는 fixed direct-z다. post-step direct-z recomputation은 sensitivity다.

#### Primary effects

```text
Delta_direction_refresh
  = progress(refreshed direction/coefficient)
    - progress(fixed direction/refreshed coefficient)

Delta_coefficient_refresh
  = progress(fixed direction/refreshed coefficient)
    - progress(fixed direction/fixed coefficient)
```

#### Kill/pivot

- direction refresh effect가 replay/near-tie noise와 동등:
  ODE/relinearization kill
- coefficient refresh만 positive:
  fixed-direction dynamic coefficient controller로 pivot
- 둘 다 없음:
  static routing으로 pivot

Spearman/top-k change는 primary가 아니다.

#### 독립 분석

별도 agent가 MV-2 artifact/report만 읽고 독립 보고서를 작성한다.

### MV-3 — Matched-progress reroutability and interaction

#### Arms

- canonical MEMIT
- global alpha first-hit
- C-normalized uniform allocation
- calibration에서 고정한 static layer allocation
- best single-layer/static selection
- minimum-predicted-displacement allocation
- fixed-direction dynamic coefficient
- refreshed joint allocation
- Gauss–Seidel small-step

#### Primary

evaluation prompt 없이 allowed rewrite metric으로만 공통 progress support를
정하고, 그 구간의 displacement–progress frontier AUC를 paired 비교한다.

추가로:

- layer-cap compensation frontier
- single-layer effect 합과 joint effect의 additivity error
- fixed proposal simultaneous 대 commit-order invariance
- context rotation에서 held-allowed-context progress

를 측정한다.

#### Kill/pivot

- global scaling/early stop과 frontier가 동등: rerouting kill
- rerouting gain은 있으나 refresh gain 없음: static capacity routing
- refresh gain은 있으나 displacement gain 없음: capacity-free dynamic scheduler
- joint prediction error가 trust region 밖: additive QP block

#### 독립 분석

별도 agent가 MV-3 artifact/report만 읽고 독립 보고서를 작성한다.

### MV-4 — Short sequential proxy relevance

MV-0~3이 생존할 때만 실행한다.

#### Stream

- 동일 100 case
- two fixed orders primary
- non-overlapping micro-window
- canonical MEMIT, global-alpha matched progress, preregistered
  displacement-reducing allocation
- MV-2가 pass하면 refreshed allocation 추가

#### Primary

동일 window start state와 cumulative achieved rewrite progress에서:

```text
retention(displacement-reducing arm)
  - retention(global-alpha arm)
```

를 평가한다. observational `corr(Gini, retention)`은 secondary다.

#### Confound

- total displacement와 total norm
- current-edit underfitting
- failed edit 수
- stream position
- target token length/base difficulty
- repeated subject/relation/order

#### Kill/pivot

- intervention effect 없음:
  capacity mechanism kill
- correlation만 존재:
  degradation marker로 낮춤
- retention gain이 under-edit로 설명:
  matched-progress procedure 재설계

단, canonical baseline damage 자체가 replay/order noise를 넘지 않으면
“capacity effect 없음”으로 kill하지 않고
`inconclusive_at_this_horizon`으로 판정하여 더 긴 horizon의 비용/필요성을
별도로 결정한다.

#### 독립 분석

별도 agent가 MV-4 artifact/report만 읽고 독립 보고서를 작성한다.

## 6. Motivation closure

`motivation supported diagnostic`으로 닫으려면:

1. MV-0 양 model pass
2. MV-1 utility가 independent actual progress 예측
3. MV-2 refresh가 stale decision보다 이득
4. MV-3 matched-progress rerouting opportunity
5. joint prediction error가 accepted trust region 안
6. MV-4 paired microstream에서 proxy의 intervention relevance
7. untouched 20-case split과 결과를 보지 않은 third order에서 threshold/controller
   retuning 없이 재현

이 모두 필요하다.

중간 단계에서 kill criterion이 충족되면 뒤의 큰 실험을 실행하지 않고
`repo research direction killed` 또는 명시된 pivot으로 Motivation을 닫는다.

## 7. ODE naming gate와 claim-closing experiment

ODE 이름을 유지하려면 MV-2/MV-3에 다음을 포함한다.

- 최소 2회 accepted refresh
- 최소 3개 step size의 refinement curve
- fixed initial direction 대 refreshed direction
- proposal C-cosine/curvature
- step refinement에 따른 endpoint/trajectory stability
- Euler–Heun 또는 step-doubling consistency
- matched NFE에서 fixed-direction/static/Gauss–Seidel small-step 대비 refresh gain
- wall-clock, forward/backward, solve/NFE

한 step에서만 이득이 있거나 평균 1–2 round가 fixed-direction iterative method와
다르지 않으면 `ODE` 이름을 제거한다.

controller/QP 해 자체가 step size `h`에 따라 불연속적으로 바뀌어 하나의
`F(W,Z)`를 정의할 수 없다면 ODE discretization이 아니라 discrete
state-dependent controller로 기술한다.

## 8. Baseline scope

### Motivation 내부 필수

- native MEMIT
- fixed-delta K-step negative control
- global alpha/first-hit
- static layer scaling
- best single-layer/WilKE-style proxy
- `C`-normalized uniform allocation
- Gauss–Seidel small-step
- refreshed joint

`WilKE-style proxy`는 WilKE 재현이라고 부르지 않는다.

### Motivation 생존 후

- AlphaEdit native
- EMMET/PMET contextual
- external pinned ENCORE, NAS, BetaEdit
- capability-data budget을 분리한 CrispEdit
- LocFT-BF/WISE 등 다른 update family

local EasyEdit에 없는 방법을 implemented/reproduced baseline이라고 쓰지 않는다.

## 9. 통계

- primary unit: event 또는 non-overlapping paired window
- layer/context/order를 독립 sample로 세지 않음
- calibration과 confirmatory/untouched split 분리
- paired moving-block bootstrap + case-ID cluster sensitivity
- model별 결과를 별도 판정
- order별 sign 공개
- multiple primary contrast는 simultaneous CI/Holm correction
- technical failure와 target-not-reached를 ITD에서 제거하지 않음
- mean, median, sign fraction, trimmed mean과 denominator flow를 함께 보고

## 10. Resource and Slurm policy

### server1

- node: `devbox`
- project GPU cap: 3
- host-memory cap: 198117 MiB per requested GPU
- 현재 exact one-shot: Llama 1 case, 1 GPU, 8 CPU, 65000 MiB
- 현재 Qwen smoke는 Llama one-shot이 이미 끝난 뒤 별도 audit로 제출한다.
- 이후 양 model의 calibration/MV-1 이상은 각 model별 dependency와 red gate가
  모두 PASS하면 GPU 1개씩, 총 2개를 동시에 제출한다. 제출 직전 aggregate
  cap 3과 두 job의 memory cap을 함께 검사한다.
- job name: `motivation_*` 또는 `odeedit_*`
- 제출 직전:

```bash
scripts/check-slurm-resource-cap.sh server1 1 65000
```

### server4

- project GPU cap: 3
- host-memory cap: 65984 MiB per requested GPU
- repo/SH/onboarding이 없으므로 현재 제출 대상이 아니다.

GPU memory peak, host RSS, wall-clock, solve/NFE를 run metadata에 기록한다.

## 11. SH instruction envelope

현재 SH가 없으므로 아래는 등록 뒤 발행할 canonical 초안이다.

```text
명령 ID: session01-motivation-<mv>-<server>
대상: <server>의 등록된 server-head
대상 Codex session ID: servers/connection-inventory.md와
  servers/active/<server>.md의 server별 SH session ID
대상 repository CWD: servers/active/<server>.md의 exact clone path
대상 Git identity: hyunjun1127/ODE-edit

목적과 배경:
  ODE-Edit Motivation kill-test <MV>를 실행한다. 성능 우위나 paper claim을
  승인하는 task가 아니다.

허용 write path:
  local/results/raw/session01_motivation/<run_id>/
  runs/<run_id>/ (compact metadata)
  experiment-reports/servers/<server>/
  audits/servers/<server>/
  plans/updates/<server>/
  messages/server-heads/<server>/

Slurm 제출 허용 여부:
  envelope에 명시된 exact sbatch file과 resource만 allowed.

GPU cap:
  servers/local/gpu-caps.tsv의 해당 server row.
  scripts/check-slurm-resource-cap.sh가 pass해야 함.

red-team gate 통과 조건:
  session boundary, source/artifact hash, firewall, MV dependency,
  no-recompute guard, CPU test, resource cap 모두 pass.

artifact broadcast 의무:
  raw artifact 생성 후 scripts/rsync-artifact-broadcast.sh 사용.
  active peer가 없으면 exception 이유를 run metadata와 server report에 기록.

완료 보고 경로:
  messages/server-heads/<server>/<date>-<run_id>.md
  experiment-reports/servers/<server>/<date>-<run_id>.md
  audits/servers/<server>/<run_id>.postrun.md

금지 사항:
  EasyEdit file 수정; stats/projector 재계산/download; eval field를 edit process에
  전달; credential/raw log/model/checkpoint Git commit; unapproved resource;
  destructive rsync; 다른 Codex session/repo 조작.

예상 산출물:
  provenance manifest, sanitized request manifest, JSONL trace, compact metric
  summary, resource summary, firewall audit, artifact broadcast receipt/exception.

중단 조건:
  session/CWD/repo mismatch; imported-file/artifact hash mismatch; no-recompute
  guard activation; firewall violation; rollback hash mismatch; non-finite;
  GPU/memory cap failure; red block; Git conflict.
```

실험 종료 뒤 결과 분석은 이 실행 SH/agent가 아니라, 해당 artifact와 compact
report만 허용받은 별도 result-analysis agent가 맡는다.

## 12. Related-work deliverable

별도 agent가 primary source만 사용해 다음을 포함한 Korean report를 작성한다.

- ROME, MEMIT, AlphaEdit, EMMET, PMET
- WilKE와 layer selection prior
- ENCORE, NAS, LyapLock, BetaEdit, CrispEdit
- LocFT-BF, WISE, GRACE
- ODESteer, ODE-M
- information budget, layer allocation, history/norm/capacity, public code,
  reproducibility risk와 ODE-Edit novelty boundary

proposal에 없는 인접 prior가 확인되면 proposal-side narrative를 별도 patch한다.

## 13. 현재 다음 action

1. reusable hook과 CPU test 완료
2. artifact/source manifest 완료
3. red preflight 및 GH direct-submit exception 기록
4. exact Llama 1-case MV-0 implementation smoke 한 건
5. 별도 analysis agent와 red agent의 smoke report
6. pass일 때만 새 audit 아래 Qwen smoke와 MV-0 calibration case로 확장
7. 두 model MV-0가 pass일 때만 MV-1로 진행

아직 Slurm job은 제출하지 않았다.

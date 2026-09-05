# GH 실행 지시문 — Ordered Response-Barrier ODE-Edit fast main table

아래 내용을 ODE-edit repository의 Global Head(GH)에게 그대로 전달한다.

---

당신은 `hyunjun1127/ODE-edit`의 global-head다. 이번 지시의 ID는
`ODEEDIT-ORBODE-FAST-MAIN-V0`이다.

목표는 장기 사전 감사 문서를 더 만드는 것이 아니라, Ordered Response-Barrier ODE-Edit의
필수 attribution table을 가장 빠르게 GPU에서 얻는 것이다. 다음 두 문서를 canonical input으로
완전히 읽고 실행하라.

```text
/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-03-ordered-response-barrier-ode-edit-proposal.md
/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-03-ordered-response-barrier-ode-edit-fast-main-table.md
```

Proposal expected identity는 SHA256
`03a6fc61258e7643fc281fab8ab0d3ca700bb80f09aaa3eb34591ce5827087be`, 55,900 bytes,
1,359 lines다. Byte identity가 다르면 어느 문서를 임의로 택하지 말고 사용자에게 보고하라.

현재 caller registry 기준 GH session은
`01a04939-8873-7673-8dca-4c7fc5e31af0`, CWD는
`/mnt/raid5/janghj/ODE-edit`, repository는 `hyunjun1127/ODE-edit`이다. 실제 실행 시
`servers/connection-inventory.md`와 `scripts/check-session-boundary.sh`로 다시 확인하고, session이
교체되었으면 registry부터 갱신하라.

## 1. 즉시 적용할 우선순위

1. 상세 G0 audit를 별도 장기 단계로 운영하지 않는다.
2. 최소 CPU tests와 실제 GPU process 안의 fail-close runtime preamble만 수행한다.
3. 같은 model load에서 B1 smoke를 통과한 뒤 즉시 sealed B100 round 0으로 진행한다.
4. Round 0 metrics를 즉시 interim table로 발행하고, jobs는 기다리지 말고 round 1–9를 계속한다.
5. 네 model×method cells를 GPU 4개로 병렬 실행한다.
6. 결과가 나쁘다는 이유로 retry, rescue, hyperparameter 변경 또는 arm 제거를 하지 않는다.

이 지시는 GPU experiment stage 진입을 승인한다. 단, 아래에 명시한 source/stream/session/resource
identity와 runtime preamble이 실패하면 submission 또는 해당 cell을 fail-close해야 한다.

## 2. Dirty root를 구현에 사용하지 말 것

현재 사용자 workspace에는 untracked proposal과 다른 사용자 변경이 있을 수 있다. Reset, stash,
cleanup, checkout overwrite를 하지 말라. 최신 `origin/main`에서 clean dedicated worktree와

```text
codex/orbode-fast-main-v0
```

branch를 만들고 구현하라. 시작 시 최신 `origin/main` commit/tree를 receipt에 기록한다. 이
지시 작성 시 관측한 remote main은
`bf9290c76b7b684ce3d6823e70d60e0b79a8820c`이지만, 실행 시점의 latest remote가 authority다.

Proposal과 fast plan은 위 SHA/path의 bytes를 source manifest에 넣는다. 기존 dirty root의 다른
변경을 branch에 섞지 않는다.

## 3. 구현 범위

새 package를 다음 경로에 만든다.

```text
project/run_scripts/ordered_response_barrier_ode/**
```

최소 구성은 `contracts`, four-cell `adapters`, grouped FULL_FP32 virtual overlay,
detached-direction terminal JVP, ordered integrator, target-context strict predicate, telemetry,
runtime, dry-plan, report와 tests다.

재사용 가능한 source는 새 plan §9에 적힌 기존 public adapters와 engineering primitives다. 과거
controller를 새 이름으로 감싸지 말라. 특히 다음은 금지한다.

- CAKE causal score/weight, causal tracing dataset 또는 새로운 external reference set
- rephrase, neighborhood, PS/NS, target_true, prior-edit replay의 controller/terminal 사용
- output/history/action QP 또는 hard barrier
- (q^{\rm res}), covariance action 또는 Frobenius action의 (u,α), terminal 영향
- `Writer(hR)` 뒤에 다시 h를 곱하는 double scaling
- (R/n)을 ours response arm 안에 숨겨 유지
- h-dependent controller, line search, candidate rejection, retry, backtracking, best waypoint, fallback
- inner AlphaEdit history append 또는 MEMIT `COV_CACHE` mutation
- stale residual/key/writer/JVP reuse
- dynamic-(z), per-request removal 또는 outcome-based active set
- model별 (N,h,u) rescue

Canonical main은 full current residual (R)에서 official-form writer direction (B)를 만들고,
detached (B)의 terminal JVP로 h-independent (u=\min(1,[g]_+/r))를 계산한 뒤
α=`h*u`만 적용한다. (T=1,N=4,h=.25), layer order `L4->L5->L6->L7->L8`, four sweeps다.

AlphaEdit terminal finalization은 intermediate/path key를 append하지 않는다. Terminal model에서
모든 edited layer key를 다시 계산하고 native semantics의 history delta를 detached shadow에서
만든 뒤 weights와 history를 한 guarded transaction으로 정확히 한 번 반영한다. MEMIT은 static
`COV_CACHE` identity를 유지한다.

## 4. 필수 arms

Arm runner는 하나의 configuration schema로 만들고 다음 순서를 고정한다.

```text
O      = stock Official R/n one pass
QCL    = current R/n, u=1, per-visit refresh, N=4, h=.25, first-hit OFF
NQFIX  = full current R, u=1, per-visit refresh, N=4, h=.25, first-hit OFF
ORBFH  = full current R, response u, per-visit refresh, N=4, h=.25, first-hit OFF
JAC    = full sweep-entry R, response u, per-sweep field refresh, N=4, h=.25, first-hit OFF
ORBHit = exact first TARGET_CONTEXT_STRICT_HIT prefix of ORBFH; no second integration
```

`QCL`은 optional이 아니다. `QCL vs NQFIX`가 (R/n) quota 제거의 유일한 matched-clock
comparison이다. `JAC`도 optional이 아니다. `ORBFH vs JAC`가 upstream actual transition 뒤
downstream field rebuild의 comparison이다.

`ORBHit`은 B1에서 direct early-stop run과 prefix-derived shadow endpoint의 weight, activation,
logit identity가 통과할 때만 derived arm으로 쓴다. 실패하면 first-hit column만
`WITHHELD_TECHNICAL`로 두고 ORBFH main table은 계속한다. Strict-z arm은 이번 submission에서
제외한다.

## 5. 데이터와 job packing

다음 sealed ten-B100 stream을 사용한다.

```text
project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json
stream_root = 467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a
order_root  = 018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3
```

Ten cohorts는 서로 독립이다. 매 cohort와 arm을 같은 (W_0)/entry cache에서 시작하고, 결과를
lifelong accumulation으로 해석하지 말라. Cohort마다 family-native (z^\star)를 W0에서 정확히
한 번 계산하고 모든 arms가 tensor bytes와 context/request order를 공유한다.

Server4를 primary execution target으로 사용하되 live registry, asset path, storage와 GPU cap을
다시 검사하라. 현재 registry의 SH4 session/CWD는 다음이다.

```text
session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
CWD: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit
```

Session이 바뀌었으면 stale session에 보내지 말고 registry를 갱신한다. GH→SH4는 app-server direct
coordination을 사용하고 Git inbox를 live fallback으로 쓰지 않는다.

Slurm array mapping은 고정한다.

```text
0 = Llama-3-8B-Instruct x MEMIT
1 = Llama-3-8B-Instruct x AlphaEdit
2 = Qwen2.5-7B-Instruct x MEMIT
3 = Qwen2.5-7B-Instruct x AlphaEdit
array = 0-3%4
gpu = 1 per cell
host memory <= 65,984 MiB per cell
--export=NONE
```

각 cell은 model을 한 번만 load하고 B1 runtime preamble, round 0, round 1–9를 같은 process에서
수행한다. Round마다 artifact를 원자적으로 flush한다. Job walltime은 dry-run profiler가 산출한
upper bound에 model-load 여유를 더해 요청하되, 결과를 본 뒤 array mapping이나 cap을 바꾸지
말라.

## 6. 실행 envelope

- Slurm: 위 one-shot four-cell array의 submission은 `allowed`다.
- Submission 전: live session boundary, exact pushed source commit, source/stream/HF/artifact hashes,
  `scripts/check-slurm-resource-cap.sh server4 4 263936`, exclusive output path, one-shot marker와
  red preflight가 모두 PASS해야 한다.
- GPU cap: server4 project cap 4를 넘지 않는다. 기존 active project GPU가 있으면 합계가 4 이하가
  될 때까지 pending한다.
- Red gate: validity/resource/data leakage `block`은 중단한다. 단순 문서 세부사항 `warn`은 명시해
  진행할 수 있다. 결과 해석을 이유로 실행을 막지 않는다.
- Retry: submit 실패 또는 cell failure 뒤 자동 resubmit 금지. 완료 round는 보존하고 정확한 failure
  stage를 보고한다.
- Artifact: raw output/log는 Git에 넣지 않고 아래 local root에 둔다.

```text
/data/janghj/ODE-edit/local/ordered-response-barrier-ode/fast-main-v0/
```

- Broadcast: terminal 후 ordinary local artifacts를 protocol helper로 active peers에 broadcast한다.
  불필요/불가하면 exact exception을 기록한다.

SH4에게 허용할 tracked write scope는 다음으로 제한한다.

```text
project/run_scripts/ordered_response_barrier_ode/**
project/run_scripts/session07_orbode_fast_main_*.py
project/run_scripts/session07_orbode_fast_main_*.sbatch
audits/servers/server4/orbode-fast-main-v0*.md
runs/orbode-fast-main-v0/**
experiment-reports/servers/server4/orbode-fast-main-v0/**
messages/server-heads/server4/2026-09-03*.md
agents/server4/server4-server-head.json
```

SH4는 dedicated non-main branch에서 source/tests를 commit할 수 있으나 GH 승인 없이 main merge를
하지 않는다. GH는 source review와 preflight PASS 뒤 execution commit을 push하고 exact SHA로만
submit한다.

## 7. Runtime preamble

별도 exhaustive audit job을 만들지 말고 각 cell 안에서 다음만 수행한다.

1. Official wrapper/direct endpoint parity와 (B(R/n)=B(R)/n) scaling smoke
2. 첫 nonzero layer의 detached terminal JVP를 FD ε
   `{2^-7,2^-8,2^-9}`와 비교하고 virtual/materialized sign identity 확인
3. L4 advance 뒤 L5 state/key/writer가 새 version을 읽는지 확인
4. FP32 overlay/shadow materialization, terminal activation/logit identity 확인
5. inner persistent mutation, algorithmic retry/rollback 0 확인
6. 첫 request에서 (T=1,N=2,4,8) resolution smoke 기록

이 검사는 `FAST_RUNTIME_PREAMBLE`이며 full G0라고 과장하지 않는다. Nonfinite, sign contradiction,
stale state, parity/transaction failure는 cell hard stop이다. Numerical tolerance는 result table을
보기 전에 lock JSON에 기록한다.

## 8. 필수 telemetry

Layer당 coefficient는 batch-global scalar이므로 request-specific guarantee를 주장하지 말라.
다음을 request level로 반드시 기록한다.

- batch (g,r,u), (u=0,(0,1),1) visit counts
- batch (g>0)일 때 request별 (g_i\le0) fraction
- actual request potential worsening fraction/tail
- (q^{res}_{max}), (q^{res}_{terminal})
- entry-normalized realization/model error (D^{R0}), (D^{model,R0})
- orthogonal response, actual/predicted response와 discrete barrier defect
- zero-command status; zero-command row를 debt=0으로 impute하지 않음
- forward/JVP/FD/solve counts, factor rank, peak memory와 wall time
- terminal-valid/scientific-attempted/technical-attempted denominators

Action (C^{reg}) binding이 준비되지 않아도 endpoint run은 진행하고 해당 열만
`TELEMETRY_WITHHELD`로 둔다. Frobenius path/net action은 항상 기록하되 architecture 간 직접
비교하지 않는다.

## 9. 보고 순서

Round 0 종료 즉시 다음 interim files를 만들고 GH에게 app-server direct로 path를 알린다.

```text
runs/orbode-fast-main-v0/round00-*/
experiment-reports/servers/server4/orbode-fast-main-v0/round00-factual-ko.md
```

SH report는 사실/수치만 쓴다. GH가 별도로 다음 세 표를 작성한다.

1. endpoint RS/PS/NS, NLL, strict/coverage/failure
2. realization/model-error/q-residual/orthogonal/defect tails
3. runtime/JVP/solve/rank/memory/action

Jobs는 interim 해석을 기다리지 않고 round 1–9를 계속한다. Final factual paths는 다음을 사용한다.

```text
runs/orbode-fast-main-v0/final-*/
experiment-reports/servers/server4/orbode-fast-main-v0/final-factual-ko.md
audits/servers/server4/orbode-fast-main-v0.postrun.md
```

GH integrated report는

```text
experiment-reports/global/orbode-fast-main-v0.md
```

에 둔다.

## 10. 결과 판정

다음 비교만 attribution authority로 사용한다.

```text
QCL   vs NQFIX : explicit R/n quota
NQFIX vs ORBFH : current response multiplier
ORBFH vs JAC   : within-sweep realized-state rebuild
ORBFH vs ORBHit: stopping-only effect
O vs all       : operational stock reference only
```

Round 0은 fast diagnostic이다. Ten paired cohorts가 완료되기 전 paper claim으로 승격하지 말라.
Mechanism tails만 개선되고 held-out PS/NS가 개선되지 않으면 writer-mechanics result로 남긴다.
Immediate efficacy와 mechanism이 모두 살아남은 경우에만 v6 lifelong 1k task를 새 envelope로
제안하고, 이번 지시 권한으로 lifelong을 제출하지 않는다.

마지막으로 완료 보고에는 다음을 구분해서 써라.

- proposal에서 고정된 내용
- repo/source/runtime에서 확인한 사실
- SH factual result
- GH interpretation
- 아직 검증되지 않은 claim


# Session 01 Motivation — MV-1 C1 fold-0 동시 execution gate

- 작성일: 2026-07-31
- 범위: server1에서 고정 Llama/Qwen의 held-out C1 fold `0`을 GPU 1개씩
  동시에 실행
- 감사 범위: D1 technical closure, frozen policy/forecast, exact split/arm,
  resource/session/artifact boundary
- 비감사 범위: C1 결과, ODE dynamics, retention, benchmark gain

## 네 범주

### proposal에서 온 내용

- same-snapshot rewrite-side signal이 event별 allocation opportunity를
  식별하는지 작은 held-out diagnostic에서 먼저 확인한다.
- 신호가 static allocation으로 설명되거나 actual progress로 이어지지 않으면
  큰 ODE 실험 전에 controller를 kill 또는 pivot한다.

### repo/protocol에서 확인한 사실

- D1 pair job `15565`는 parent와 두 child 모두 `COMPLETED`, `ExitCode=0:0`다.
  두 child는 `2026-07-31T00:55:10`에 동시에 시작했고 각각 GPU 1, CPU 8,
  `65000M`을 사용했다.
- 양 model 모두 planned/attempted/pass `12/12/12`, outcome `60`, exact
  rollback/count/firewall을 통과했다. Projector는 load하지 않았고 pinned
  covariance 5개만 사용했다.
- model별 독립 D1 분석은 `artifact_valid=true`, `panel_complete=true`,
  `error_codes=[]`이며 D0+D1 17-case feature-only static policy와
  calibration-only forecast를 생성했다.
- C1은 seed `ode-edit-mv1-score-mix-confirmatory-folds-v1`로 outcome-blind
  5-way round-robin hash한 confirmatory 60 중 fold `0`의 exact 12 case다.
- server1 cap은 project GPU 3, GPU당 host memory `198117 MiB`다.

### GH 추정

- Calibration-only expected `adaptive - frozen_static` gap은 Llama
  `+0.0109636724`, Qwen `+0.1120153104` raw progress unit다. 이는 C1
  비교용 사전 기대치이며 method gain, benchmark %, retention 또는
  population effect 추정이 아니다.
- D1 wall time과 C1의 6-arm panel을 고려하면 양 model paired wall time은
  대략 1.5–2.5시간으로 추정한다. `06:00:00`은 실패 진단 여유를 포함한다.

### 사용자 확인 필요

- 없음. 사용자는 필수 감사만 거쳐 Llama와 Qwen을 순차가 아니라 함께
  빠르게 제출하라고 명시했다.

## 고정 policy와 사전 기대효과

| model | static policy hash | forecast policy hash | expected raw gap | calibration bootstrap 95% | residual envelope |
| --- | --- | --- | ---: | ---: | ---: |
| `llama3-8b-inst` | `e37443fa075245aa52828773ffc062217d983fb46254d1f40010d396be3d5e06` | `da5b352cbae20a44ff24e22f9c598f4a5df88aeab373995df3025ce2d0894533` | `+0.0109636724` | `[+0.0052916783,+0.0175079456]` | `0.0012365315` |
| `qwen2.5-7b-inst` | `381e22334e0e3d07c37f07db16c22e8cdd1b689deba66ec7b032dd32ba0b2955` | `a1feb93481f64785539c102a1ac58340fee71ba4e91a9ccc4b8f699cc6c04c99` | `+0.1120153104` | `[+0.0647644145,+0.1762966650]` | `0.0323610987` |

Static policy는 D0+D1 feature만 사용하고 outcome field를 받지 않는다.
Forecast의 `beta`만 calibration adaptive-vs-uniform outcome을 사용하며
`confirmatory_outcomes_used=[]`다. C1 runner는 두 policy의 tracked path,
canonical hash, model/run identity를 outcome 전에 검증하고 action receipt에
고정한다.

## exact 실행 envelope

| 항목 | 고정값 |
| --- | --- |
| parent | `odeedit_mv1mix_c1_pair_v1`; `devbox`; A6000 2; task 2; CPU 16; `130000M`; `06:00:00` |
| Llama child | GPU 1; CPU 8; `65000M`; `llama3-8b-inst mv1mix_llama_c1_v1` |
| Qwen child | GPU 1; CPU 8; `65000M`; `qwen2.5-7b-inst mv1mix_qwen_c1_v1` |
| runner | `python -m project.run_scripts.ode_edit_motivation.mv1_score_mix_confirmatory` |
| q / fold | `1/256`; fixed fold `0`; exact 12 case/model |
| output | `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_c1_v1/` |
| marker | `local/state/slurm-submissions/session01_motivation/mv1mix_c1_pair_v1.submitted/` |

Outcome arm order는 다음 여섯 개로 고정한다.

```text
score_mix
frozen_static_mix
uniform
ordered_global_alpha
native_memit_full
no_op_replay
```

Primary는 같은 `C`의
`progress(score_mix) - progress(frozen_static_mix)`다. Static/uniform/global
alpha까지 네 arm만 matched-`C` oracle에 포함하고, native MEMIT은 contextual
reference, no-op은 numerical envelope로만 쓴다.

## 최소 red gate와 판정 경계

- hard block: session/repo/job/model/fold/count/arm/policy hash/resource 불일치,
  dirty 또는 unpushed Git, 중복 output/marker/job, online download,
  moments/projector 재계산·수정·deserialize, EasyEdit write, outcome leakage,
  unequal-`C`, non-durable receipt, rollback/hash/firewall 위반, child nonzero
- pair clear: 양 model 모두 primary mean이 replay envelope보다 크고,
  20% trimmed mean·median·positive sign `>=7/12` 중 하나 이상이 같은 방향
- architecture-conditional 또는 gray: 한 model만 clear하거나 핵심 summary가
  엇갈리면 고정 policy의 fold `1` 한 번만 허용하고 retuning 금지
- routing kill: 양 model 모두 mean과 trimmed mean이 envelope 이하,
  sign `<=0.50`, matched-`C` oracle mean도 envelope 이하일 때만 발동
- adaptive가 null이지만 oracle이 남으면 연구 전체를 kill하지 않고
  controller/static-policy pivot으로 판정
- CI endpoint 하나나 secondary metric 하나만으로 core mean/robust/sign
  signal을 기각하지 않는다.

Clear도 method GO나 MV-2 승인이 아니다. 같은 policy/threshold의 untouched
20-case replication이 재현될 때만 최소 stale-vs-refreshed MV-2를 연다.

## 실행·Git·artifact 경계

- 허용 명령은 인자 없는
  `project/run_scripts/submit_session01_mv1mix_c1_pair_server1.sh` 한 건이다.
- Helper는 이 audit의 exact verdict와 D1 post-run `C1 PREPARE`, tracked
  policy/report/code, clean `main == origin/main`, output/marker/job 부재,
  session 및 resource cap을 검사한 뒤 `sbatch --export=NONE`을 한 번만 호출한다.
- `py_compile`, Motivation suite `105/105`, wrapper `bash -n`/`shellcheck`,
  `git diff --check`, agent access, session boundary가 통과했다.
- cap check는 active project GPU `0` + requested `2` ≤ `3`, memory
  `130000M` ≤ `396234 MiB`로 `submit_now`다.
- 현재 active peer clone과 SH가 없으므로 artifact broadcast는 예외다.
  Raw artifact/log/direct-z는 ignored `local/`에만 남기고 Git에는 small
  report/policy/hash/metric summary만 기록한다.
- GH 직접 제출 사유는 사용자의 time-critical 동시 제출 지시와
  `PROTOCOL.md:506-509`다. 영향 범위는 이 exact pair와 위 `local/` 경로뿐이다.

- 최종 판정: `PASS` — MV-1 C1 fold-0 동시 pair 한 건에만 유효

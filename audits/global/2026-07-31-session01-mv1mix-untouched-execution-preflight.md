# Session 01 Motivation — MV-1 untouched 20 동시 execution gate

- 작성일: 2026-07-31
- 범위: server1에서 고정 Llama/Qwen의 canonical untouched 20을 GPU 1개씩
  동시에 실행
- 감사 범위: C1 pair clear, split/policy/controller lock, resource/session/
  artifact boundary
- 비감사 범위: untouched outcome, MV-2, ODE dynamics, retention, benchmark gain

## 네 범주

### proposal에서 온 내용

- Same-snapshot rewrite-side signal은 작은 held-out diagnostic과 untouched
  replication에서 재현된 뒤에만 stale-vs-refreshed mechanism test로 승격한다.
- Signal이 재현되지 않거나 static/controller로 설명되면 큰 ODE 실험 전에
  kill 또는 pivot한다.

### repo/protocol에서 확인한 사실

- C1 job `15576`은 parent와 두 child 모두 `COMPLETED`, `ExitCode=0:0`다.
  두 child는 `2026-07-31T02:28:04`에 동시에 시작했고 각각 GPU 1, CPU 8,
  `65000M`을 사용했다.
- 양 model 모두 planned/attempted/pass/failure `12/12/12/0`, outcome `72`,
  exact rollback/count/firewall을 통과했다.
- 독립 분석의 primary `adaptive - frozen_static`은 다음과 같다.

| model | mean | 20% trimmed mean | median | positive sign | paired bootstrap 95% | replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama | `0.0087552766` | `0.0066670030` | `0.0055541992` | `12/12` | `[0.0044102512,0.0139340937]` | `1e-12` |
| Qwen | `0.0594923000` | `0.0359131545` | `0.0374884605` | `11/12` | `[0.0171300006,0.1230557804]` | `1e-12` |

- §10.8 pair clear는 양 model 모두 mean이 replay보다 크고, model별
  trimmed mean·median·sign `>=7/12` 중 하나 이상이 같은 방향일 때다.
  두 model 모두 이 조건을 충족한다.
- Server1 cap은 project GPU 3, GPU당 host memory `198117 MiB`다.

### GH 추정

- Calibration forecast 대비 realized mean은 Llama `-0.0022083958`, Qwen
  `-0.0525230105` raw progress unit다. 두 모델 모두 방향은 재현됐지만 Qwen
  forecast는 규모를 높게 예측했다.
- C1 per-event wall time의 단순 환산은 untouched pair wall time 약
  1시간 45분–2시간 20분이다. 이는 runtime 추정이며 effect claim이 아니다.

### 사용자 확인 필요

- 없음. 사용자는 필수 감사만 수행하고 Llama/Qwen을 함께 빠르게 제출하라고
  명시했다.

## exact lock

| 항목 | 고정값 |
| --- | --- |
| split | selection seed `ode-edit-motivation-counterfact-v1`의 canonical untouched exact 20; 최초 1회 개봉 |
| parent | `odeedit_mv1mix_untouched_pair_v1`; `devbox`; A6000 2; task 2; CPU 16; `130000M`; `08:00:00` |
| Llama child | GPU 1; CPU 8; `65000M`; `mv1mix_llama_untouched_v1` |
| Qwen child | GPU 1; CPU 8; `65000M`; `mv1mix_qwen_untouched_v1` |
| runner | `python -m project.run_scripts.ode_edit_motivation.mv1_score_mix_followup --mode untouched` |
| q / seed | `1/256`; run seed `17` |
| outcome arms | `score_mix`, `frozen_static_mix`, `uniform`, `ordered_global_alpha`, `native_memit_full`, `no_op_replay` |
| Llama policy | static `e37443fa075245aa52828773ffc062217d983fb46254d1f40010d396be3d5e06`; forecast `da5b352cbae20a44ff24e22f9c598f4a5df88aeab373995df3025ce2d0894533` |
| Qwen policy | static `381e22334e0e3d07c37f07db16c22e8cdd1b689deba66ec7b032dd32ba0b2955`; forecast `a1feb93481f64785539c102a1ac58340fee71ba4e91a9ccc4b8f699cc6c04c99` |
| output | `local/results/raw/session01_motivation/mv1mix_{llama,qwen}_untouched_v1/` |
| marker | `local/state/slurm-submissions/session01_motivation/mv1mix_untouched_pair_v1.submitted/` |

Fold1은 이 경로와 함께 실행하지 않는다. Direct-z, synchronous/ordered factor와
probe는 event별 한 번만 만들고 six arm에 재사용한다. Static/forecast action과
receipt는 outcome 전에 고정하며 policy refit, threshold/controller retuning,
moments/projector 재계산·deserialize, EasyEdit write와 online download를
금지한다.

## 최소 red gate와 결과 경계

- hard block: session/repo/job/run/model/split/count/policy/q/arm/resource
  불일치, dirty/unpushed Git, 중복 output/marker/job, outcome leakage,
  unequal-`C`, receipt/rollback/hash/firewall 위반, child nonzero
- MV-1 reproduced input: 양 model 모두 mean이 replay보다 크고, model별
  trimmed mean·median·positive sign `>=11/20` 중 하나 이상이 같은 방향
- architecture-conditional: 한 model만 clear이고 다른 model이 materially
  negative가 아닐 때만 model-specific로 제한
- routing kill: 양 model 모두 mean·trimmed mean·oracle이 null이고
  sign `<=10/20`
- oracle만 남으면 controller/static-policy pivot이며 MV-2를 열지 않는다.
- CI endpoint 하나나 secondary metric 하나로 core mean/robust/sign signal을
  기각하지 않지만, technical block은 lenient하게 넘기지 않는다.

Untouched clear도 benchmark %, retention 또는 완성된 ODE-Edit method gain을
뜻하지 않는다. 허용되는 다음 조치는 최소 MV-2 stale-vs-refreshed diagnostic
뿐이다.

## 실행·Git·artifact 경계

- 허용 명령은 인자 없는
  `project/run_scripts/submit_session01_mv1mix_untouched_pair_server1.sh`
  한 건이다.
- Helper는 C1 post-run exact `UNTOUCHED PREPARE`, 이 audit의 exact `PASS`,
  tracked code/tests/spec/wrappers/policies/reports, clean
  `main == origin/main`, output/marker/job 부재, session/resource cap을
  검사한 뒤 `sbatch --export=NONE`을 한 번만 호출한다.
- Motivation suite `126/126`, `py_compile`, wrapper `bash -n`,
  `git diff --check`, agent access와 credential/private-endpoint scan이
  통과했다. `shellcheck`는 host에 없어 실행하지 못했다.
- Cap check는 requested GPU `2` ≤ project cap `3`, memory
  `130000M` ≤ `396234 MiB`다. 실제 제출 직전에 active usage를 다시 검사한다.
- 현재 active peer clone과 SH가 없어 artifact broadcast는 예외다.
  Raw/log/direct-z는 ignored `local/`에만 두고 Git에는 small
  report/hash/metric summary만 기록한다.
- GH 직접 제출 사유는 사용자의 time-critical 동시 제출 지시와
  `PROTOCOL.md:506-509`다. 영향 범위는 이 exact pair와 위 `local/` 경로뿐이다.

- 최종 판정: `PASS` — MV-1 untouched 동시 pair 한 건에만 유효

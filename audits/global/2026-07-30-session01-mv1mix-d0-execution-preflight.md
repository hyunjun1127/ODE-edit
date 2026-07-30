# Session 01 Motivation — MV-1 score-mix D0 동시 execution gate

- 작성일: 2026-07-30
- 범위: server1에서 고정 Llama/Qwen의 `calibration[3:8]` score-mix D0를
  GPU 1개씩 동시에 실행

## 네 범주

### Proposal에서 온 내용

- same-snapshot rewrite-side signal이 같은 `C` budget의 static allocation보다
  나은 event-specific direction을 식별할 수 있는지 먼저 진단한다.
- Motivation signal이 없으면 큰 실험이나 ODE refresh로 넘어가지 않고
  현재 continuous-routing 방향을 빠르게 kill한다.

### repo/protocol에서 확인한 사실

- v1 pilot은 실행 fidelity를 통과했지만 six-arm set으로 continuous mixture를
  식별할 수 없어 `REVISE-before-full` 판정을 받았다.
- v2 action은 다섯 single-layer central-FD slope만으로 outcome 전에
  `relu(s)/L2` mixture를 고정하며, primary는 같은 `C` budget의
  `G_mix=P(score_mix)-P(uniform)`이다.
- D0는 기존 pilot `[0:3]`을 제외한 calibration `[3:8]` 다섯 case다.
- 고정 대상은 `llama3-8b-inst`, `qwen2.5-7b-inst`이며 하나의 paired
  allocation에서 동시에 시작한다.
- EasyEdit와 pinned moments/projector는 read-only다. moments/projector
  재계산, 수정, download와 projector deserialization은 금지된다.
- GH session은 `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`, CWD는
  `/mnt/raid5/janghj/ODE-edit`, repository ID는 `hyunjun1127/ODE-edit`다.
- server1 project cap은 GPU 3개, GPU당 Slurm host-memory cap은
  `198117 MiB`다. 이번 요청은 GPU 2개와 host memory `130000M`이다.
- raw artifact/log/state는 `local/`에만 쓰며 Git에는 넣지 않는다.

### GH 추정

- v1 slope-only pure-rotation forecast의 약 `+0.71%`(Llama),
  `+1.22%`(Qwen)는 D0를 정당화하는 작은 국소 가설일 뿐 actual method
  gain이나 population estimate가 아니다.
- 다섯 case는 이 작은 realized contrast가 replay envelope를 한 event에서라도
  넘는지만 확인하는 최소 early-kill diagnostic이다.

### 사용자 확인 필요

- 없음. 사용자가 필수 감사만 수행하고 Llama/Qwen을 순차가 아니라 함께
  빠르게 제출하라고 명시했다.

## exact 실행 envelope

| 항목 | 고정값 |
| --- | --- |
| parent | `odeedit_mv1mix_d0_pair_v1`; `devbox`; A6000 2; task 2; CPU 16; `130000M`; `06:00:00` |
| Llama child | GPU 1; CPU 8; `65000M`; `llama3-8b-inst mv1mix_llama_d0_v1 3 5` |
| Qwen child | GPU 1; CPU 8; `65000M`; `qwen2.5-7b-inst mv1mix_qwen_d0_v1 3 5` |
| runner | `python -m project.run_scripts.ode_edit_motivation.mv1_score_mix` |
| output | `local/results/raw/session01_motivation/{mv1mix_llama_d0_v1,mv1mix_qwen_d0_v1}` |
| marker | `local/state/slurm-submissions/session01_motivation/mv1mix_d0_pair_v1.submitted/` |

허용 명령은 인자 없는 다음 한 건뿐이다.

```bash
project/run_scripts/submit_session01_mv1mix_d0_pair_server1.sh
```

Helper는 지정 wrapper·runner·analyzer·test·spec·audit의 Git 추적 상태,
clean `main == origin/main`, agent/session/repository boundary, 두 output과
marker 부재, 동일 job 부재, server1 GPU/host-memory cap을 fail-closed로
검사한 뒤 `sbatch --export=NONE`을 정확히 한 번 호출한다.

## 최소 red-team 중단 조건

- 두 child가 동시에 시작하지 않거나 각각 GPU 1, CPU 8, memory `65000M`이 아님
- repo/session/job/run ID/slice/case count가 exact envelope와 다름
- dirty worktree, `main != origin/main`, 기존 output/marker, 동일 active job
- online download, moments/projector 재계산·수정·deserialize, EasyEdit write
- score mixture가 outcome을 보거나 receipt보다 operational outcome이 먼저 열림
- `score_mix`와 `uniform`의 actual `C` energy 불일치
- exact rollback, replay, hash, selection, teacher suffix, firewall 위반
- 어느 child든 nonzero, 또는 model별 `planned/attempted/pass != 5/5/5`

D0 positive는 D1만 허용하고 GO, confirmatory, MV-2 또는 ODE-Edit gain
claim을 허용하지 않는다. Model별 replay envelope는 결과 전에
`e_m=max(1e-12, max_i abs(P_i(no_op_replay)))`로 고정한다. Replay
exact-logits/hash와 technical validity가 선행 조건이며 missing/non-finite를
0으로 바꾸지 않는다. 양 model 각각에서 다섯 `G_mix`가 모두 `e_m` 이하일
때만 current continuous-routing 방향을 early kill한다.

제출 직후 두 Slurm step의 동시 `RUNNING`과 child별 자원을 확인한다.
불일치 시 해당 pair 한 건만 취소하고 재제출하지 않는다. 완료 후 model별
독립 analysis agent와 pair post-run red audit만 수행한다. 현재 다른 active
server-head가 없으므로 즉시 artifact broadcast는 예외로 기록하고, 향후
peer가 활성화되면 protocol helper로 검증·전송한다.

GH 직접 제출 근거는 사용자의 time-critical 동시 실행 지시와
`PROTOCOL.md:506-509`다. 영향 범위는 이 paired D0 한 건과 위 `local/`
경로뿐이다. 후속 보고 경로는
`experiment-reports/global/2026-07-30-mv1mix-*-d0-v1-analysis.md`,
`audits/global/2026-07-30-mv1mix-d0-pair-v1.postrun.md`,
`runs/mv1mix_*_d0_v1/`이다.

정적 검증은 Motivation CPU suite `72/72`, runner/analyzer `py_compile`,
세 shell script `bash -n`, global-head write boundary, session boundary,
GPU/host-memory cap, `git diff --check`를 통과했다. `sbatch --test-only`는
GPU 2, CPU 16, `130000M` allocation을 수용했고 실제 job은 만들지 않았다.

독립 red review는 첫 analyzer 초안에서 실제 runner와 다른 feature/action
schema와 arm label, runner보다 과도하게 엄격한 equal-`C` tolerance를
발견해 제출을 차단했다. Analyzer와 synthetic fixture를 runner schema와
exact parity로 수정하고, `actual_unit_c_energy`, arm별 budget label,
runner와 같은 `math.isclose(rel_tol=3e-5, abs_tol=3e-5)`, replay threshold,
invalid-panel early-kill 차단을 추가했다. 재검토에서 전용 tests `9/9`,
전체 suite `72/72`와 P0/P1 blocker 없음이 확인됐다.

- 최종 판정: `PASS` — MV-1 score-mix D0 동시 pair 한 건에만 유효

# Session 02 P0 technical pair — zero-outcome failure synthesis

- 작성 시각: **2026-08-03T18:03:44+09:00**
- GH session: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- canonical SH1 session: `019fc63e-5217-7250-9c22-c5b2ec4248f0`
- instruction: `ODEEDIT-S02-P0-TECH-PAIR-V1`
- execution source: `f0db6743bed05be4c7073dbfe4d2d0ec11ea0961`
- numerical-lock proposal ID:
  `7d15789d5952f281166db8579db1cd7397d3dbb419b447178eeb0c7976d06a95`
- 최종 판정: **`P0_TECHNICAL_BLOCK; ZERO_SCIENTIFIC_OUTCOME; P1_MAIN_TABLE_HOLD`**

## 한 줄 결론

두 모델은 loading, offline manifest와 entry event identity까지는 통과했지만, Qwen은 dense
synchronous-system 조립 중 GPU OOM, Llama는 Full arm의 hook-reference identity gate 실패로
각각 중단됐다. 따라서 이번 pair는 ODE-Edit의 성능이나 가설을 평가한 실험이 아니라 서로 다른
두 implementation blocker를 발견한 기술 profiler다.

## 네 범주 구분

### Proposal에서 온 내용

- 고정 direct-z 아래에서 모든 edit layer의 proposal을 동일한 current joint model state에서
  synchronous하게 다시 선형화한다.
- joint partial update 뒤 proposal과 allocation을 다시 계산하고 first-hitting state에서 멈춘다.
- Llama와 Qwen에 동일한 method/controller가 작동해야 하며 모델별 rescue는 허용하지 않는다.

### Repo/protocol 및 SH 보고에서 확인한 사실

- 두 job은 같은 submission batch에서 각 1 GPU로 제출됐다. 실행은 약 5분 안에 모두
  fail-closed했으며 자동 retry, source/parameter 변경, output cleanup은 없었다.
- 두 모델 모두 4/4 model shards를 load하고 offline preflight, manifest와 Native arm entry를
  통과했다.
- combined-event identity의 보고값은 Llama `new=1.2397766e-05`,
  `old=9.8347664e-06`; Qwen `new=2.6702881e-05`, `old=2.5153160e-05`로
  둘 다 해당 technical gate를 통과했다.
- Qwen job `16026`은 Static synchronous field에서
  `covariance_weight * covariance + keys @ keys.T`를 만들 때 추가 `2.67 GiB`를 요청하며
  CUDA OOM으로 종료됐다. 당시 GPU total `47.40 GiB`, free `1.93 GiB`, PyTorch allocated
  `43.04 GiB`, reserved-unallocated `2.07 GiB`로 보고됐다.
- Llama job `16025`는 Full warm-up rep-0에서 `ActuatorDirectionalHook`과 locked
  `epsilon=1e-3` one-sided scalar finite difference가 tolerance 밖이라
  `MethodContractError`로 종료됐다.
- Llama stderr에는 per-layer hook/finite-difference 수치가 없고, 두 실행 모두 compute,
  controller, evaluation JSONL과 terminal manifest가 완성되지 않았다.
- evaluation/generation은 실행되지 않았고 scientific outcome은 `0`건이다.

### GH 추정

- Qwen 실패는 dense system 자체의 수학이나 solver가 아니라, 같은 크기의 dense temporary가
  동시에 살아 있는 조립식 때문에 peak memory가 높아졌을 가능성이 크다. 이는 아직 SH의
  alias/lifetime test 전 추정이다.
- Llama 실패는 기존 float64 CPU oracle에서 hook contraction이 dense reference와 일치했다는
  사실과, BF16 output에 `1e-3` 크기의 one-sided perturbation을 넣었다는 구현을 함께 보면
  fixed finite difference의 representability/quantization 문제일 가능성이 있다. 실제 mismatch
  행이 남아 있지 않으므로 hook 구현이 옳다고 확정할 수는 없다.
- 두 blocker 모두 controller의 과학적 정의를 바꾸지 않고 고칠 가능성이 있지만, common-code
  identity를 다시 검증하기 전에는 paired retry를 열 수 없다.

### 사용자 확인 필요

- 현재 없음. 이번 단계는 기존 P0 authority 안의 기술 RCA이며 GPU/Slurm은 닫혀 있다.
- solver, dtype, covariance 정의, controller 또는 모델별 parameter 변경이 필요해지면 기존
  P0 repair 범위를 넘어가므로 사용자 확인이 필요하다.

## 모델별 판정

| 모델 | 통과한 마지막 공통 gate | terminal blocker | 과학적 해석 |
|---|---|---|---|
| Llama3-8B-Instruct | load, manifest, event identity, Native/Static/One-refresh entry | hook 대 fixed FD identity 실패 | 불가 |
| Qwen2.5-7B-Instruct | load, manifest, event identity, Native entry | dense system assembly CUDA OOM | 불가 |

`eff`, `gen`, `loc`, retention, capacity와 `Full/Native` compute ratio는 산출되지 않았다. 빈
JSONL이나 partial direct-z artifact를 metric 결과로 취급하지 않는다.

## 승인된 repair 방향

### Qwen memory-only repair

동일한 dense system과 `torch.linalg.solve`를 유지한다. `covariance.clone()` 하나를 destination
buffer로 삼아 `mul_`과 `addmm_`로 조립하고, covariance GPU tensor가 layer를 넘어 불필요하게
누적되지 않도록 transient lifetime을 닫는다. Woodbury/iterative solve, dtype 변경, artifact
재계산이나 모델별 경로는 허용하지 않는다.

필수 gate는 기존 식과 repaired 식의 numerical identity, input alias 불변, solver 입력의
symmetry/dtype/device identity, exception cleanup, peak-allocation 감소와 전체 CPU regression이다.

### Llama hook-reference repair

primary hook을 결과 없이 바꾸거나 tolerance를 느슨하게 하지 않는다. 먼저 BF16 synthetic
fixture에서 fixed `1e-3` one-sided FD가 실제 perturbation을 표현하는지 분리 측정한다.

선호 hard oracle은 target dense gradient를 만들지 않는 model-common scalar-gate autograd다.
각 layer low-rank direction에 float32 scalar를 두고 forward에서는 output dtype으로 cast한 뒤
`alpha=0`에서 combined event의 `dPhi/dalpha`를 한 번에 얻는다. 이 forward graph가 실제로
동일 dtype/state를 보존하고, primary captured-activation contraction과 독립 경로를 이룬다는
CPU oracle 및 negative-control 검증이 선행돼야 한다. 기존 fixed FD는 hard gate가 아니라
precision diagnostic으로만 남길 수 있다.

## 재제출 gate

다음이 모두 충족될 때만 GH가 별도 paired P0 retry envelope를 발행한다.

1. 두 repair가 같은 model-independent backend에 있고 모델 alias 분기가 없다.
2. dense system 수학, solver, dtype, covariance와 proposal 값이 보존된다.
3. scalar-gate reference가 hook sign/orientation 오류 negative control을 검출하고 target
   parameter `.grad`, pointer, version과 requires-grad state를 바꾸지 않는다.
4. fixed `1e-3` 실패의 원인이 수치로 식별되거나, 식별 불가능하면 독립 hard oracle이
   fail-close한다.
5. 전체 focused CPU suite, warnings-as-errors, compile, access, diff gate가 통과한다.
6. 두 repair의 추가 reference cost와 peak memory가 별도 counter/timer로 계측된다.
7. 기존 partial output은 보존되고 새 retry는 새 output root를 사용한다.
8. SH1 failure report와 post-run audit가 raw artifact hash 및 broadcast/exception 상태를
   기록한다.

Retry가 열리더라도 동일 case `2022`, seed `17`, 동일 arm/repetition의 **technical P0 전체
pair**만 다시 수행한다. 한 모델만 살리는 설정, 성공한 partial arm 재사용, scientific
evaluation 추가는 허용하지 않는다.

## Claim boundary

이번 실행으로 확인된 것은 두 모델 모두 offline load와 초기 event batching까지 도달했다는 것,
그리고 현재 P0 구현에 model-scale blocker 두 개가 있다는 것뿐이다. ODE-Edit의 correctness,
compute 감소, preservation, MEMIT 대비 우위, 두 모델 공통 작동은 확인되지 않았다.

Raw logs와 partial artifacts는 server1 ignored `local/` 아래에 보존 중이라고 SH1이 보고했다.
정확한 hash와 ordinary artifact broadcast 또는 예외는 SH1 terminal failure report에서 닫아야
한다.

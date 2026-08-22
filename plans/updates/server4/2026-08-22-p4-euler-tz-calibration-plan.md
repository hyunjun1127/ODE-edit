# P4 Euler `T_z` calibration plan — server4

- instruction: `ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1`
- owner: `server4-server-head`
- session: `codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d`
- status: `PENDING_GH_APPROVAL / GPU_SLURM_HOLD`
- authoritative contract SHA256:
  `1899bbfa3610ee2edc94044548475dbb39353ebb7aeafeea371c9a879e7d4398`

## 1. 목적과 선택 경계

목표는 raw, unnormalized field를 쓰는 projected explicit Euler의 공통
target pseudo-time `T_z`를 Llama/Qwen에 하나로 고정하는 것이다. Adam learning
rate, held-out Eff/Gen/Loc, Native endpoint, final performance, A±−A+ 효과는
선택 입력으로 사용하지 않는다.

선택 단위는 기존 sealed stream의 `B1_CASE01` train context만이다.

- stream root:
  `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6`
- order SHA:
  `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- held-out evaluator access: 0
- Native access: 0
- sample 재추출·재배열: 0
- `B1_CASE01`은 `PERMANENT_CALIBRATION_ONLY`로 표시하고 confirmatory
  A±−A+ denominator에서는 제외한다. 최종 denominator는 GH 승인 시
  B2–B10의 9개 independent slice로 고정한다.

마지막 항목은 전체 B10 denominator를 유지하는 것보다 calibration leakage를
차단하는 선택이다. GH가 별도 calibration-only sealed unit을 제공할 경우에만
그 unit으로 대체하고 B1을 confirmatory denominator에 복귀시킨다.

## 2. 공통 실행 입력

Stage 1과 Stage 2의 모든 cell에 다음을 동일하게 결속한다.

- models: `llama3-8b-inst`, `qwen2.5-7b-inst`
- arms: Euler Z+, Euler Z±
- `W=W0`, writer call 0, outer K8 repeat 0
- 각 solve는 fresh `z0=y0`, warm start 0
- 동일 request/order/train-context/KL teacher/origin/radius
- 기존 모델별 pinned KL/decay/clamp factor만 사용
- full FP32 tensor-state/solver, autocast/quantization off
- optimizer/Adam/SGD/state/moment/backward 0
- microstep당 aggregate objective에 `torch.autograd.grad` 정확히 1회
- final iterate only
- model별 `h`, `T_z`, 선택 규칙 branch 0

## 3. Stage 1 — fixed-`h` strength/pseudo-time curve

### 후보와 근거

공통 fixed-`h` grid를 다음처럼 선고정한다.

| ordinal | `h` |
|---:|---:|
| 1 | 0.0625 |
| 2 | 0.25 |
| 3 | 1.0 |
| 4 | 4.0 |

이는 raw-gradient 자연 단위 `h=1`을 중심으로 한 factor-4 ladder다. 기존 Adam
lr 0.1/0.5를 재사용하지 않는다. immutable Adam train telemetry에서 첫 상태의
request 평균 total-gradient norm은 Llama 약 1.37–1.97, Qwen 약
0.172–0.266이었다. 따라서 이 grid는 양 모델에서 명백한 weak 구간부터 Llama의
projection-boundary domination이 예상되는 상단까지 bounded하게 덮는다. 이
수치는 Adam step 크기를 Euler step으로 변환한 것이 아니라 raw field의 관측
scale만 bracket 근거로 사용한 것이다.

### 곡선

각 `h`에서 `M={1,3,5,10}`을 모두 실행한다. 한 곡선 안에서 `h`는 고정되고
`T_z=Mh`이므로, 이 비교는 numerical refinement가 아니다.

> `M={1,3,5,10}` at fixed `h` = target strength/pseudo-time curve.

각 model×arm×h×M에 다음 train-only 값을 기록한다.

- new/old NLL과 new−old margin의 request median/p90/max
- raw field norm의 origin/step/final 값과 slowing ratio
- raw Euler 및 누적 target displacement
- clamp hit fraction, ratio, removed norm/energy
- nonfinite, W content hash, Adam/optimizer/parameter-grad count
- logical field evaluation, actual forward/autograd.grad, suffix/KL token 및
  wall/GPU/peak-memory count

### 사전 고정 선택 규칙

먼저 네 `M` 곡선을 모두 종료한 후에만 `M=5` row를 `T_z` 선택 후보로 본다.
`h`는 다음을 양 모델·양 arm 모두 만족할 때만 admissible이다.

1. 모든 target/objective/field/update가 finite.
2. W content hash가 entry/exit에서 exact match.
3. optimizer/Adam/SGD/moment/parameter grad/backward count가 전부 0.
4. 각 model×arm curve의 clamp-hit fraction이 0.5 미만.
5. 동일 fixed-`h`에서 median displacement가 `M=1→3→5→10`으로
   nondecreasing. 허용치는 상대 `1e-5`뿐이다.
6. barrier 결과에 맞춘 선택을 피하기 위해 strength adequacy는 Euler Z+만
   사용한다. 양 모델 모두에서 M10 median train new NLL이 M1보다 절대
   `1e-4` 초과 악화하지 않아야 한다. Z± new/old/margin은 observation-only다.

가장 큰 admissible `h`를 선택한다. 그 다음 큰 `h`는 inadmissible이어야 하며,
이 둘을 safe/unsafe 선택 구간으로 기록한다. 그 후:

`T_z = 5 * h_selected`

로 하나의 공통 horizon을 잠근다. admissible 후보가 없거나 grid 상단 `h=4.0`도
admissible이면 닫힌 bracket이 없으므로 grid를 즉석 확장하지 않고 HOLD한다.

## 4. Stage 2 — fixed-`T_z` M5/M10 refinement

Stage 1에서 잠근 하나의 `T_z`에 대해 다음 두 endpoint만 비교한다.

- M5: `M=5`, `h=T_z/5`
- M10: `M=10`, `h=T_z/10`

이 단계만 numerical refinement다. request별 endpoint discrepancy는:

`d_i = ||z_i,M5 - z_i,M10|| / max(||z_i,M10-y_i||, 1e-12)`

로 계산한다. 두 모델·두 arm 각각에서 다음을 모두 만족해야 한다.

- request median `d_i <= 0.10`
- request p90 `d_i <= 0.25`
- request max `d_i <= 0.50`
- clamp-hit fraction `<0.50`
- finite/W-freeze/Adam0 gate 유지

train new/old NLL, margin, field norm, displacement, clamp 차이도 함께 보고하지만
위 endpoint 수치와 safety gate 외에는 선택 영향이 없다. M5/M10의 held-out,
Native, final W 성능은 열지 않는다.

## 5. 즉시 HOLD 조건

다음 중 하나면 `T_z` lock과 ZA 제출을 열지 않는다.

- 어느 model/arm/step에서든 nonfinite
- W content hash 변화 또는 parameter gradient 발생
- optimizer/Adam/SGD/moment/backward count 비영
- 어떤 curve든 clamp-hit fraction `>=0.50`
- Stage 1 safe/unsafe bracket 미형성
- M5/M10 endpoint discrepancy가 median/p90/max threshold 중 하나라도 초과
- Llama/Qwen에 다른 `h`, `T_z`, 선택 분기 필요
- held-out/Native/final performance/barrier causal delta의 선택 영향 발생

HOLD 후 grid·threshold·sample을 같은 결과를 보며 확장하거나 재튜닝하지 않는다.
새 후보가 필요하면 별도 GH 승인 numerical-lock revision으로만 진행한다.

## 6. 승인 후 산출물

승인 후 calibration 실행은 별도 Euler namespace에서 cap2를 지키며 Llama/Qwen
두 cell로 수행한다. 산출물에는 fixed-`h` strength curve임을 명시한 Stage 1
receipt, fixed-`T_z` refinement인 Stage 2 receipt, 공통 `T_z/M/h` numerical
lock, source/session/stream/HF/EasyEdit identity를 결속한다. 승인 전 model load,
GPU, Slurm submit은 0으로 유지한다.

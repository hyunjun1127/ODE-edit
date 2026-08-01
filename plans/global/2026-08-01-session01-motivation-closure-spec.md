# Motivation closure: adaptive direction gate와 AlphaEdit projector transfer

날짜: 2026-08-01

세션: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`

방법론 명칭: **ODE-Edit**

이 문서는 결과를 보기 전에 고정하는 Motivation 실험 계약이다. 큰
baseline sweep이나 paper claim 확정이 아니라, 직전 `qstep4`에서 드러난
Llama/Qwen 차이를 outcome-free controller로 설명·개선하고 같은 원리가
AlphaEdit projector를 합성한 actuator에도 남는지 확인한다.

## 네 범주의 출처

- proposal에서 온 내용:
  layer utility heterogeneity(H1), state에 따른 ranking non-stationarity(H2),
  useful update의 layer concentration(H3), static direction/scaling 대비
  path 중 relinearization(RQ2), 그리고 AlphaEdit null-space proposal로의
  transfer(RQ4)를 Motivation에서 진단한다. edit-time controller는 rewrite
  request와 허용된 MEMIT context만 볼 수 있고 evaluation outcome은 볼 수
  없다.
- repo/protocol에서 확인한 사실:
  직전 fresh `qstep4`의 full native distance `D`를 `D/4` 네 번 적용했을 때
  unconditional direction refresh 효과는 Llama에서 약 `+0.0370`, Qwen에서
  약 `+1.6590`이었다. coefficient refresh 효과는 각각 약 `+0.1249`,
  `+0.8742`였다. exact native split noise는 약 `1.1e-6`이고, step 4의 W0
  direction cosine은 약 `0.62`까지 변했다. EasyEdit에는 두 fixed model의
  projector와 Wikipedia covariance가 이미 존재하며 repo manifest에
  SHA-256/size가 고정돼 있다. protocol은 SH 부재와 사용자 명시적
  time-critical 요청에서 GH 최소 직접 제출을 허용한다.
- GH 추정:
  Llama에서는 매 hop 무조건 방향을 바꾸는 비용/오차가 이득보다 클 수
  있으므로, 같은 현재 state에서 refreshed와 transported-W0 후보의 작은
  central probe를 비교해 필요한 hop만 refresh하면 Llama의 손실을 줄이고
  Qwen의 큰 refresh 이득은 유지할 수 있다. projector 합성 뒤에도 state
  dependence가 남는다면 동일 gate가 stale projected direction보다 나을 수
  있다.
- 사용자 확인 필요:
  없음. 두 fixed model, 동시 pair 실행, EasyEdit read-only hook, precomputed
  cache 재사용, AlphaEdit ODE-style transfer, lenient gate, 빠른 제출·분석을
  사용자가 명시했다.

## 왜 proposal이 최종 paper plan이 아닌가

proposal의 ODE 관점은 출발 가설이다. 현재 증거는 CounterFact 소수 사례의
teacher-forced rewrite utility와 MEMIT-derived actuator에 한정되고,
generalization/locality/large-batch/sequential editing 또는 다른 benchmark
우위를 아직 증명하지 않는다. 이번 결과가 양성이어도 `ODE-Edit가
AlphaEdit보다 우월하다`거나 projector가 exact invariant를 보장한다는
claim으로 확장하지 않는다.

## 고정 실험

### 모델·사례

- model: `llama3-8b-inst`, `qwen2.5-7b-inst`만 사용
- model별 fresh CounterFact 8건
- canonical salted rank: `[124:132]`
- case IDs: `13217, 18009, 14146, 20773, 15772, 7579, 17129, 19476`
- selection manifest:
  `2d0d98b48692e82fa04efaa8df97863c48202bc23818e38cf2bc1a2548c81cf7`
- selected order hash:
  `11a7c5aab40b9fbf571c38589621a48f7ec0e9ca0e0aa71c0e380e67e530358d`
- prior evidence `[0:124]` order hash:
  `dc170655b14c5e3d4a94e31b8762da8c319013d102d59533cd7d96ac00dcf81a`

### 두 editor track

1. `memit`: existing synchronous/ordered MEMIT factor proposal를 ODE-Edit
   trajectory actuator로 사용한다.
2. `alphaedit_projected`: 같은 MEMIT factor proposal `B=L R^T`에 EasyEdit의
   **기존 고정 projector** `P`를 ODE-edit hook에서 `B P = L(P^T R)^T`로
   합성한다. projector는 mmap/read-only로 열고 기존 SHA-256/size를
   검증하며 계산·갱신·저장을 금지한다.

두 번째 track은 projector-composed proposal에 ODE-style controller가
transfer되는지를 보는 controlled diagnostic이다. EasyEdit의 native
`apply_AlphaEdit_to_model` 전체 구현을 재현했다는 뜻이 아니며,
`alphaedit_projected`라는 이름을 유지한다. direct-z는 case/track마다 한 번
계산해 네 경로에 고정하고, Wikipedia covariance는 기존 verified NPZ만
읽는다.

### 거리와 경로

- 기준 거리: 해당 track의 ordered full proposal C-distance `D`
- hop: 정확히 4개, 각 `D/4`; per-hop C-energy `E_native/16`
- central probe: `D/64`, 각 action의 `+/-` 대칭 평가
- layer: `4,5,6,7,8`
- step 1은 모든 정책이 동일한 W0 score-mix action
- 최종 arm:
  - `G4`: adaptive gate
  - `A4`: direction+coefficient를 매 hop 무조건 refresh
  - `B4`: W0 direction을 transport하고 coefficient만 refresh
  - `C4`: W0 direction+coefficient 모두 고정
  - no-op, ordered full, exact split4 controls

G는 step 2--4마다 **현재 G state 하나**에서 refreshed proposal와 W0
transport proposal를 동일한 `D/64` panel로 평가한다. predicted score는
positive per-layer central slope의 L2 norm이다. normalized advantage
`(score_refresh-score_fixed)/max(abs(score_refresh),abs(score_fixed),1e-12)`가
`0.02`를 초과할 때만 refresh하고 tie는 fixed로 보낸다. `0.02`는 이전
qstep의 probe panel만 이용해 정한 calibration 값이며 새 8건 outcome으로
바꾸지 않는다.

case당 controlled NFE는 `164`다: baseline 1, probe panel 13개 × action
6개 × 양·음 2, 최종 arm 7개. proposal build는 8회, direct-z는 1회다.

## 최소 technical gate

- fresh selection이 `[0:124]`와 disjoint
- fixed model/revision/layer/hparams/covariance/projector hash 일치
- offline, EasyEdit/data/cache/projector write 없음
- exact frozen target/direct-z lineage와 target token identity 유지
- G의 두 후보가 매 hop 동일 current G state에서 평가됨
- 각 hop C budget, common first hop, branch order, receipt-before-outcome
  정확
- no-op 및 native full/split4 control 성공
- 모든 temporary edit 뒤 exact W0/RNG rollback
- 8/8 case, exact stream count; partial-case scientific rescue 금지

하나라도 실패하면 해당 model/track의 scientific 해석을 중단한다.

## 사전 고정 primary contrast

- direction value: `A4-B4`
- coefficient value: `B4-C4`
- adaptive value: `G4-A4`, `G4-B4`, `G4-C4`
- native-gap reduction: `abs(A4-native)-abs(G4-native)`
- gate behavior: model/track별 refresh 선택률 및 step별 선택률
- geometry: W0 대비 current refreshed C-cosine
- projector transfer: `alphaedit_projected`의 `G4-B4`, `G4-C4`와
  right-factor norm retention

case 평균, paired bootstrap 95% CI(`seed=20260803`, 4,000 resamples)를
함께 보고하되 Motivation 판정은 아래 lenient point-estimate gate를 먼저
사용한다. CI는 불확실성 표시이며 소규모 진단의 자동 kill로 쓰지 않는다.

## lenient scientific gate

다음은 방향 일치 여부를 보는 Motivation gate이며 paper-level 유의성 기준이
아니다.

2026-08-01 실행 중 첫 outcome 전 사용자 지시를 반영해 model별 성공 규칙을
두지 않는다. 두 model은 같은 code path와 exact 동일 hyperparameter
(`K`, hop/probe fraction, layer, score 식, margin)를 사용해야 하며 model
alias에 따른 method branch나 threshold는 금지한다. 동일 함수가 서로 다른
state/probe를 입력받아 다른 `refresh/fixed` 선택을 내리는 것은 허용되는
data-dependent 출력이지 별도 method가 아니다.

1. 동일-method gate:
   두 manifest의 controller constant/config가 exact 동일하고 model별
   override가 없어야 한다.
2. model 공통 broad benefit:
   **각 model 모두** 아래 중 하나 이상을 만족해야 한다.
   `mean(G4-A4) > 1e-4`, `mean(G4-B4) > 1e-4`, 또는
   `mean(abs(A4-native)-abs(G4-native)) > 1e-4`.
3. model 공통 non-collapse:
   각 model에서 `mean(G4-max(A4,B4)) >= -0.10`이어야 한다. 이 식의
   `max`는 case별 outcome 선택이 아니라 사후 진단용 case별 상한 비교이며
   operational policy에는 들어가지 않는다.
4. mechanism evidence:
   refresh 선택률과 step별 선택은 model별로 보고하되 서로 달라야 한다는
   조건을 성공 gate로 사용하지 않는다.
5. AlphaEdit projector transfer:
   exact 같은 controller로 **각 model 모두** `mean(G4-B4) > 1e-4` 또는
   `mean(G4-C4) > 1e-4` 중 하나를 만족하고, model별
   `mean(G4-max(A4,B4)) >= -0.10`이어야 한다.

1--3을 만족하면 MEMIT Motivation은 `lenient-pass`, 1과 5를 만족하면
AlphaEdit transfer는 `lenient-pass`다. 둘 다 통과하면 이번 Motivation을
positive closure하고 다음 baseline/benchmark 세션으로 넘긴다. 일부만
통과하면 살아남은 claim만 좁혀 `partial-pivot`한다. 한 model만을 위한
threshold/policy rescue는 허용하지 않는다.

## kill 및 pivot 조건

- technical gate 실패, cache/projector 재계산 또는 evaluation leakage:
  즉시 kill, scientific result 사용 금지
- 두 model에서 `A4-B4`와 `B4-C4`가 모두 practical floor 이내이고
  direction cosine이 거의 1로 유지: path non-stationarity claim kill
- G가 두 model의 A/B보다 동시에 악화하고 architecture-adaptive 선택도
  나타나지 않음: learned/adaptive refresh Motivation kill; static 또는
  coefficient-only controller로 pivot
- projected proposal가 0/non-finite로 붕괴하거나 Alpha transfer gate 실패:
  AlphaEdit generality claim만 kill하고 MEMIT 결과와 분리
- full/split4 control 차이가 practical effect와 같은 규모:
  implementation/numerics confound로 전체 scientific 해석 보류

## 기대효과 예측과 경계

이전 qstep panel을 outcome 없이 재조합한 calibration forecast에서는 Llama
G가 A보다 약 `+0.15` 높고 native gap을 약 `0.10`으로 줄이며, Qwen G는
A보다 약 `-0.04` 이내에서 B보다 약 `+1.62` 높을 것으로 예상됐다. 이는
경로가 실제로 G 선택을 따라 진행된 결과가 아닌 **stitched forecast**이므로
이번 fresh-case 성과로 취급하지 않는다. AlphaEdit projector transfer의
효과 크기는 사전 증거가 없어 방향만 예측하며 숫자 forecast를 만들지
않는다.

## 실행·자원·artifact 경계

- pair 하나: `2 GPU / 16 CPU / 130000M / 12:00:00`
- child 하나: `1 GPU / 8 CPU / 65000M`
- server1 GPU cap 3; 두 track pair를 동시에 제출하지 않는다.
- 먼저 MEMIT pair, technical complete 뒤 Alpha pair를 제출한다.
- raw/log/direct-z/receipt는 `local/`만; Git에는 compact summary/report/hash만
- active peer SH/clone이 없으므로 이번 artifact broadcast는 no-peer 예외를
  완료 보고에 남기고 임의 SSH/rsync를 하지 않는다.
- GH 직접 제출 예외 사유: SH 부재와 사용자의 빠른 동시 pair 제출 명시.
  영향은 server1의 위 두 one-shot pair와 명시된 ignored local path뿐이다.

각 track 완료 후 raw를 보지 않은 별도 Terra Ultra agent가 model별 report를
작성해야 한다. runtime metadata가 `gpt-5.6-terra`, `ultra`로 확인되지 않은
agent 결과는 report 승격에 사용하지 않는다.

## Zero-outcome execution amendment

original MEMIT v1 job `15730`은 첫 case의 outcome/feature/action 생성 전에
gate path가 새 lineage label을 사용해 existing quarter-step allowlist에
거부됐다. 양 model outcome은 0건이다. scientific contract는 바꾸지 않고
gate 선택을 existing `fixed_step_i/refreshed_step_i` lineage label에
매핑한 v2 identity로 재실행한다. v1 raw/log/marker는 보존한다.

v2 job `15731`은 lineage를 통과한 뒤 첫 action commit에서 key 이름에
`outcome`이 포함되어 generic firewall이 거부했고, 양 model outcome은 다시
0건이다. firewall은 완화하지 않는다. key를 outcome-free 이름으로 바꾸고
reusable commit helper에 adaptive 7-arm/event/receipt envelope를 명시하는
v3 identity로 재실행한다. scientific controller와 판정 기준은 불변이며
v1/v2 raw/log/marker를 모두 보존한다.

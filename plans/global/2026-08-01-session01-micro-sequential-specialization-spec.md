# Session 01 Motivation — 4-edit preservation specialization diagnostic

날짜: 2026-08-01

GH session: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`

방법명: **ODE-Edit**

상태: **conditional precommit; atomic MEMIT/Alpha diagnostics technical-valid 후에만 실행**

## 목적과 claim boundary

이 진단은 ODE-Edit가 native sequential MEMIT보다 짧은 누적 편집에서 과거
edit와 model behavior를 보존할 가능성이 있는지 보는 Motivation signal이다.
4 edits는 lifelong scale이 아니므로 결과가 양성이어도 sequential-collapse
해결, long-horizon retention, downstream capability, accuracy 또는 method
superiority를 주장하지 않는다.

### Proposal에서 온 내용

- 동일 direct-z의 parameter realization은 하나가 아닐 수 있다.
- cumulative capacity와 layer load concentration을 trajectory state로 보고,
  current-state proposal을 다시 평가해 write를 재배분한다.
- H3/H4/H5는 cumulative load concentration, matched-efficacy capacity,
  long-horizon retention을 예측하지만 proposal 자체는 검증된 claim이 아니다.

### Repo/protocol에서 확인한 사실

- fixed model은 `llama3-8b-inst`, `qwen2.5-7b-inst`다.
- adaptive controller는 양 모델에 동일 code/config를 사용하며 `K=4`, hop
  `D/4`, central probe `D/64`, gate margin `0.02`로 고정돼 있다.
- EasyEdit source는 read-only이고 Wikipedia covariance와 AlphaEdit projector는
  pinned precomputed artifact로만 사용할 수 있다.
- CounterFact evaluation prompt는 action/stopping/controller에 사용할 수 없다.
- raw prompt, logits, generations, direct-z, weights는 ignored `local/`만 허용된다.

### GH 추정

- 4-edit chain은 collapse를 증명하기에는 너무 작지만, prior-edit retention,
  held-out neighborhood KL, cumulative C-load가 native와 다른 방향으로
  움직이는지 확인하는 최소 specialization signal로는 사용할 수 있다.
- atomic adaptive gate가 양 모델에서 technical-valid하지 않으면 cumulative
  branch를 여는 것은 원인 분리를 해치므로 실행하지 않는다.

### 사용자 확인 필요

- 없음. 사용자는 Motivation 규모의 작은 signal로 gate를 판단하고, 양 모델에
  동일 method를 적용하며, sequential collapse/model preservation 같은 특정
  specialization을 검토하도록 명시했다.

## 고정 data/model envelope

- models: `llama3-8b-inst`, `qwen2.5-7b-inst`
- chain length: 정확히 `4` atomic edits/model
- canonical salted rank: `[140:144]`
- order: salted rank order 그대로; model 간 동일
- prior Session evidence 및 별도 diagnostic에 사용된 `[0:140]`과 disjoint
- edit layer: `[4,5,6,7,8]`
- fixed seed: `37`
- pair resource: `2 GPU / 16 CPU / 130000M / 12:00:00`
- child resource: `1 GPU / 8 CPU / 65000M`

selection manifest는 구현 preflight에서 source hash, exact case IDs, order hash,
prior `[0:140]` hash를 생성해 Git의 small audit에 고정한다.

## 두 cumulative branch

각 model child 안에서 editable weights의 두 독립 branch를 유지한다.

1. `native_sequential_memit`
   - 각 branch current state에서 EasyEdit의 ordered MEMIT direct-z/proposal을
     계산하고 native full update를 1회 누적한다.
2. `ode_adaptive_gated_memit`
   - 같은 branch current state에서 direct-z를 1회 계산하고, precommitted
     adaptive G controller로 정확히 네 `D/4` hop을 누적한다.

각 branch는 자기 current state에서 editor target을 계산한다. 서로 다른
cumulative state에서 direct-z bytes가 달라질 수 있으며, 이를 model별 method
차이나 target leakage로 해석하지 않는다. 같은 branch/edit 안에서는 direct-z를
한 번만 계산해 모든 hop에 고정한다.

AlphaEdit-projected atomic transfer는 별도 track에서 검증한다. 이 4-edit primary
chain에는 projected branch를 추가하지 않아 compute와 원인 축을 제한한다.

## 동일-method 및 state isolation

- 모델 alias에 따른 branch, threshold, case, stopping, rescue 금지
- 두 model에서 exact same controller constants와 evaluator 사용
- branch editable weight snapshot은 CPU local artifact로 분리하고, branch 전환
  전후 exact hash를 확인한다.
- original W0 editable weights를 immutable anchor로 유지한다.
- branch B의 평가나 결과를 branch A의 다음 action에 사용하지 않는다.
- 한 branch/model 실패 시 해당 pair scientific 해석을 중단하고 sibling을
  fail-fast한다. partial-case rescue는 금지한다.

## Action-before-evaluation firewall

각 edit index `t=1..4`에서 다음 순서를 지킨다.

1. 두 branch의 current-state identity와 direct-z/proposal을 만든다.
2. 두 branch action hashes, cumulative state parents, budgets를 feature/action
   stream에 기록하고 fsync한다.
3. exclusive receipt를 만든다.
4. receipt 뒤에만 current edit, prior edits, neighborhood/generation prompt를
   평가한다.
5. evaluation scalar는 다음 edit의 controller, scaling, stopping, retry에
   들어가지 않는다.

Controller가 볼 수 있는 것은 current rewrite request, 기존 MEMIT 5 contexts,
current weights, pinned covariance뿐이다.

## 고정 평가

checkpoint `t=0..4`와 branch별로 아래 scalar만 남긴다.

### Edit quality

- current edit teacher-forced utility/NLL/min margin/all-token top-1
- 지금까지 편집된 `1..t` request의 teacher-forced utility
- prior-only `1..t-1` mean utility와 success rate
- edit age별 utility; raw logits/token IDs는 저장하지 않음

### Evaluation-only preservation proxy

- 각 edit의 first 4 neighborhood prompts에서 W0 대비 next-token KL
- 각 edit의 first 4 generation prompts에서 W0 대비 next-token KL
- target-true teacher-forced NLL delta
- prompt contents는 local memory에만 존재하고 JSON/Git에 저장하지 않음

### Cumulative geometry

원 W0 대비 각 layer cumulative displacement `Delta_l`에 대해:

`Psi_l = tr(Delta_l C_l Delta_l^T) / (tr(W0_l C_l W0_l^T)+eps)`

- `sum(Psi_l)`, max-layer share, layer Gini
- cumulative Frobenius norm와 native-distance-equivalent path length
- current edit compute: wall time, proposal builds, controlled NFE

`Psi`, KL, short-chain retention은 proxy이며 capability나 causal preservation으로
정의하지 않는다.

## Primary comparisons

모든 차이는 `ODE - native`로 정의하되 KL/load는 낮을수록 좋음을 별도
표기한다.

1. `final_prior_retention_delta`: t=4 prior-only mean utility 차이
2. `retention_auc_delta`: t=1..4 all-edited mean utility AUC 차이
3. `current_edit_noncollapse`: t=1..4 current utility 차이 평균
4. `neighborhood_kl_reduction`: native KL - ODE KL
5. `generation_kl_reduction`: native KL - ODE KL
6. `capacity_reduction`: native sum(Psi) - ODE sum(Psi)
7. `max_layer_share_reduction`: native - ODE

case/batch가 네 개뿐이므로 CI는 descriptive bootstrap으로만 표시하고 자동
kill에 사용하지 않는다.

## Lenient common-model gate

technical gate는 strict, scientific gate는 Motivation 규모에 맞게 lenient다.

1. `same_method`: 두 model manifest/controller/evaluator constants exact 동일
2. `current_noncollapse`: 각 model에서 mean current-edit `ODE-native >= -0.10`
3. `common_specialization`: 아래 동일 축 중 하나가 **양 모델 모두** 양수
   - `final_prior_retention_delta > 1e-4`
   - `retention_auc_delta > 1e-4`
   - `neighborhood_kl_reduction > 1e-6`
   - `generation_kl_reduction > 1e-6`
   - `capacity_reduction > 1e-6`
4. `no_hidden_tradeoff`: 선택된 공통 축을 보고한 뒤 다른 모든 불리한 primary
   metric도 함께 공개; metric을 case별로 골라 성공시키지 않음

1--4를 만족하면 `microseq_lenient_signal`, technical-valid하지만 공통 축이
없으면 `microseq_no_common_signal`, current noncollapse가 깨지면
`microseq_harm_signal`이다.

## Technical kill conditions

- model/config/case/order/branch mismatch
- direct-z per branch/edit 1회 위반
- covariance miss/download/recompute 또는 EasyEdit write
- evaluation prompt가 receipt 전 load되거나 controller에 유입
- branch state hash collision, W0 anchor mutation, rollback/snapshot failure
- exact `4 edits × 2 branches × 2 models` stream count 불일치
- non-finite scalar, missing failure case, raw artifact Git 유입
- Slurm/resource/session boundary 위반

technical kill이면 scientific 수치를 해석하지 않는다.

## Next-stage boundary

- 양성 결과: larger but still bounded 16–32 edit replication을 별도 session에서
  검토할 수 있다. lifelong/full CounterFact로 바로 확장하지 않는다.
- 무신호: sequential specialization claim을 닫고 atomic routing signal만 남긴다.
- 해로움: cumulative ODE controller를 kill하고 Motivation을 종료한다.


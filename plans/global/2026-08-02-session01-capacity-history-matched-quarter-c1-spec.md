# Motivation — capacity/history matched-quarter c1 실행 계약

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 상태: implementation lock; 실행 전
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- 비교 family: MEMIT, canonical-history AlphaEdit
- 공통 policy: `capacity-qp-history-k4-native-progress-v2`

## 결론과 재실행 이유

c0는 방법의 유효한 음성 실험이 아니라 **under-edit를 허용한 구현 계약의 실패**다.
코드상 QP는 최대 세 번만 실행되어 nominal 상한도 `3D/4`였고, 각 회차에는
reachable progress의 `0.75`만 요구했다. 더구나 ordered native 수준이 아니라
`exact-top1`에서 종료했다. 실제 accepted path/native-distance 비율 평균은 Llama
MEMIT `0.454052`, Llama Alpha-history `0.457809`, Qwen MEMIT `0.399978`, Qwen
Alpha-history `0.409667`이었다. 따라서 c0의 낮은 capacity/KL과 낮은 efficacy는
capacity routing 효과가 아니라 edit strength mismatch로 설명된다.

c1은 같은 작은 4-edit Motivation panel을 유지하면서 이 confound만 제거한다.

## 네 범주

### Proposal에서 온 내용

- 같은 direct-z를 실현하는 layer actuator를 current state에서 동기적으로 다시
  계산하고, rewrite progress 대비 cumulative capacity cost가 낮은 layer로 배분한다.
- Controller는 rewrite request와 MEMIT authorized prefixes만 보며 evaluation prompt는
  보지 않는다.
- MEMIT proposal과 AlphaEdit projected/history proposal에 동일 controller가 작동해야
  한다.
- First-hitting terminal과 trust ratio는 edit strength를 줄이기 위한 장치가 아니라
  rewrite goal을 만족한 뒤 불필요한 write를 막기 위한 장치다.

### Repo/protocol에서 확인한 사실

- EasyEdit source는 read-only이며 ODE-edit-side hooks만 수정한다.
- model, case order, seed `41`, layers `4--8`, direct-z-once, covariance/projector/history
  artifact identity는 기존 capacity/history contract와 동일하다.
- AlphaEdit는 precomputed mmap projector와 기존 Wikipedia covariance를 read-only로
  사용하며 재계산하지 않는다.
- server1 active project cap은 GPU `4`, memory `260000M`이다.
- 현재 server1 SH가 없으므로 사용자가 승인한 GH time-critical direct-submit 예외를
  사용하고 명령·영향을 audit에 기록한다.

### GH 추정

- c0 harm의 주원인은 QP algebra 자체보다 fractional progress와 weak terminal의 결합이다.
- native 수준 rewrite progress를 맞춘 뒤에도 capacity/preservation signal이 남는지가
  Motivation에서 확인 가능한 최소 신호다.
- 4 edits는 lifelong capacity claim을 검증하지 못한다. Hard overload가 관측되지 않더라도
  soft cumulative-cost allocation은 진단할 수 있지만, overloaded-layer rerouting claim은
  열지 않는다.

### 사용자 확인 필요

- 없음. 사용자가 구현 수정과 재실험을 명시적으로 지시했다.

## c1 controller 계약

각 branch/edit의 current state에서 다음 순서를 고정한다.

1. direct-z를 한 번 계산하고 ordered native proposal `B_native`를 만든다.
2. `B_native`를 temporary exact application하여 authorized rewrite utility
   `U_native`를 얻고 원상복구한다. Evaluation-only field는 아직 decode하지 않는다.
3. QP branch는 최대 `K=4` accepted macro-round를 사용한다. 매 round initial trust cap은
   current edit의 native C-distance `D`에 대해 `D/4`; 실제 trust failure에만 한 번
   `D/8` retry를 허용한다.
4. 매 current state에서 synchronous layer proposal, central-probe slope, cumulative
   `Psi_l`, cross term, barrier cap을 다시 계산한다.
5. QP progress request는 fixed fraction이 아니라
   `max(0, U_native - U_current)` **전체 잔여 gap**이다. Feasible maximum보다 크면
   solver가 trust/barrier frontier까지 사용하고 slack을 기록한다.
6. `exact-top1`은 diagnostic일 뿐 종료 조건이 아니다. Endpoint utility가
   `U_native - 1e-4` 이상일 때만 조기 종료한다.
7. 두 trust attempt가 모두 reject되면 partial endpoint를 성공으로 내보내지 않고
   technical fail-closed한다.
8. 네 accepted round 뒤에도 native reference를 못 맞추면 artifact는 보존하되
   `budget_exhausted=true`인 scientific negative로 판정한다.

Native branch는 ordered proposal을 그대로 한 번 적용하고 reference-matched invariant를
제공한다. 두 모델과 두 editor family에는 byte-identical policy를 사용한다. 모델별
threshold, branch, rescue는 금지한다.

## Gate

### Technical gate

- 8 controller + 8 evaluator 모두 4/4 terminal/pass
- policy/hash/case order/model/layer/seed 동일
- action-before-evaluation firewall, direct-z-once, exact rollback/lineage
- precomputed covariance/projector only, Alpha history append exactly once/edit
- QP accepted diagnostic마다 `requested_gain == remaining_reference_gain_before`
- unmatched QP는 반드시 4 accepted rounds와 `budget_exhausted=true`
- barrier/cap violation 없음, raw prompt/logit/token/NFE artifact 없음

### Lenient Motivation gate

각 family에서 Llama와 Qwen이 모두 다음을 만족해야 한다.

- QP-minus-native current utility mean delta `>= -0.10`
- capacity/concentration 또는 preservation proxy 중 공통 양의 축 최소 하나
- 동일 policy, no model-specific rescue

통과해도 claim은 “4-edit, same-policy, direction-aligned signal”로 제한한다. Lifelong,
general method superiority, deployable preservation guarantee는 열지 않는다.

## Kill / 중단 조건

- startup에서 source/session/Git/resource/artifact identity mismatch
- EasyEdit write 또는 covariance/projector/Wikipedia recomputation 시도
- evaluation field가 controller 이전에 decode됨
- native remaining gap 대신 fixed fractional request가 관측됨
- QP가 native reference 미달인데 4 round 이전에 종료함
- 한 worker가 실패하면 siblings를 종료하고 새 scientific result로 사용하지 않음

## 실행 envelope

- job: `odeedit_capacity_history_pair_c1_v1`
- resource: GPU `4`, CPU `32`, memory `260000M`, server1
- four simultaneous 1-GPU workers: Llama/Qwen × MEMIT/Alpha-history
- controller phase 후 evaluator phase, 이어 model analysis와 pair synthesis
- output: `local/results/raw/session01_motivation/caphist_*_c1_v1/`
- log: `local/logs/slurm/session01_motivation/`
- Git에는 code/test/spec/audit/small summary만 남기고 raw artifact는 `local/`에 둔다.

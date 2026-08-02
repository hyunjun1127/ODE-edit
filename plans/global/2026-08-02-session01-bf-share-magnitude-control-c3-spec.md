# Motivation — BF-share magnitude-only c3 실행 계약

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 상태: implementation/test 완료, 제출 전
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- editor family: MEMIT, canonical-history AlphaEdit
- 공통 policy: `bf-common-frontier-share-radial-exact-quarter-k4-v1`

## 결론과 목적

c1은 BF QP coefficient를 layer 가중치이자 실제 C-distance로 동시에 사용해 Llama
path가 native의 약 67--69%로 줄었다. c2에서 exact `D/4 × 4`를 적용하자 Llama는
크게 회복했지만, c2는 common-frontier/cap 규칙도 함께 바꿔 Qwen의 mixed-negative를
magnitude 또는 share 중 하나에 귀속할 수 없었다.

c3는 **c1 BF layer weighting algorithm을 유지하고 global magnitude만 분리**한다.
이는 새 tuning이나 method 우위 실험이 아니라 사용자의 “낮은 update가 구현 문제”라는
가설을 모델 공통으로 판별하는 마지막 Motivation 원인분리 control이다.

## 네 범주

### Proposal에서 온 내용

- layer velocity는 rewrite slope와 cumulative capacity state에 따라 달리 둔다.
- current state마다 layer-synchronous direction과 weighting을 다시 계산한다.
- MEMIT과 AlphaEdit history/projector에 같은 controller를 적용한다.

### Repo/protocol에서 확인한 사실

- c1의 layer terms는 common frontier를 모든 layer의 coefficient cap으로 썼고, QP가
  full remaining native rewrite gap을 요청했다.
- c1은 QP solution coefficient를 그대로 write해 actual path가 native보다 작아졌다.
- c2는 global magnitude뿐 아니라 non-overloaded layer cap을 full envelope로 바꿨다.
- c2의 c1→c2 current utility 변화는 Llama MEMIT/Alpha `+2.663/+1.358`, Qwen
  `-0.399/-0.430`이다.

### GH 추정

- c1 BF share를 보존한 exact-magnitude 결과가 양 모델에서 회복되면 under-update가
  cross-model 구현 원인이었다고 닫을 수 있다.
- Qwen이 다시 악화하면 Llama의 under-update cause는 유지되지만 현재 BF share의
  model-common Motivation은 성립하지 않는다.

### 사용자 확인 필요

- 없음. 사용자가 구현 오류 판정, 재구현, 실험 제출과 분석을 승인했다.

## c3 고정 수식과 구현

각 QP branch/edit/round에서 c1-compatible BF QP가 coefficient vector `x_BF`를 만든다.

1. common frontier를 모든 layer의 allocation cap으로 사용한다.
2. native endpoint까지 남은 rewrite utility 전체를 QP request로 사용한다.
3. QP는 c1과 동일하게 cap/trust region 안에서 `x_BF`를 계산한다.
4. 실제 write share는 `s = x_BF / ||x_BF||_2`다.
5. 실제 update는 `Delta = (D/4) * sum_l s_l u_l`이며 `u_l`은 unit-C layer
   direction이다.
6. 따라서 nonzero layer ratio와 zero support는 c1 BF 그대로이고, 각 hop distance만
   정확히 `D/4`다. 네 hop 총 path는 `D`다.
7. Native reference를 일찍 통과해도 fixed magnitude control이므로 4 hops를 완료한다.
   남은 gap이 tolerance 이하면 해당 state의 maximum-progress BF share를 사용한다.

Allocation cap은 `x_BF`를 만드는 **share-generator constraint**다. Radial normalization
뒤 coefficient는 그 cap을 넘을 수 있으며, c3는 applied hard capacity barrier를
주장하지 않는다. 이 구분을 숨기지 않고 manifest/feature/evaluator/report에 고정한다.

## Technical gate

- controller 8개 + evaluator 8개 terminal/pass, checkpoint 32개
- QP edit마다 exact 4 hops, hop `D/4`, total path `D`
- `allocation_coefficients`는 c1 common-frontier QP cap/barrier를 정확히 만족
- applied coefficient는 allocation coefficient의 단일 양의 radial multiple
- applied share L2=1, zero support 보존, positive overloaded zero suppression 보존
- Llama/Qwen 및 MEMIT/Alpha에 같은 policy/hash/code/case/layer/seed
- action-before-evaluation, direct-z-once, exact lineage/rollback/replay
- EasyEdit와 precomputed covariance/projector/Wikipedia artifact read-only

## Lenient Motivation gate

4-edit signal이므로 lifelong 수준 절대 우위는 요구하지 않는다.

- cross-model implementation signal: Llama와 Qwen 각각에서 family-average
  `c3 current utility - c1 current utility > 0`이며, 각 모델의 적어도 한 family가
  회복한다.
- direction-aligned viability: 위 recovery와 함께 capacity/KL/concentration 중 하나
  이상의 pair-common positive axis가 남는다.
- strong signal: 양 모델·양 family current delta가 native 대비 `>= -0.10`이면 별도 표기.
- 같은 code라도 한 모델만 회복하면 model-common method signal로 승격하지 않는다.

## Kill / next

- exact magnitude/radial proportionality/model-common identity가 깨지면 technical kill.
- Qwen family-average가 c1보다 회복하지 않으면 cross-model BF realization은 Motivation
  단계에서 kill하고 추가 model-specific rescue/tuning을 금지한다.
- 양 모델이 회복하면 낮은 update 구현 원인을 닫고, applied hard barrier/QP projection을
  다시 설계하는 Method Session으로 넘긴다.
- 어느 결과든 이 c3 뒤 동일 Motivation panel의 추가 K/share/threshold sweep은 하지
  않고 Motivation을 닫는다.

## 실행 envelope

- 목적: c1 BF share를 고정한 global magnitude-only 원인분리
- 허용 write path: `local/results/raw/session01_motivation/caphist_*_c3_v1/`,
  `local/logs/slurm/session01_motivation/`,
  `local/state/slurm-submissions/session01_motivation/caphist_pair_c3_v1.submitted/`,
  small Git report/audit/metadata
- Slurm: clean pushed main과 exact preflight PASS 뒤 GH one-shot만 allowed
- GPU/memory: server1 GPU 4; parent `260000M`, child별 GPU 1 / `65000M`
- artifact broadcast: active peer clone이 없으면 no-peer exception 기록
- 완료 보고: `experiment-reports/global/`, `audits/global/`, `messages/head/`
- 금지: EasyEdit 수정, cache/projector 재계산, online download, model별 rescue,
  raw artifact/log/credential Git 유입, direct-z 임시 session 접촉
- 중단 조건: session/repo/job/run/resource collision, dirty/unpushed Git, worker nonzero,
  exact-hop/share/proportionality/firewall/lineage/read-only 위반

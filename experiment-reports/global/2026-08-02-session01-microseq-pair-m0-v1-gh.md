# Session 01 Motivation — microseq pair m0 v1 GH decision

날짜: 2026-08-02

방법명: **ODE-Edit**

상태: **technical-valid / full-distance always-refresh kill / trade-off mechanism survives**

## Proposal에서 온 내용

- ODE-Edit은 작은 step 자체가 아니라 current state에서 proposal을 다시 만들고,
  rewrite progress 대비 capacity cost가 낮은 trajectory를 선택하려는 방향이다.
- capacity를 줄여도 efficacy가 무너지면 성공이 아니며, first-hit·utility constraint·
  trust ratio·capacity routing이 필요하다.
- proposal은 검증된 paper plan이 아니다.

## Repo/protocol에서 확인한 사실

- original job `15798`은 첫 Llama action commit 전 boolean firewall attestation
  naming 충돌로 실패했다. action/receipt 0이며 과학 결과로 사용하지 않는다.
- value-locked repair와 full 290-test gate 뒤 exact rerun job `15799`는
  2026-08-02 03:44:13--05:17:00 KST에 `COMPLETED`됐다.
- 양 model은 동일 cases, layers 4--8, seed 37, `K=4`, per-hop `D/4`, fixed
  policy hash와 evaluator hash를 사용했다.
- 모든 controller/evaluator summary는 all-pass이고 evaluation firewall, fresh-W0
  replay, state lineage, precomputed cache, finite metric gate를 통과했다.
- pair analysis SHA-256:
  `c3e21c22b0447c084dcfb72186629ac31824d9c314b2b0841658399d08e80d7c`.

## Pair 결과

| Axis: ODE minus native | Llama | Qwen | 공통 방향 |
|---|---:|---:|---|
| current-edit utility mean | -0.421082 | -0.245538 | harm |
| final prior retention | -0.407042 | -0.636994 | harm |
| retention AUC | -0.435517 | -0.330656 | harm |
| neighborhood KL reduction | +0.0000759 | +0.0001897 | 개선 |
| generation KL reduction | +0.003248 | -0.011227 | 불일치 |
| capacity reduction | +0.00000956 | +0.0001785 | 개선 |
| max-layer-share reduction | +0.243931 | -0.043938 | 불일치 |

- common positive axes: `capacity_reduction`, `neighborhood_kl_reduction`.
- 양 model current noncollapse: false (`-0.10` floor 실패).
- pair verdict: `MICROSEQ_HARM_SIGNAL`.

## GH 판정

### Kill

- unconditional always-refresh full-distance sequential skeleton
- capacity 감소만으로 preservation/efficacy가 좋아진다는 주장
- current central-probe selector와 model별 rescue
- 현 단계의 lifelong·downstream·method-superiority claim

### Survive

- MEMIT과 projected Alpha의 atomic panel에서 direction refresh signal이 양 model에
  반복된 사실
- sequential path가 두 model에서 capacity와 neighborhood disturbance를 낮출 수
  있다는 공통 trade-off signal
- 낮은-capacity path를 rewrite efficacy constraint 아래 선택해야 한다는 proposal의
  controller motivation

이 결과는 ODE 표현 자체를 성공시킨 것이 아니라, **relinearization만 항상 켜는
방식으로는 부족하고 control problem을 실제로 풀어야 한다**는 결론이다.

## Compute 판정

ODE는 native 대비 proposal build 4배, controlled NFE 49배, measured wall 약
6.80배(Llama)/6.45배(Qwen)였다. 현재 harm 결과로 이 비용은 정당화되지 않는다.
다음 method는 accepted macro-round 평균 2 이하 또는 명확한 frontier 이득을
요구한다.

## 다음 method-stage 최소 조건

1. 양 model byte-identical rewrite-only first-hit/accept-reject rule.
2. current utility 악화 step rollback과 trust-ratio step shrink.
3. layer별 velocity를 동일 progress에서 capacity cost로 배분하는 small QP.
4. 4-edit held-out pilot에서 양 model current mean `>= -0.10`.
5. 같은 retention 또는 capacity/locality 축이 양 model에서 양수.
6. 통과 전 100+ edit나 lifelong scale 제출 금지.

## 사용자 확인 필요

- Motivation은 이 결과로 닫을 수 있다. 다음은 rescue Motivation이 아니라
  proposal의 실제 controller를 구현하는 Method Session이며, 시작 여부는 사용자
  확인이 필요하다.

# Session 01 Motivation — microseq Llama m0 v1 GH provisional

날짜: 2026-08-02

방법명: **ODE-Edit**

상태: **technical-valid / common-policy sequential harm**

분석 주체: GH. 요청한 Terra Ultra model agent는 runtime metadata에서
`gpt-5.6-terra`, `reasoning=ultra`를 검증할 수 없어 어떤 파일도 읽지 않고
종료했다. 따라서 이 문서는 독립 agent report가 아닌 deterministic compact
analysis에 대한 GH provisional 판정이다.

## Proposal에서 온 내용

- current-state proposal refresh가 sequential write의 capacity concentration과
  preservation을 바꿀 가능성을 본다.
- efficacy를 유지한 채 capacity를 줄여야 하며, rewrite progress를 잃으면
  routing/first-hit/trust-region controller가 필요하다.
- 4 edits는 lifelong 증거가 아니다.

## Repo/protocol에서 확인한 사실

- job `15799`, model `llama3-8b-inst`, same-policy 4-edit chain.
- native/ODE controller 각각 actions 4, receipts 4, target artifacts 4,
  all-pass. ODE proposal artifacts는 16, native는 4다.
- branch별 fresh-W0 evaluator checkpoints 4, combined checkpoints 8,
  evaluation firewall pass, raw text/logit/token persistence false.
- model analysis SHA-256:
  `92abde172ee457124b35a8437fa668e55b86f4f529b934a82d98eb00d6ce46b3`.
- deterministic analysis replay가 byte-identical하게 통과했다.

## 결과: ODE minus native

| Axis | Delta | 방향 |
|---|---:|---|
| current-edit utility mean | -0.421082 | harm; floor -0.10 실패 |
| final prior retention | -0.407042 | harm |
| retention AUC | -0.435517 | harm |
| neighborhood KL | +0.0000759 reduction | 보존 proxy 개선 |
| generation KL | +0.003248 reduction | 보존 proxy 개선 |
| final capacity | +0.00000956 reduction | 개선 |
| max layer share | +0.243931 reduction | concentration 개선 |

current-edit case delta는 `[-0.351903, -0.765501, -0.096172, -0.470752]`로
4/4가 음수다.

## Compute

- native: NFE 4, proposal builds 4, measured wall total 524.90초.
- ODE: NFE 196, proposal builds 16, measured wall total 3566.50초.
- ODE wall은 약 6.80배, controlled NFE는 49배다.

## GH 추정

Llama에서는 always-refresh full-distance path가 capacity와 KL disturbance를
줄였지만 rewrite/current utility와 prior retention을 더 크게 잃었다. 이는
state-dependent direction이 전혀 의미 없다는 결과가 아니라, capacity가 낮은
path가 efficacy constraint 없이 자동으로 좋은 endpoint가 되지는 않는다는
trade-off 신호다.

## 판정

- analyzer verdict: `MICROSEQ_HARM_SIGNAL`.
- kill: unconditional `K=4`, 매 hop `D/4`, always-refresh full-distance policy.
- survive: capacity/locality proxy를 낮출 수 있는 path freedom과 이를 efficacy
  constraint로 묶어야 한다는 method motivation.
- 미성립: sequential preservation, lifelong, downstream와 method superiority.

## 다음 단계 조건

동일 model-common controller에서 rewrite-only first-hit, step reject/rollback,
trust ratio와 capacity-aware layer routing을 결합해 current noncollapse를 먼저
회복해야 한다. Llama 전용 threshold/policy는 금지한다.

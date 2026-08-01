# Session 01 Motivation — microseq Qwen m0 v1 GH provisional

날짜: 2026-08-02

방법명: **ODE-Edit**

상태: **technical-valid / common-policy sequential harm**

분석 주체: GH. 요청한 Terra Ultra model agent는 runtime metadata에서
`gpt-5.6-terra`, `reasoning=ultra`를 검증할 수 없어 어떤 파일도 읽지 않고
종료했다. 따라서 이 문서는 독립 agent report가 아닌 deterministic compact
analysis에 대한 GH provisional 판정이다.

## Proposal에서 온 내용

- Llama와 동일한 current-state refresh policy가 sequential capacity와
  preservation에 공통 신호를 주는지 본다.
- model별 rescue는 허용하지 않으며 efficacy 손실을 capacity 개선으로 상쇄하지
  않는다.
- 4 edits는 lifelong 증거가 아니다.

## Repo/protocol에서 확인한 사실

- job `15799`, model `qwen2.5-7b-inst`, same-policy 4-edit chain.
- native/ODE controller 각각 actions 4, receipts 4, target artifacts 4,
  all-pass. ODE proposal artifacts는 16, native는 4다.
- branch별 fresh-W0 evaluator checkpoints 4, combined checkpoints 8,
  evaluation firewall pass, raw text/logit/token persistence false.
- model analysis SHA-256:
  `c1dca51e6ffbbae274cbbad78284072ba170f9eef7249764a8a29f0735b5f579`.
- deterministic analysis replay가 byte-identical하게 통과했다.

## 결과: ODE minus native

| Axis | Delta | 방향 |
|---|---:|---|
| current-edit utility mean | -0.245538 | harm; floor -0.10 실패 |
| final prior retention | -0.636994 | harm |
| retention AUC | -0.330656 | harm |
| neighborhood KL | +0.0001897 reduction | 보존 proxy 개선 |
| generation KL | -0.011227 reduction | 악화 |
| final capacity | +0.0001785 reduction | 개선 |
| max layer share | -0.043938 reduction | concentration 악화 |

current-edit case delta는 `[+0.051590, -1.099023, -0.811885, +0.877167]`로
2개는 양수지만 두 큰 음수가 평균을 noncollapse floor 아래로 내렸다.

## Compute

- native: NFE 4, proposal builds 4, measured wall total 731.10초.
- ODE: NFE 196, proposal builds 16, measured wall total 4712.10초.
- ODE wall은 약 6.45배, controlled NFE는 49배다.

## GH 추정

Qwen도 capacity와 neighborhood KL은 개선했지만 current utility와 prior retention이
악화됐다. generation KL과 layer concentration은 Llama와 같은 방향이 아니므로
그 축의 model-common benefit도 열 수 없다. 특히 positive case와 negative case가
섞인 것은 fixed full-distance integration 대신 state-dependent accept/stop 조건이
필요함을 시사하지만, 현재 결과만으로 적절한 threshold를 고를 수는 없다.

## 판정

- analyzer verdict: `MICROSEQ_HARM_SIGNAL`.
- kill: unconditional `K=4`, 매 hop `D/4`, always-refresh full-distance policy.
- survive: common capacity reduction 가능성과 efficacy-constrained controller의 필요성.
- 미성립: sequential preservation, lifelong, downstream와 method superiority.

## 다음 단계 조건

Llama와 byte-identical한 first-hit/trust-ratio/capacity-routing policy만 허용한다.
Qwen 결과를 보고 threshold나 branch를 별도로 조정하지 않는다.

# Session 01 capacity/share exact-quarter c2_v1 — pair synthesis

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- policy: `capacity-share-history-exact-quarter-k4-v3`
- technical verdict: **pass**
- diagnostic verdict: **`LOW_UPDATE_CONFOUND_CONFIRMED_ON_LLAMA; CROSS_MODEL_CAUSE_BUNDLED`**
- Motivation status: **resolved by c3; Motivation closed directional-positive**
- claim boundary: same-policy 4-edit implementation diagnostic only

## 결론

> **c3 최종 resolution:** c1 BF share를 보존한 magnitude-only c3가 네 cell 모두
> c1보다 회복해 low-update implementation cause를 cross-model로 확인했다. Final
> verdict는
> `experiment-reports/global/2026-08-02-session01-caphist-pair-c3-v1-synthesis.md`다.

사용자 판단의 핵심은 맞았다. c1에서 실제 update가 낮게 들어간 것은 단순 수치상의
작은 차이가 아니라 Llama efficacy를 크게 무너뜨린 구현 confound였다. Exact
`D/4 × 4`로 보정하자 MEMIT은 `+2.663`, Alpha-history는 `+1.358` 회복했다.

그러나 c2는 전체 magnitude와 layer 가중치 규칙을 동시에 바꿨다. Qwen은 두 family에서
c1보다 약 `0.40--0.43` 악화했고, c1/c2 layer share 자체가 크게 다르다. 따라서 c2로
“BF layer weighting이 틀렸다”거나 “update 확대가 Qwen을 해쳤다”는 scientific 결론을
낼 수 없다. 남은 최소 실험은 **c1 BF layer share를 그대로 유지하고 전체 norm만 exact
quarter로 맞추는 magnitude-only control**이다.

## 네 범주

### Proposal에서 온 내용

- current-state direction refresh와 layer별 capacity-aware velocity를 결합한다.
- MEMIT/AlphaEdit에 공통 policy를 적용한다.
- efficacy와 preservation/capacity를 함께 보되 Motivation에서는 작은 signal로 판정한다.

### Repo/protocol에서 확인한 사실

- job `15875`는 4 GPU에서 네 model×family worker를 동시에 실행해
  `COMPLETED 0:0`, elapsed `01:28:11`로 끝났다.
- controller 8개와 evaluator 8개가 모두 terminal/pass이고 checkpoint는 총 32개다.
- 16 QP edits, 64 hops에서 max path relative error `1.624e-16`, max hop relative
  error `1.882e-16`, max share-L2 error `2.220e-16`이다.
- capacity barrier violation과 trust-gate failure는 0이고 overload observation은 41개다.
- EasyEdit/source/cache는 수정하지 않았고 precomputed artifact만 read-only로 썼다.

### GH 추정

- Llama recovery는 low-update 구현 오류가 c1 negative의 주요 원인이었다는 강한 신호다.
- Qwen mixed-negative는 magnitude와 changed share의 interaction일 가능성이 있어 순수
  magnitude effect가 아니다.
- c2의 native 대비 손실은 deployable ODE-Edit superiority가 아직 없다는 뜻이지,
  magnitude-only 구현 수정을 과학적으로 kill하는 증거는 아니다.

### 사용자 확인 필요

- 없음. 사용자가 구현 문제를 우선 판정하고 재구현·실험까지 명시적으로 승인했다.

## Pair 결과

모든 수치는 QP minus native다.

| Model | Family | current utility | c1→c2 변화 | final all-edit | prior retention | capacity reduction |
|---|---|---:|---:|---:|---:|---:|
| Llama | MEMIT | `-1.369350` | `+2.663042` | `-1.371123` | `-1.477975` | `+0.000041` |
| Llama | Alpha-history | `-1.137219` | `+1.357904` | `-1.120088` | `-1.531226` | `+0.000380` |
| Qwen | MEMIT | `-0.797188` | `-0.399352` | `-0.795757` | `-0.645751` | `+0.000803` |
| Qwen | Alpha-history | `-0.400738` | `-0.430445` | `-0.416153` | `-0.363292` | `+0.001816` |

Old analyzer의 `CAPACITY_HISTORY_HARM_SIGNAL`은 수치 gate 결과로 보존하지만 최종
causal verdict로 쓰지 않는다. Pre-registered analyzer는 c2가 두 구현 축을 함께 바꾼
식별 문제를 알 수 없기 때문이다.

## 무엇이 확인됐고 무엇이 남았나

- 확인: c1 absolute QP coefficient를 실제 update로 쓴 구현은 Llama에 과도한 under-update를
  만들었다.
- 확인: exact magnitude 적용은 구현 가능하며 64/64 hop에서 정확했다.
- 미확인: c1 BF layer weighting을 유지했을 때 exact magnitude만으로 양 모델이 같이
  회복하는가.
- 미확인: current BF weighting이 native보다 나은 efficacy/preservation frontier인가.
- 금지: Llama/Qwen별 다른 share, K, threshold 또는 rescue.

## Agent와 artifact

Llama, Qwen, red/RCA Terra Ultra agents를 각각 관련 c2 파일에만 격리 호출했으나 세
agent 모두 runtime metadata에서 `gpt-5.6-terra / ultra`를 검증하지 못해 파일을 읽지
않고 `BLOCK` 종료했다. 결과 수신 뒤 active agent는 남아 있지 않다. 독립 분석 pass는
0건이며 GH fallback만 쓴다.

- Llama analysis SHA-256:
  `8053ac2c753ed6bd7059709b478cf0a7e8558dcbdb3820d57b328e3cae731e18`
- Qwen analysis SHA-256:
  `60086eeb7be8882f636e0fdf7630160ff9a3ccdfa7fdc3ccc8c7140842791a7f`
- Pair analysis SHA-256:
  `43f4582e8f2cf6dbddcef25a2e831f5e4880dfe6cf147b3ff76d800bcbaafc59`
- evaluator/combined/pair compact 29 files aggregate SHA-256:
  `52fc5a6629b2c0d60b26adc656d9d2df4fc4787ebd557825438bc0f47673793f`

## 다음 gate

다음 control은 c1 QP가 만든 layer coefficient 비율과 zero support를 보존하고, 이를
joint L2-normalize한 뒤 global `D/4`만 곱한다. 이 control은 c1 cap을 hard deployment
barrier로 주장하지 않고 **share generator**로만 사용한다.

- 양 모델·양 family가 c1보다 회복: low-update implementation cause를 cross-model로
  닫고 Motivation을 method-stage 설계로 넘긴다.
- Llama만 회복하고 Qwen은 재현 실패: magnitude는 Llama cause, model-common BF
  realization은 Motivation에서 미성립으로 닫는다.
- 양 모델 모두 회복하지 않음: low-update 단독 설명을 kill하고 BF share/direction
  refresh를 주 원인으로 올린다.

# Session 01 Motivation — projected Alpha Llama g0 v3 GH provisional

날짜: 2026-08-02

방법명: **ODE-Edit**

상태: **technical-valid / selector collapse / always-refresh transfer pass**

분석 주체: GH. 모델별 Terra Ultra agent는 runtime에서
`gpt-5.6-terra`, `reasoning=ultra` metadata를 검증할 수 없어 어떤 파일도
읽지 않고 종료했다. 따라서 이 문서는 독립 agent report가 아닌 locked compact
scalar에 대한 GH provisional 판정이다.

## Proposal에서 온 내용

- AlphaEdit projector가 허용 update subspace를 제한한 뒤에도 current-state
  direction relinearization이 fixed/transported direction보다 유리한지 본다.
- 작은 step 자체가 아니라 매 waypoint proposal direction을 다시 만드는 것이
  ODE motivation이다.
- proposal은 native AlphaEdit preservation이나 method superiority를 증명하지 않는다.

## Repo/protocol에서 확인한 사실

- job `15772`, model `llama3-8b-inst`, track `alphaedit_projected`, exact 8 cases.
- features/actions/analysis/events 8, outcomes 56, receipts 8, target artifacts 8,
  failure 0, rollback exact, evaluation firewall pass.
- EasyEdit의 Wikipedia covariance 5개와 pinned projector
  `6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec`를
  read-only load했고 재계산하지 않았다.
- raw summary SHA-256: `c28e6285963e3bab012f886aa9ebf6c5df4d0fff006811fbf3e9dd2c1134f5c6`.
- compact analysis SHA-256: `0694347b268e7a2f282d855c4b3e8ec4a3ed3699f43b665def62c068846a2744`.

## Scalar 결과

| Contrast | Mean | 8-case sign |
|---|---:|---:|
| always direction refresh `A4-B4` | +0.826986 | 8 positive / 0 negative |
| coefficient refresh `B4-C4` | +0.384075 | 8 positive / 0 negative |
| gated selector `G4-A4` | -0.770814 | 0 positive / 8 negative |
| naive `split4-native` | -0.000002 | 8/8 practical floor |

- selector refresh: 2/24; normalized predicted advantage mean `-0.165816`.
- projector right-factor norm retention: count 80, mean `0.999997`, minimum
  `0.999971`.
- controlled NFE 1312, proposal builds 64, measured wall total 20,928.09초.

## GH 추정

Projection 뒤에도 always-refresh `A4`가 transported direction `B4`보다 8/8
높았다. MEMIT의 Llama `A4-B4=+0.828098`과 거의 같은 크기라, 현재 작은 panel에서
null-space projection이 relinearization signal을 제거하지 않았다는 해석이 가장
단순하다. 반면 `G`가 22/24 fixed를 골라 `A4` 이득을 잃었으므로 selector는
Alpha에서도 kill한다.

Projector retention이 1에 가깝다는 것은 이 panel의 right factor가 projection으로
거의 소거되지 않았다는 뜻이지, locality·downstream·과거 edit preservation이
보장됐다는 뜻이 아니다.

## 판정과 다음 gate

- locked postpivot model gate: pass (`A4-B4>1e-4`, casewise noncollapse `0.0`,
  projector finite/nonzero).
- atomic 기대효과: transported-direction 대비 teacher-forced utility 약 `+0.827`.
- 미검증: native AlphaEdit reproduction, preservation, sequential retention,
  lifelong robustness와 method superiority.
- 다음: 두 model 공통 always-refresh 4-edit microseq에서 retention/KL/capacity의
  동일 축 신호를 확인한다.

## 사용자 확인 필요

- 없음. model별 rescue 없이 사전 고정 공통 policy를 유지한다.

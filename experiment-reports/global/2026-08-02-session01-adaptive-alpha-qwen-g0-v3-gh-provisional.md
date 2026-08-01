# Session 01 Motivation — projected Alpha Qwen g0 v3 GH provisional

날짜: 2026-08-02

방법명: **ODE-Edit**

상태: **technical-valid / selector collapse / always-refresh transfer pass**

분석 주체: GH. 모델별 Terra Ultra agent는 runtime에서
`gpt-5.6-terra`, `reasoning=ultra` metadata를 검증할 수 없어 어떤 파일도
읽지 않고 종료했다. 따라서 이 문서는 독립 agent report가 아닌 locked compact
scalar에 대한 GH provisional 판정이다.

## Proposal에서 온 내용

- projected actuator에서도 state-dependent direction refresh가 static/transported
  direction보다 추가 신호를 주는지 확인한다.
- 양 model은 동일 method와 동일 threshold를 써야 한다.
- Motivation signal을 preservation 또는 lifelong claim으로 확장하지 않는다.

## Repo/protocol에서 확인한 사실

- job `15772`, model `qwen2.5-7b-inst`, track `alphaedit_projected`, exact 8 cases.
- features/actions/analysis/events 8, outcomes 56, receipts 8, target artifacts 8,
  failure 0, rollback exact, evaluation firewall pass.
- EasyEdit의 Wikipedia covariance 5개와 pinned projector
  `d3a9687d196f7ee06739253813917a632542717ce1166c0433aa7eec70c5c8ff`를
  read-only load했고 재계산하지 않았다.
- raw summary SHA-256: `d55a98e88a0918fb3343f515be41029b3f184bc5348eb39fa0331cbe81803a50`.
- compact analysis SHA-256: `fd3d77e532db6085b978dff3b7947e1b54ac4ea853645c6dacb3fb063e14594d`.

## Scalar 결과

| Contrast | Mean | 8-case sign |
|---|---:|---:|
| always direction refresh `A4-B4` | +1.668561 | 8 positive / 0 negative |
| coefficient refresh `B4-C4` | +0.540036 | 7 positive / 1 negative |
| gated selector `G4-A4` | -0.127810 | 0 positive / 3 negative / 5 floor |
| naive `split4-native` | -0.0000005 | 8/8 practical floor |

- selector refresh: 21/24; normalized predicted advantage mean `+0.273186`.
- projector right-factor norm retention: count 80, mean `0.987945`, minimum
  `0.959304`.
- controlled NFE 1312, proposal builds 64, measured wall total 26,235.76초.

## GH 추정

Qwen에서도 projected `A4`가 `B4`보다 8/8 높았고, MEMIT의 `+1.583319`보다
약간 큰 `+1.668561`이었다. 이 차이는 같은 8-case Motivation panel 안의 관측이며
projection이 refresh 효과를 증폭한다는 일반 claim으로 쓰지 않는다.

`G`는 대부분 refresh를 골랐지만 세 case의 일부 hop에서 fixed를 선택해 평균
`A4`보다 낮았다. Llama/Qwen별 selector 조정은 금지하므로 `G`는 계속 kill하고
양 model 모두 같은 always-refresh policy를 사용한다.

## 판정과 다음 gate

- locked postpivot model gate: pass (`A4-B4>1e-4`, casewise noncollapse `0.0`,
  projector finite/nonzero).
- atomic 기대효과: transported-direction 대비 teacher-forced utility 약 `+1.669`.
- 미검증: native AlphaEdit reproduction, locality/model preservation,
  sequential/lifelong robustness와 method superiority.
- 다음: 동일 always-refresh 4-edit microseq의 공통 retention/KL/capacity 축을 본다.

## 사용자 확인 필요

- 없음. model별 rescue 없이 공통 policy를 유지한다.

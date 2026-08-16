# P1R24 Qwen 기술 보고서

## 범위와 무결성

- 실행 과학 HEAD/base: `ce8c6c36348752f1407f7d713d30e6b5c727379b` / `a343d1f6967ef37763009b92d227ade85cd93de0`
- 동일 seal/order와 full-six evaluator를 사용했다. 유효 B1 smoke는 K8/tau1, action-freeze, W0 restore를 통과했다.
- Production RS Neutral/Soft는 각각 K8/tau1, terminal/manifest/W0 restore를 통과했다. BG는 Neutral k4/tau0.5 이후 typed `NO_POSITIVE_DIRECTION/Q_NUMERICAL_DEGENERACY` 경계에 도달했다. BG Soft는 시작하지 않았고 endpoint/manifest/W0 restore는 확립되지 않았다.

## FACT: endpoint

| allocation | arm | status | Eff | Gen | Loc | Eff new NLL | Gen new NLL | Loc new NLL | z8 NLL | cumulative P | functional KL | capacity sum | edit-core |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RS | Neutral | K8 complete | 10/10 | 19/20 | 80/100 | 0.538097 | 2.404022 | 8.829219 | 0.048375 | 0.253839 | 0.074767 | 28.272942 | 107.232s |
| RS | Soft | K8 complete | 10/10 | 20/20 | 80/100 | 0.021383 | 1.612979 | 8.828750 | 0.014902 | 0.126967 | 0.061247 | 12.349511 | 88.419s |
| BG | Neutral | typed boundary at k4 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | partial only | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| BG | Soft | not started | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |

W0는 `0/10,4/20,80/100`; Official AlphaEdit reference는 `10/10,20/20,80/100`, wall 49.88s이다. Official continuous NLL/pure-edit compatibility는 `NOT_RECORDED`다.

## FACT: strength, P, compute

- RS의 16 accepted field 모두 equality residual 최대 `8.88e-16`, h 적용 1회/field, second remaining division 0회를 기록했다.
- RS Soft는 8/8 same-state에서 cumulative-P proxy를 Neutral shadow보다 낮췄다. Terminal에서도 cumulative-P `0.253839→0.126967`, functional KL `0.074767→0.061247`, capacity `28.272942→12.349511`로 낮아졌다.
- RS actual realization은 Neutral/Soft 모두 7개 delayed step 중 1개가 음수였다. 평균 ratio는 `0.232/0.247`로 under-realization이 크다.
- RS arm별 `180F/125B`, tokens `24,687`, materialization 8회였다.
- BG Neutral의 첫 네 field는 모두 양의 slope/q와 predicted progress를 가졌으나 k4 이후 다음 필드에서 과학적 typed totality가 발생했다. partial prefix는 endpoint로 쓰지 않는다.

## INFERENCE

- RS Soft는 RS Neutral보다 Gen count를 `19→20`, Eff/Gen NLL과 P/KL/capacity를 모두 개선했다. 이 completed pair 안에서는 Soft signal이 있다.
- 그러나 RS Neutral이 Official/P1R23 Gen count보다 낮고 Eff NLL이 이전 P1R23 RS Neutral `0.037970`보다 `0.538097`로 크게 나빠 strength recovery gate를 충족하지 못한다.
- BG 두 endpoint가 없으므로 8-cell matrix와 BG rescue 여부를 평가할 수 없다. 계약상 typed boundary에는 rescue/retry가 없고, 누락 값을 대체할 수 없다.

## NOT_RECORDED / 경계

- Qwen BG terminal endpoint, Soft allocation, manifest/action-freeze/W0 restore
- Official AlphaEdit continuous metrics와 formal pure-edit ratio
- fresh-sample/보편적 회복 증거

따라서 fail-closed package 분류는 `ATOMIC_NOT_RECOVERED`, `scientific_promotion=false`다.

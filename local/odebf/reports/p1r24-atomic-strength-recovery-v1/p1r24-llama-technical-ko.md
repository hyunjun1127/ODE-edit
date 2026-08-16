# P1R24 Llama 기술 보고서

## 범위와 무결성

- instruction: `ODEEDIT-S05-P1R24-ATOMIC-STRENGTH-RECOVERY-V1`
- 실행 과학 HEAD: `ce8c6c36348752f1407f7d713d30e6b5c727379b` (base `a343d1f6967ef37763009b92d227ade85cd93de0`)
- seal/order: `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628` / `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- numerical/source root: `647c18243298dd342a2c4f78669bf9a07e019fc08cddad836fd6c330b62dc850` / `baefc1621e1275b5c1a4d3440f98fcf095109103cd5e7746e6829a212971e8f5`
- 유효 B1 smoke는 K8/tau1, action-freeze, W0 restore를 통과했다. Production RS/BG의 Neutral/Soft 네 arm도 각 8회 accepted update, tau1, manifest 및 W0 restore를 통과했다.

## FACT: endpoint

| allocation | arm | Eff | Gen | Loc | Eff new NLL | Gen new NLL | Loc new NLL | z8 NLL | cumulative P | functional KL | capacity sum | edit-core |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RS | Neutral | 10/10 | 20/20 | 82/100 | 0.021823 | 0.961012 | 8.311719 | 0.039689 | 0.003428 | 0.006004 | 2.503783 | 78.965s |
| RS | Soft | 10/10 | 20/20 | 81/100 | 0.026463 | 1.066080 | 8.316563 | 0.039909 | 0.003540 | 0.005619 | 2.630140 | 86.042s |
| BG | Neutral | 10/10 | 20/20 | 81/100 | 0.769477 | 1.302243 | 8.257188 | 0.099445 | 0.005571 | 0.007328 | 4.064854 | 89.747s |
| BG | Soft | 10/10 | 20/20 | 82/100 | 0.672276 | 1.363885 | 8.273125 | 0.084926 | 0.005884 | 0.006124 | 4.238934 | 84.564s |

W0는 Eff/Gen/Loc `2/10, 3/20, 84/100`이었다. 동일-seal Official AlphaEdit reference는 `10/10,20/20,79/100`, wall 51.90s이나 연속 NLL과 phase-compatible pure-edit 분모는 `NOT_RECORDED`다.

## FACT: strength, P, compute

- 모든 32 accepted field에서 `a^T v=rho_write` 잔차는 최대 `8.88e-16`, physical-h 적용은 field당 정확히 1회, 두 번째 remaining division은 0회였다.
- Soft는 RS 8/8, BG 8/8 같은-state에서 selected cumulative-P proxy를 Neutral shadow보다 낮췄다. 하지만 terminal cumulative-P와 BF16 capacity는 Soft가 두 allocation 모두 더 높았다.
- delayed actual progress는 모든 기록 step에서 양수였다. realization ratio 평균은 RS Neutral/Soft `0.523/0.525`, BG Neutral/Soft `0.435/0.440`으로 예측보다 작았다.
- arm별 compute는 `180F/125B`, processed tokens `27,207`, materialization 8회였다. 증가된 target KL gradient가 포함되어 이전 P1R23 `130F/80B`보다 무겁다.
- edit-core는 이전 P1R23 대비 RS Neutral/Soft 각각 `-24.08s/-16.64s`, BG Neutral/Soft `-7.68s/-16.46s`였지만, Official AlphaEdit와의 formal pure-edit ratio는 호환 phase 분모 부재로 계산하지 않았다.

## INFERENCE

- RS는 절대 edit strength를 회복했고 Official AlphaEdit의 Eff/Gen count를 맞췄다. 그러나 Soft가 RS에서 Loc를 1점 낮추고 NLL도 악화했으며, terminal cumulative-P/capacity도 증가해 일반적인 BF 보존 신호는 아니다.
- BG Soft는 BG Neutral 대비 Eff NLL과 functional KL, Loc count를 개선했지만 Gen NLL, terminal cumulative-P와 capacity는 악화했다. 이는 mixed signal이다.
- 같은-step matched predicted strength는 성립하지만, 서로 다른 K8 trajectory의 누적 결과를 동일-state 인과효과로 해석할 수 없다.

## NOT_RECORDED / 경계

- Official AlphaEdit continuous NLL 및 non-overlapping pure-edit phase identity
- fresh-sample, sequential/Historical, B100 evidence
- 과학 승격 근거

이 결과는 outcome-selected 재사용 B10의 atomic mechanistic/descriptive evidence이며 `scientific_promotion=false`다.

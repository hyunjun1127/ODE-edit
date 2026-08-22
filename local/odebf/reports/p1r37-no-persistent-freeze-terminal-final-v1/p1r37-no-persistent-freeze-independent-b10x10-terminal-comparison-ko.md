# P1R37 Current-State Instantaneous Freeze 독립 B10×10 종료 비교 보고서

- instruction: `ODEEDIT-S05-P1R37-P1R36-NO-PERSISTENT-FREEZE-INDEPENDENT-B10X10-V1`
- 상태: `TERMINAL`, job `19528_0..7` 모두 `COMPLETED`, `ExitCode=0:0`
- P1R37 분모: 80 attempts / 73 endpoints / 7 typed incomplete
- P1R36 분모: 80 attempts / 71 endpoints / 9 typed incomplete
- 정확한 공통 성공 pair: 71
- technical failure: 0 (최종 scientific array 기준)
- `scientific_promotion=false`

## 1. FACT — source, sample, evaluator, 결과 무결성

| 항목 | 값 |
|---|---|
| P1R37 source head | `27fd7714532599abfe169dc9cccf83b7a66735b9` |
| source tree | `587e1ac06cc1dda64400ee202499048249269582` |
| execution parent | `0f0907b881eec8d4efdce02b3bd860821bf1b928` |
| P1R36 source parent | `87efd168fc9680cabde878cd3b595e98db66d8fa` |
| P1R35 scientific parent | `a625e3d1cded3ced0e9128ef7a44205953041447` |
| source manifest SHA | `215893b1c993e6fa367e86a8b649f691d609beeef9237b22a019bb48a7efb2c6` |
| source manifest root | `46ab24c1ca8739313c4c657c5e344de1e61d02e2d0bf20e0a55e43390bb4f378` |
| numerical lock SHA | `c980677247280cca5cba7a8c840d33266cb5d44cb3698e303651da5b5d1bfde2` |
| numerical lock root | `388332e966c6b75bfa794929148bee2ba8c88d01c4e9ba97b88adeffb3d575ca` |
| stream root | `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` |
| request order | `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c` |
| evaluator source SHA | `25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145` |
| freeze policy | `CURRENT_STATE_INSTANTANEOUS_NO_CARRY` |

확인 사실:

- source manifest 12개 항목은 실제 파일 size/SHA와 모두 일치했다. P1R36→P1R37 diff의 13개 경로는 manifest 12개 항목과 manifest 파일 자체로 닫힌다.
- 8개 result-root terminal/manifest identity, 73개 endpoint terminal/manifest/action-freeze SHA, 631개 accepted-prefix receipt identity를 확인했다.
- 성공 endpoint 73개는 K=8, accepted-state field refresh=8, materialization=8, W0 byte/pointer restore와 action-freeze가 모두 PASS이다.
- 7개 typed incomplete prefix도 W0 byte/pointer restore가 PASS이며 retry=0이다.
- P1R36 raw-free package의 13개 payload SHA/size/mode와 JSON parse를 확인했다. package manifest SHA는 `b03d28556f16c86bfc13f35174d76788eb90b544e019eaa708b29e2615708756`, package root는 `f5deaf43f5a9e42e158032bf2ce73de49d62ebbae1678182425be5fefa2e4653`이다.
- P1R37 80개 objective plan의 request order는 P1R36 seal의 case별 ordered digest와 모두 일치했다.

## 2. FACT — 유일한 scientific delta

P1R36의 영구 누적 latch:

`persistent_mask_k = persistent_mask_{k-1} OR [phi_k < 0.05]`

P1R37의 현재-state mask:

`instantaneous_mask_k = [phi_k < 0.05]`

threshold `0.05`, target/writer/router, RS/BG, Neutral/Soft, K8/h=1/8, stream/order/evaluator/W0는 고정했다. P1R37 runtime receipt에서 `persistent_mask_decision_influence_count=0`, `carried_frozen_input_count=0`이다.

## 3. FACT — cell별 completion 분모

| Model | Allocation | Arm | P1R36 endpoint | P1R37 endpoint | 공통 pair |
|---|---|---:|---:|---:|---:|
| Llama | RS | Neutral | 10 | 10 | 10 |
| Llama | RS | Soft | 8 | 10 | 8 |
| Llama | BG | Neutral | 10 | 10 | 10 |
| Llama | BG | Soft | 10 | 10 | 10 |
| Qwen | RS | Neutral | 8 | 8 | 8 |
| Qwen | RS | Soft | 9 | 9 | 9 |
| Qwen | BG | Neutral | 8 | 8 | 8 |
| Qwen | BG | Soft | 8 | 8 | 8 |
| **합계** |  |  | **71** | **73** | **71** |

P1R37은 P1R36에서 성공한 endpoint를 잃지 않았다. P1R37에만 추가된 endpoint는 `Llama RS-Soft case02`와 `case06`이다. 공통 continuous metric delta에는 이 두 endpoint를 넣지 않았다.

## 4. FACT — freeze/no-carry 궤적

| 범위 | accepted steps | phi values | active→frozen | reactivation | frozen request-step | persistent influence | carried input |
|---|---:|---:|---:|---:|---:|---:|---:|
| 모든 성공/실패 prefix | 631 | 6,310 | 224 | 8 | 539 | 0 | 0 |
| 성공 endpoint만 | 584 | 5,840 | 196 | 8 | 486 | 0 | 0 |

631개 step의 6,310개 값에서 `mask == (phi < 0.05)`가 모두 성립했고, prior mask와 current mask로 계산한 transition도 receipt와 일치했다. reactivation이 기록된 case는 다음과 같다.

| Cell/case | reactivation |
|---|---:|
| Llama RS-Neutral case09 | 1 |
| Qwen BG-Neutral case09 | 1 |
| Qwen BG-Soft case02 | 2 |
| Qwen BG-Soft case06 | 1 |
| Qwen RS-Neutral case08 | 1 |
| Qwen RS-Neutral case10 | 1 |
| Qwen RS-Soft case07 | 1 |

사전 선언 sentinel `Llama RS-Soft case02 k7`은 1회 관측되었다. 해당 step은 active request 7, frozen request 3, predicted progress `0.257059`, actual progress `0.161199`, accepted k7이다. P1R36의 같은 prefix에는 terminal actual이 기록되지 않았다.

## 5. FACT — P1R37 전체 73 endpoint 성능

아래 표는 P1R37의 유효 endpoint만 집계한다. typed incomplete는 0으로 채우지 않았다.

| Model | Alloc | Arm | endpoint | Eff | Gen | Loc | Eff new NLL | Eff margin | z8 NLL | W8 NLL | W−z gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | BG | N | 10/10 | 98/100 | 178/200 | 871/1000 | 0.413279 | 9.799534 | 0.309746 | 0.434665 | 0.124919 |
| Llama | BG | S | 10/10 | 97/100 | 179/200 | 873/1000 | 0.492727 | 9.655398 | 0.345314 | 0.483595 | 0.138281 |
| Llama | RS | N | 10/10 | 100/100 | 181/200 | 874/1000 | 0.135492 | 11.981852 | 0.092311 | 0.147567 | 0.055256 |
| Llama | RS | S | 10/10 | 100/100 | 180/200 | 875/1000 | 0.123753 | 12.022497 | 0.059848 | 0.117396 | 0.057548 |
| Qwen | BG | N | 8/10 | 76/80 | 131/160 | 662/800 | 0.807791 | 8.937457 | 0.513257 | 0.657296 | 0.144039 |
| Qwen | BG | S | 8/10 | 74/80 | 123/160 | 676/800 | 0.980932 | 8.369345 | 0.687846 | 0.890294 | 0.202448 |
| Qwen | RS | N | 8/10 | 77/80 | 125/160 | 682/800 | 0.532875 | 11.052805 | 0.335436 | 0.431718 | 0.096282 |
| Qwen | RS | S | 9/10 | 87/90 | 141/180 | 760/900 | 0.480938 | 11.508959 | 0.344247 | 0.403433 | 0.059186 |

## 6. FACT — P1R37 progress, P/capacity, functional-P, compute

각 값은 해당 cell의 유효 endpoint 평균이다. `Σpred`, `Σactual`은 8-step 합, realization은 step ratio의 case 평균, `ΣP`는 8-step Structural-P 합이다.

| Model/Cell | Σpred | Σactual | realization | neg actual | terminal P | ΣP | terminal cap | terminal energy | terminal fP | edit-core s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama BG-N | 13.242660 | 9.953832 | 0.712040 | 0 | 0.005406 | 0.019880 | 0.063410 | 0.105915 | 0.014099 | 72.366 |
| Llama BG-S | 13.180622 | 9.904902 | 0.697676 | 1 | 0.005253 | 0.019682 | 0.052102 | 0.088043 | 0.014640 | 77.636 |
| Llama RS-N | 13.170454 | 10.240930 | 0.722700 | 1 | 0.006239 | 0.024366 | 0.043314 | 0.071270 | 0.021520 | 73.886 |
| Llama RS-S | 13.108966 | 10.271101 | 0.727543 | 1 | 0.006084 | 0.024198 | 0.033286 | 0.054799 | 0.019122 | 74.461 |
| Qwen BG-N | 13.934060 | 9.715251 | 0.666107 | 0 | 0.194937 | 0.682923 | 0.948679 | 0.705896 | 0.002539 | 98.583 |
| Qwen BG-S | 13.242893 | 9.396062 | 0.688015 | 0 | 0.134983 | 0.486268 | 0.384380 | 0.332018 | 0.004420 | 93.500 |
| Qwen RS-N | 13.544234 | 10.097811 | 0.727207 | 0 | 0.179832 | 0.692650 | 0.567263 | 0.420259 | 0.001390 | 88.941 |
| Qwen RS-S | 13.500986 | 10.150905 | 0.749748 | 0 | 0.109887 | 0.473377 | 0.081426 | 0.084410 | 0.003041 | 88.730 |

모든 유효 endpoint의 logical compute는 case당 F=220, B=125, materialization=8이다. token 수는 case 길이에 따라 달라진다.

## 7. FACT — 71개 exact-pair P1R37−P1R36

| 항목 | P1R36 | P1R37 | Δ |
|---|---:|---:|---:|
| Eff | 693/710 | 689/710 | -4 |
| Gen | 1206/1420 | 1204/1420 | -2 |
| Loc | 6103/7100 | 6104/7100 | +1 |
| mean z8 NLL | 0.300308 | 0.329892 | +0.029584 |
| mean W8 NLL | 0.410680 | 0.439523 | +0.028843 |
| mean W−z gap | 0.110373 | 0.109631 | -0.000741 |
| mean predicted progress sum | 13.392784 | 13.354247 | -0.038537 |
| mean actual progress sum | 10.003415 | 9.974572 | -0.028843 |
| negative-actual steps | 3 | 3 | 0 |
| mean terminal Structural-P | 0.074522 | 0.074434 | -0.000088 |
| mean 8-step ΣP | 0.280565 | 0.281480 | +0.000915 |
| mean terminal capacity | 0.277801 | 0.251072 | -0.026729 |
| mean terminal energy | 0.252636 | 0.219380 | -0.033256 |
| F / B / tokens / materializations | 220 / 125 / 35,752.14 / 8 | 동일 | 0 |
| mean edit-core seconds | 115.203 | 82.861 | -32.342 |
| mean terminal evaluator seconds | 20.004 | 17.464 | -2.540 |

### Cell별 exact-pair delta

`E/G/L`은 공통 성공 case 내 count 합의 차이, 나머지는 공통 case 평균의 차이다.

| Cell | E/G/L Δ | z/W NLL Δ | terminal P / cap / energy Δ |
|---|---|---|---|
| Llama RS-N | 0 / 0 / -1 | +0.000002 / +0.000034 | -0.000001 / +0.000073 / +0.000121 |
| Llama RS-S | 0 / +1 / +1 | -0.000043 / -0.002369 | +0.000033 / +0.002094 / +0.003381 |
| Llama BG-N | 0 / 0 / 0 | 0 / 0 | 약 0 / 약 0 / 약 0 |
| Llama BG-S | 0 / -1 / 0 | +0.036530 / +0.028614 | -0.000026 / -0.000063 / +0.000094 |
| Qwen RS-N | 0 / 0 / -1 | -0.000143 / -0.000196 | -0.000468 / +0.000447 / +0.000330 |
| Qwen RS-S | -2 / -1 / +2 | +0.064986 / +0.057749 | -0.004856 / -0.067551 / -0.062609 |
| Qwen BG-N | 0 / 0 / 0 | +0.000141 / -0.000052 | -0.000034 / -0.001537 / -0.001164 |
| Qwen BG-S | -2 / -1 / 0 | +0.143829 / +0.157818 | +0.005182 / -0.162243 / -0.227527 |

## 8. FACT — P1R37에만 존재하는 두 endpoint

| Case | Eff | Gen | Loc | Eff NLL | margin | Σpred | Σactual | realization | neg | terminal P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama RS-S case02 | 10/10 | 17/20 | 86/100 | 0.071448 | 12.822302 | 14.612944 | 11.159703 | 0.733742 | 0 | 0.005842 |
| Llama RS-S case06 | 10/10 | 17/20 | 83/100 | 0.015149 | 13.941101 | 11.540948 | 9.384474 | 0.747364 | 0 | 0.006202 |

두 case는 P1R36에서 `P1R34NonSemanticTargetMove` typed incomplete였으므로 paired continuous delta를 계산하지 않았다.

## 9. FACT — P1R37 typed incomplete 7건

| Cell | case | last accepted k | type |
|---|---:|---:|---|
| Qwen BG-N | 04 | 6 | `P1R34NonSemanticTargetMove` |
| Qwen BG-N | 06 | 7 | `P1R34NonSemanticTargetMove` |
| Qwen BG-S | 01 | 7 | `P1R34NonSemanticTargetMove` |
| Qwen BG-S | 03 | 6 | `P1R34NonSemanticTargetMove` |
| Qwen RS-N | 02 | 7 | `P1R34NonSemanticTargetMove` |
| Qwen RS-N | 09 | 7 | `P1R34NonSemanticTargetMove` |
| Qwen RS-S | 09 | 7 | `P1R34NonSemanticTargetMove` |

이 7건은 P1R36에도 같은 cell/case와 last accepted k로 존재한다. endpoint 보간/대체는 0이다.

## 10. FACT — Official Native(Official AlphaEdit) 경계

Official AlphaEdit는 동일 stream의 별도 one-shot direct-z comparator이다. P1R36 또는 P1R37 ODE arm이 아니며, RS/BG/Neutral/Soft 변형도 아니다.

| Model | Eff | Gen | Loc | Eff new NLL | Eff margin |
|---|---:|---:|---:|---:|---:|
| Llama Official AlphaEdit | 100/100 | 185/200 | 859/1000 | 0.001169 | 14.751643 |
| Qwen Official AlphaEdit | 100/100 | 192/200 | 828/1000 | 0.032646 | 14.282432 |

Official comparator의 standalone z trajectory, Structural-P, capacity, P1R37 freeze/reactivation은 `NOT_RECORDED`이다. 따라서 위 endpoint 값과 ODE의 z/P/capacity 표를 결합하지 않았다.

## 11. FACT — resource와 compute

| Array task / cell | scheduler elapsed | MaxRSS |
|---|---:|---:|
| 19528_0 / Llama RS-N | 18:11 | 7.411 GiB |
| 19528_1 / Llama RS-S | 18:25 | 7.331 GiB |
| 19528_2 / Llama BG-N | 17:51 | 7.304 GiB |
| 19528_3 / Llama BG-S | 18:57 | 7.974 GiB |
| 19528_4 / Qwen RS-N | 20:07 | 13.406 GiB |
| 19528_5 / Qwen RS-S | 20:59 | 10.286 GiB |
| 19528_6 / Qwen BG-N | 22:03 | 10.316 GiB |
| 19528_7 / Qwen BG-S | 21:22 | 10.413 GiB |

P1R37 73 endpoint 합계는 edit-core `6030.261 s`, terminal evaluator `1274.608 s`, F `16,060`, B `9,125`, tokens `2,612,512`, materializations `584`이다. scheduler elapsed와 edit-core는 서로 다른 시간 계정이다.

## 12. NOT_RECORDED / 비교 제한

- P1R36 raw-free package에는 P1R36 terminal functional-P가 없다. P1R37 functional-P 값은 기록했지만 P1R36 대비 delta는 `NOT_RECORDED`이다.
- P1R36 freeze/reactivation transition telemetry는 package에 없다. P1R36에 대한 reactivation count를 복원하지 않았다.
- P1R36/P1R37 GPU peak-memory 동등 비교는 endpoint package에 없다. 위 MaxRSS는 P1R37 Slurm host-memory 값이다.
- typed incomplete 7건에는 endpoint/action-freeze가 없고 W0 restore/failure prefix만 있다.
- P1R36 sibling transport receipt는 server2 package 내부에 없으므로 그 sibling 파일은 이번 분석에서 `NOT_INDEPENDENTLY_REVERIFIED`이다. package 내부 payload/manifest/root는 독립 확인했다.

## 13. TASK-SCOPED FREEZE COMPARISON

허용된 세 분류 중 이 결과의 분류는 **`NEAR_IDENTICAL_OUTCOME`** 이다.

근거 수치: 공통 71 pair에서 Eff `-4/710`, Gen `-2/1420`, Loc `+1/7100`, negative-actual `3→3`; z/W NLL delta는 `+0.029584/+0.028843`; terminal P delta는 `-0.000088`, 8-step ΣP delta는 `+0.000915`이다. cell별 변화 방향은 동일하지 않다. 별도 completion 사실은 Llama RS-Soft endpoint가 `8→10`이고, Qwen 7개 typed failure set은 동일하다는 것이다.

이 분류는 사용자가 허용한 freeze 비교 범위에만 적용한다. method superiority, 다른 sample로의 일반화, sequential/Historical 효과, promotion은 주장하지 않는다. `scientific_promotion=false`.

## 14. Raw-free 산출물

- `p1r37-comparison-per-case.json/csv`: 80 case, endpoint/failure 분모 및 P1R36 exact-pair delta
- `p1r37-comparison-per-step.json/csv`: 631 accepted-prefix rows, phi/mask/transition/progress/P/capacity
- `p1r37-terminal-summary.json`: cell 및 global summary
- `p1r37-result-identities.json`: 8 root, 80 case, 631 step identity roots
- source bundle/patch/locks/manifest와 분석 receipt는 최종 create-once handoff package에 포함한다.

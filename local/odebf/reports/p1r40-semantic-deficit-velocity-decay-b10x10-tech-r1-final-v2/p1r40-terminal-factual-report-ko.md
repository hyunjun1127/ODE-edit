# P1R40 Semantic-Deficit Velocity-Decay Atomic B10×10 사실 보고서

## 1. 실행 및 무결성 사실

- instruction: `ODEEDIT-S05-P1R40-P1R38-SEMANTIC-DEFICIT-VELOCITY-DECAY-ATOMIC-V1`
- scientific parent: `6f48ac2800b257ceb16368fff5137212dfa6037f`
- scientific implementation: `a9444e3db6d7ff7b2e3ef05037e2298c2cce5101`
- execution checkpoint: `ca42fd9559951dc6337e6610379cfa222bf570ba` (tree `dc3046684bcb27f9ce2a30a288b2ece0f273a197`)
- Slurm array: `19663`; task 4/4 `COMPLETED`, ExitCode `0:0`.
- 시도 40/40 완료, accepted step 320/320, request-step 3200/3200.
- case technical failure 0, typed scientific failure 0, retry 0, history 0, cross-case state 0.
- 40/40 case에서 K8, action-freeze, terminal evaluator, byte/pointer W0 restore가 확인됐다.
- frozen stream/order는 P1R38 reference와 동일하며 case/model/arm 키 40/40이 매칭됐다.

## 2. 셀별 terminal 및 P1R38 산술 비교

| 모델 | 팔 | 완료 | Eff | Gen | Loc | Eff NLL | Eff margin | z8 full6 | W8 full6 | z-W gap | gamma<1 | gamma mean/p10 | P1R38 E/G/L | ΔE/ΔG/ΔL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | Neutral | 10/10 | 98/100 | 184/200 | 829/1000 | 0.629373 | 9.390940 | 0.436198 | 0.676719 | 0.240521 | 157/800 | 0.930661/0.671239 | 99/186/827 | -1/-2/+2 |
| llama3-8b-inst | Soft | 10/10 | 99/100 | 185/200 | 830/1000 | 0.519744 | 9.524162 | 0.434627 | 0.551907 | 0.117280 | 158/800 | 0.930305/0.667816 | 99/184/827 | +0/+1/+3 |
| qwen2.5-7b-inst | Neutral | 10/10 | 98/100 | 169/200 | 842/1000 | 0.575495 | 10.825799 | 0.350382 | 0.437670 | 0.087288 | 373/800 | 0.778693/0.291905 | 98/173/842 | +0/-4/+0 |
| qwen2.5-7b-inst | Soft | 10/10 | 99/100 | 168/200 | 844/1000 | 0.551185 | 10.823621 | 0.335317 | 0.404336 | 0.069019 | 374/800 | 0.783441/0.296235 | 98/175/842 | +1/-7/+2 |

Eff/Gen/Loc 델타는 `P1R40-P1R38` 동일 model/arm/case 합이다. NLL·margin·trajectory SHA의 byte equality는 요구되지 않았다.

| 모델 | 팔 | Gen NLL | Gen margin | Loc NLL | Loc margin | ΔEff NLL | ΔEff margin | ΔGen NLL | ΔLoc NLL | Δz8 full6 | ΔW8 full6 | Δz-W gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | Neutral | 2.470654 | 6.337267 | 9.504207 | 4.623406 | +0.035069 | -0.251476 | +0.095473 | +0.037796 | +0.017356 | +0.047734 | +0.030378 |
| llama3-8b-inst | Soft | 2.375602 | 6.411392 | 9.546734 | 4.669791 | +0.000110 | -0.174017 | +0.027764 | +0.044293 | +0.018210 | -0.001043 | -0.019253 |
| qwen2.5-7b-inst | Neutral | 3.294256 | 5.717952 | 10.048797 | 4.790475 | +0.060513 | -0.908395 | +0.277871 | +0.018566 | +0.028613 | +0.029191 | +0.000578 |
| qwen2.5-7b-inst | Soft | 3.297022 | 5.653056 | 10.062635 | 4.811617 | +0.024126 | -0.826027 | +0.271597 | +0.015433 | +0.037658 | +0.015885 | -0.021774 |

Official direct-z는 P1R38 transferred package에 포함된 별도 comparator다: Llama E/G/L=100/185/859, Qwen E/G/L=100/192/828 (각 분모 100/200/1000). P1R40-P1R38 델타와 혼합하지 않았다.

## 3. gamma 및 target/writer 계수 사실

- gamma<1 request-step: 1062/3200.
- held/reactivated/rejected request-step: 23/0/19.
- negative actual progress step: 1/320.
- 전 320 step에서 batch de-mean factor=10, `q_request=10*q_batched` max residual≤1e-10.
- 전 320 step에서 gamma target displacement 적용 1회, writer gamma 적용 0회.
- gamma controller 추가 model forward/backward/materialization 및 per-request Python model call은 모두 0.
- gamma<1인 모든 request-step에서 nominal/scaled column SHA가 달랐다.
- request별 D, q_signed, q_positive, gamma, nominal/scaled norm/hash, NLL, hold/accept, Adam 상태는 per-request 표에 기록됐다.

| 모델 | 팔 | D bin | request-step | gamma mean | gamma p10 |
| --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | Neutral | [0,0.05) | 18 | 0.855077 | 0.518880 |
| llama3-8b-inst | Neutral | [0.05,0.25) | 91 | 0.847051 | 0.454972 |
| llama3-8b-inst | Neutral | [0.25,1) | 156 | 0.850991 | 0.456552 |
| llama3-8b-inst | Neutral | [1,inf) | 535 | 0.970656 | 1.000000 |
| llama3-8b-inst | Soft | [0,0.05) | 19 | 0.855614 | 0.556429 |
| llama3-8b-inst | Soft | [0.05,0.25) | 88 | 0.856849 | 0.507850 |
| llama3-8b-inst | Soft | [0.25,1) | 159 | 0.844490 | 0.444256 |
| llama3-8b-inst | Soft | [1,inf) | 534 | 0.970619 | 1.000000 |
| qwen2.5-7b-inst | Neutral | [0,0.05) | 47 | 0.630475 | 0.296069 |
| qwen2.5-7b-inst | Neutral | [0.05,0.25) | 128 | 0.529979 | 0.170315 |
| qwen2.5-7b-inst | Neutral | [0.25,1) | 143 | 0.620314 | 0.213370 |
| qwen2.5-7b-inst | Neutral | [1,inf) | 482 | 0.906182 | 0.596286 |
| qwen2.5-7b-inst | Soft | [0,0.05) | 48 | 0.633059 | 0.280164 |
| qwen2.5-7b-inst | Soft | [0.05,0.25) | 127 | 0.546422 | 0.172203 |
| qwen2.5-7b-inst | Soft | [0.25,1) | 143 | 0.611785 | 0.225061 |
| qwen2.5-7b-inst | Soft | [1,inf) | 482 | 0.911795 | 0.625341 |

각 셀의 k8 target request 분포(mean/median/p90/worst/count<.05)는 cell-summary에 기록됐다.

## 4. writer·preservation 수치

| 모델 | 팔 | predicted Σ | actual Σ | realization | negative step | P terminal | P AUC | capacity terminal | energy terminal | functional-P | ΔP | Δcapacity | Δenergy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | Neutral | 123.550829 | 97.117773 | 0.786055 | 1 | 0.018599 | 0.007126 | 0.323990 | 0.449297 | 0.103302 | +0.000323 | +0.096191 | +0.131652 |
| llama3-8b-inst | Soft | 122.972877 | 98.365894 | 0.799899 | 0 | 0.016827 | 0.006792 | 0.153964 | 0.220533 | 0.094083 | +0.000081 | +0.036947 | +0.052878 |
| qwen2.5-7b-inst | Neutral | 132.547221 | 100.168091 | 0.755716 | 0 | 0.615977 | 0.254423 | 3.015336 | 1.454707 | 0.039164 | -0.044986 | -0.172581 | -0.090313 |
| qwen2.5-7b-inst | Soft | 131.733108 | 100.501429 | 0.762917 | 0 | 0.384189 | 0.165227 | 0.918364 | 0.633712 | 0.038821 | -0.038119 | -0.227121 | -0.140392 |

## 5. compute

| 모델 | 팔 | P1R40 edit-core/case(s) | P1R38 edit-core/case(s) | ratio | F | B | tokens | materialize |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3-8b-inst | Neutral | 73.111 | 99.291 | 0.7363 | 2200 | 1250 | 371778 | 80 |
| llama3-8b-inst | Soft | 77.762 | 108.388 | 0.7174 | 2200 | 1250 | 371778 | 80 |
| qwen2.5-7b-inst | Neutral | 90.228 | 136.178 | 0.6626 | 2200 | 1250 | 340727 | 80 |
| qwen2.5-7b-inst | Soft | 92.779 | 144.895 | 0.6403 | 2200 | 1250 | 340727 | 80 |

각 case의 terminal evaluator wall은 edit-core와 분리되어 per-case 표에 기록됐다. scheduler elapsed는 task별 18:15, 19:16, 21:54, 21:13이다.

scheduler MaxRSS(KiB): task0 7516248, task1 7487804, task2 10678764, task3 10748988.

## 6. 기록 경계

- stepwise heldout Eff/Gen/Loc: `NOT_RECORDED` (terminal-only evaluator 계약).
- P1R40 actual은 k1–k7 delayed reuse와 accepted-k8 terminal actual을 합쳐 320/320 기록됐다.
- P1R38 reference의 actual은 transition k1–k7만 기록됐고 terminal k8 actual은 `NOT_RECORDED`; actual 합/realization의 cross-run 델타는 산출하지 않았다.
- GPU utilization/time beyond component receipts: `NOT_RECORDED`.
- P1R38 reference는 transferred raw-free per-case/per-step/per-request tables만 사용했다.
- P1R38와의 scientific trajectory/hash equality는 `NOT_REQUIRED`; 표는 동일 case/model/arm의 수치와 산술 델타만 제시한다.
- `scientific_promotion=false`.

## 7. 실행 시도 보존

- 최초 array 19654는 `PURE_TECHNICAL_WRONG_NODE`로 model/result action 전 ExitCode2였고 immutable 보존됐다.
- 유일한 TECH-R1 delta는 execution node `server2` 고정과 `tech-r1` result/state/log namespace 격리다.
- P1R40 과학 source bytes는 scientific checkpoint 이후 변경되지 않았다.

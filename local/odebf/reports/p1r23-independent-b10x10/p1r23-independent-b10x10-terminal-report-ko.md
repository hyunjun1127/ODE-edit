# P1R23 Progress-Simplex 독립 B10×10 Atomic 최종 보고서

## 1. 범위와 결론

이 실험은 동일한 고정 100-request stream을 B1 100개로 평탄화하거나 100-edit sequential trajectory로 누적하지 않았다. 고정된 10개 B10 batch 각각을 동일 W0에서 시작하는 독립 Atomic edit로 실행하고, 각 batch 평가 후 W0를 exact restore했다.

- ODE: Llama/Qwen × BG/RS × Neutral/Soft = 8 long-lived jobs, 각 10개 독립 B10.
- 기준선: Official AlphaEdit Llama/Qwen = 2 long-lived jobs, 각 10개 독립 B10.
- 총 100개 독립 B10 trajectory: ODE 80 + AlphaEdit 20.
- ODE 성공 endpoint는 68/80, typed case failure는 12/80이다. AlphaEdit은 20/20 성공했다.
- 성공 ODE endpoint는 대체로 높은 efficacy를 보였지만, AlphaEdit보다 연속 target-new NLL과 Gen이 약했다. ODE가 보인 더 높은 locality count를 edit-strength-matched preservation gain으로 해석할 수 없다.
- Soft는 같은 batch에서 capacity를 거의 일관되게 낮췄지만 functional-P 개선은 모델/할당에 따라 혼재했다. 보편적 preservation 개선은 성립하지 않는다.
- 이 결과는 outcome-selected stream에서의 기계론적·기술적 비교이며 scientific promotion은 false다.

## 2. 고정 계약과 계보

- Instruction: `ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-INDEPENDENT-B10X10-V1`
- Scientific checkpoint: `a343d1f6967ef37763009b92d227ade85cd93de0`
- Execution checkpoint: `dd3751f998563e276dca68676a616cc4cb089691`
- Execution tree: `5fe371aa9cc5b4f6e0e3408ffa637c9987af44ce`
- Slurm array: `18853_[0-9]%4`, 모두 `COMPLETED 0:0`
- Frozen stream seal root: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6`
- Frozen 100-request order: `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- Numerical lock root: `f6097d532d8c88ce4eee16cc1d9d3e34efae4f590b99dd7143fa86fa5becccfb`
- Source manifest SHA/root: `4e2b10879c37eb04ea6d02493a1d7e6d85b799753a4fac9f0f21bdfeb340ebba` / `97c4a7009595462bbd37ad08be1a7d16579655ef981e3f5748fc67240eb9dbdb`
- Submission receipt SHA: `b2cdaf3b9a60461af933307de464a35575fe4e6e739383d4e4acb44ee89885ab`

교정 경계: 초기 B1×100 해석은 source/model/GPU/Slurm action 전에 폐기되었다. B1 job은 제출되지 않았고 본 checkpoint에는 B1 cardinality path가 없다. Exact Progress-Simplex router와 Atomic runtime 두 핵심 파일은 `a343d1f`와 byte-identical이다.

## 3. 실행 의미

각 case (j=1,dots,10)에서 다음 순서를 독립적으로 수행했다.

1. 해당 B10과 6개 controller context를 사용해 full-six target/slope와 exact P1R23 functional/structural P objective를 구성한다.
2. K=8, h=1/8, tau=1 Progress-Simplex trajectory를 실행한다.
3. action-freeze 뒤 frozen evaluator로 W0와 edited endpoint를 읽는다.
4. 동일 parameter pointer를 유지한 채 W0 bytes를 복원하고 검증한다.
5. case-level terminal/manifest 또는 typed failure receipt를 create-once 저장한다.

History는 항상 OFF였다. history append/replay/sketch, functional-H, structural-H, cross-case controller/weight state의 count와 influence는 모두 0이다. 한 case의 실패는 같은 case를 retry/impute하지 않고, W0 복원 후 다음 case를 계속했다.

## 4. Endpoint 집계

아래 분모는 **scientific endpoint가 존재하는 성공 case만** 포함한다. 실패 case는 imputation하지 않았다. E/G/L 분모는 성공 B10당 10/20/100이다. NLL은 성공 B10별 evaluator mean의 산술평균이다.

| Model | Method | 성공 | E | G | L | E new-NLL | G new-NLL | structural-P | functional-P | capacity | actual/pred progress | edit-core s/B10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | BG-Neutral | 9/10 | 88/90 | 159/180 | 785/900 | 0.4517 | 2.3316 | 0.000227767 | 0.0155890 | 0.06694 | 0.2443/0.4791 | 122.80 |
| Llama | BG-Soft | 8/10 | 79/80 | 143/160 | 694/800 | 0.3427 | 2.3900 | 0.000202631 | 0.0168258 | 0.05483 | 0.2500/0.4747 | 138.21 |
| Llama | RS-Neutral | 7/10 | 70/70 | 124/140 | 603/700 | 0.1594 | 2.0022 | 0.000235454 | 0.0471097 | 0.07375 | 0.1380/0.3093 | 125.40 |
| Llama | RS-Soft | 9/10 | 90/90 | 163/180 | 788/900 | 0.1182 | 1.8981 | 0.000207714 | 0.0394305 | 0.05945 | 0.1371/0.2327 | 120.78 |
| Llama | Official AlphaEdit | 10/10 | 100/100 | 185/200 | 859/1000 | 0.0012 | 1.7627 | N/R | N/R | N/R | N/R | 29.74 |
| Qwen | BG-Neutral | 8/10 | 75/80 | 127/160 | 682/800 | 0.8229 | 3.8976 | 0.0116099 | 0.00610915 | 0.61211 | 0.4720/1.1248 | 132.69 |
| Qwen | BG-Soft | 9/10 | 85/90 | 143/180 | 770/900 | 0.7942 | 3.8036 | 0.00654414 | 0.00471253 | 0.29370 | 0.3431/0.9059 | 143.05 |
| Qwen | RS-Neutral | 8/10 | 79/80 | 131/160 | 666/800 | 0.4347 | 3.4588 | 0.00762773 | 0.00364018 | 0.40554 | 0.2219/0.4230 | 139.28 |
| Qwen | RS-Soft | 10/10 | 99/100 | 164/200 | 845/1000 | 0.3329 | 3.3281 | 0.00428203 | 0.00368743 | 0.19485 | 0.2058/0.3850 | 150.00 |
| Qwen | Official AlphaEdit | 10/10 | 100/100 | 192/200 | 828/1000 | 0.0326 | 1.7722 | N/R | N/R | N/R | N/R | 23.44 |

공통 W0 전체 10 B10 기준:
- Llama: E/G/L = 13/100, 28/200, 892/1000.
- Qwen: E/G/L = 11/100, 36/200, 849/1000.

AlphaEdit 대비 matched-success case에서 ODE의 locality count는 자주 1–2개 높았지만, efficacy/generalization new-NLL은 거의 항상 AlphaEdit보다 컸다. 예를 들어 Qwen BG-Neutral은 matched 8 batch에서 AlphaEdit 대비 E new-NLL +0.7889, G new-NLL +2.1595였고 Gen count는 평균 -3.25/B10였다. 따라서 locality count 증가는 보존 이득으로 승격할 수 없다.

## 5. Soft–Neutral matched-batch 비교

W/T/L은 Soft 기준 win/tie/loss다. NLL/P/capacity는 작을수록 win이다. Δ는 Soft−Neutral 평균이다.

| Model/Alloc | matched n | E W/T/L | G W/T/L | L W/T/L | ΔE new-NLL | ΔG new-NLL | functional-P W/T/L; Δ | capacity W/T/L; Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama BG | 7 | 0/7/0 | 0/7/0 | 0/7/0 | -0.0161 | +0.0218 | 4/0/3; +0.000218 | 6/0/1; -0.01569 |
| Llama RS | 7 | 0/7/0 | 1/6/0 | 0/7/0 | -0.0108 | +0.0035 | 4/0/3; +0.000997 | 6/0/1; -0.01184 |
| Qwen BG | 7 | 1/5/1 | 2/3/2 | 1/5/1 | -0.0403 | +0.1634 | 3/0/4; -0.001418 | 7/0/0; -0.27906 |
| Qwen RS | 8 | 0/8/0 | 1/5/2 | 2/5/1 | -0.0240 | +0.0864 | 4/0/4; +0.000556 | 8/0/0; -0.21347 |

FACT:
- 모든 성공 Soft step에서 preservation router가 결정에 관여했다: Llama BG 64, Llama RS 72, Qwen BG 72, Qwen RS 80 step.
- equality residual 최대값은 (7.61	imes10^{-15})였고 H influence는 0이었다.
- Soft의 평균 energy ratio는 Llama BG/RS 0.851/0.868, Qwen BG/RS 0.694/0.638이었다.
- Soft는 capacity를 matched case의 27/29에서 낮췄다.
- 그러나 functional-P는 Llama BG 4/7, Llama RS 4/7, Qwen BG 3/7, Qwen RS 4/8에서만 개선됐다.

INFERENCE:
- Soft allocation은 same-state constraint 안에서 실제 coefficient를 바꾸고 energy/capacity를 줄였지만, terminal functional-P와 Gen에 대한 보편적 이득은 없다.
- Neutral/Soft가 동일 predicted-progress 계약을 사용해도 BF16 actual realization과 trajectory가 달라지므로 endpoint 차이는 단일-step router causality가 아니다.

## 6. BG–RS와 B10 cardinality

BG와 RS는 B10에서 붕괴하지 않았다. Llama에서 RS는 matched Neutral 7개 모두 E/G new-NLL을 낮췄고, Qwen에서도 RS가 대체로 더 낮은 E/G new-NLL을 보였다. 이는 B1 cardinality가 아니라 원래 B10 allocation 정의를 실행했기 때문이다. 다만 성공 case 집합이 다르므로 arm-level aggregate를 직접 순위로 해석하면 selection bias가 생긴다.

## 7. 실패 격리와 완전성

ODE typed case failure 12건은 모두 `ODEBFContractError`였고 accepted k3–k7 사이에서 발생했다.

- Llama: BG-N case3; BG-S case1/5; RS-N case1/3/9; RS-S case3.
- Qwen: BG-N case3/9; BG-S case8; RS-N case1/9; RS-S 없음.
- 실패마다 exception message는 raw-free SHA로만 저장됐다.
- 12/12 모두 W0 pointer/byte restore PASS, retry=0, imputation=0, next-case continuation=true.
- 실패 내부의 세부 first-false solver/certificate component는 case failure schema에 직렬화되지 않아 NOT_RECORDED다.
- 성공 endpoint 88개는 모두 action-freeze와 case terminal/manifest를 가졌고, 10개 job 모두 최종 W0 restore/manifest PASS였다.

Job이 `COMPLETED 0:0`인 것은 case-level typed failure를 격리해 다음 독립 B10을 계속하도록 사전 고정했기 때문이다. 이는 실패 case를 scientific success로 간주한다는 뜻이 아니다.

## 8. Compute와 메모리

성공 ODE case 하나의 online ledger는 26 model forwards, 16 backwards였다. 성공 ODE 68개 합계는 1,768 F / 1,088 B / 1,760,158 processed tokens이며, 성공 case edit-core 합계는 9,139.04 s다. 실패한 partial trajectory의 내부 F/B/token 합산은 job terminal에 완전하게 중복 없는 형식으로 기록되지 않았으므로 전체 80 case authoritative total은 NOT_RECORDED다.

Official AlphaEdit 20개 성공 case의 edit-core 합은 531.84 s다. AlphaEdit 내부 F/B/token counters는 NOT_RECORDED이며 wall-time만 비교할 수 있다.

| Task | Cell | elapsed s | MaxRSS KiB |
|---:|---|---:|---:|
| 0 | Llama BG-N | 1403 | 7,296,400 |
| 1 | Llama BG-S | 1543 | 7,361,896 |
| 2 | Llama RS-N | 1324 | 7,271,260 |
| 3 | Llama RS-S | 1436 | 7,309,424 |
| 4 | Llama AlphaEdit | 514 | 15,082,980 |
| 5 | Qwen BG-N | 1536 | 10,670,864 |
| 6 | Qwen BG-S | 1632 | 10,561,452 |
| 7 | Qwen RS-N | 1560 | 10,608,996 |
| 8 | Qwen RS-S | 1785 | 10,604,020 |
| 9 | Qwen AlphaEdit | 476 | 26,209,948 |

ODE 평균 edit-core는 120.78–150.00 s/B10, AlphaEdit은 23.44–29.74 s/B10였다. 이는 end-to-end matched scientific equivalence ratio가 아니라 관측된 edit-core wall 비교다.

## 9. FACT / INFERENCE / NOT_RECORDED

### FACT

- B1×100은 실행되지 않았다. 정확히 10개 whole-B10 batch를 W0에서 독립 실행했다.
- Scheduler 10/10 `COMPLETED 0:0`; job terminal/manifest/W0 restore 10/10 PASS.
- ODE endpoint 68/80, typed isolated failure 12/80; AlphaEdit endpoint 20/20.
- 성공 ODE는 K8/tau1, full-six, H-OFF, no retry/backtracking/early stop였다.
- Soft는 capacity를 강하게 낮췄지만 functional-P 개선은 혼재했다.
- AlphaEdit은 두 모델에서 E 100%, Gen 92.5%/96.0%로 ODE aggregate보다 강했다.

### INFERENCE

- Independent B10 조건에서도 exact Progress-Simplex는 강한 편집을 자주 만들지만, AlphaEdit 수준의 연속 NLL/Gen을 안정적으로 재현하지 못했다.
- Qwen에서 RS가 BG보다 편집 강도가 좋았고, Soft는 capacity를 낮췄지만 preservation Pareto dominance는 아니다.
- 실패율 15%는 이 method package의 robustness 한계다. 세부 solver 원인을 재분류하려면 더 풍부한 failure certificate schema가 필요하지만, 이 보고서에서 결과를 재실행하거나 보정하지 않았다.

### NOT_RECORDED

- 실패 12건의 authoritative first-false solver/certificate component.
- 실패 partial trajectory를 포함한 완전한 whole-job F/B/token 합계.
- Official AlphaEdit 내부 F/B/token counters.
- z-oracle endpoint는 본 독립-B10 evaluator schema에 기록되지 않았다.
- fresh promotion set, confidence-calibrated gate, lifelong/sequential retention.

## 10. Claim boundary

이 데이터는 이전 선택 과정과 연결된 고정 100-request stream에서 얻은 raw-free mechanistic/descriptive 결과다. 독립 B10 Atomic 성능만 측정하며 sequential retention이나 lifelong editing을 말하지 않는다. Soft의 낮은 capacity 또는 더 높은 locality를 actual semantic edit strength가 맞지 않은 상태에서 preservation gain으로 부르지 않는다. `scientific_promotion=false`.


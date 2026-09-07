# AlphaEdit JV: Server4 병렬 작업과 Llama 개선 중심의 연구 계획

작성일: 2026-09-07 KST. 상태: **DESIGN_ONLY — 신규 구현·GPU 실행·scientific promotion 없음.**

## 0. 권고안

**지금은 Qwen의 성과를 기준 결과로 고정하고, Llama의 두 실패 원인을 분리해 해결한 뒤 method와 ablation을 닫는 단계다.** L8 분산을 성공 조건으로 강제하지 않는다. 목표는 신규 editability를 유지하면서 실제 locality와 유효한 과거 edit retention을 개선하는 것이다.

진행 중인 `odeedit_orbode_cum_s4_r1`, `odeedit_alpha_l8_s4_takeover`는 그대로 둔다. 이들과 중복되지 않는 병렬 작업의 순서는 다음과 같다.

1. **최우선: Llama B10 near-stall의 exact-state normalization/response 진단.** 이미 완료된 λ shadow로 해결되지 않은 문제다.
2. **동시에 준비: Llama/Qwen 각각의 작은 Euler 실험.** 고정 (T)의 (N) refinement와 고정 (h)의 (T) exposure를 분리한다.
3. **별도 소형 mechanism 실험: historical key 외에 과거 edit의 현재 margin response를 추가.** 외부 reference bank나 request별 hard barrier 없이, 기존 5차원 NNLS에 기능적 정보를 넣는다.
4. **Llama의 정상 상태 PS/NS 열세 개선.** B10을 고쳤다는 이유만으로 종료하지 않는다. 정상 상태의 좁은 λ/시간 운영점과 필요하면 history variant를 비교한다.
5. **두 Server4 결과가 봉인되면 claim과 최종 arm 수를 줄인다.** L8-only가 JV와 같다면 다층 mixing의 필요성 주장을 줄이고, 반복 실현·response 제어의 기여를 검증한다.
6. **선택된 Llama 후보의 독립 audit → 두 모델 method lock → 최소 ablation → 더 긴 lifelong.** 지금부터 다수의 1k/10k full-chain sweep을 시작하지 않는다.

이 보고서의 설정값은 실행을 위한 **제안**이다. 기존 실험의 lock을 바꾸는 지시나 새 GPU-hour 사용 승인이 아니다. 작업 목록은 [task-matrix.csv](task-matrix.csv), source와 제안값은 [design-manifest.json](design-manifest.json)에 함께 기록했다.

## 1. 근거와 작업 경계

### 1.1 확인한 source

| 구분 | 고정 reference |
|---|---|
| 작성 시 원격 main | `e25a5685a4ad7f8606a8c265abf070b0d77a83d1` |
| 기존 Alpha JV scientific runtime | `77358b1546d1baf83b3e251afcce663b08d7bfd7` |
| 완료 O/JV 네 chain publication | `0d0a0131e4a6a2a645dfa6530377d420a084d136` |
| L8 takeover source | `44602a1a80554c67da0ef9646b43d843104785f2` |
| ORBODE cumulative source | `8610faf0e114059a5e08116f1164baf31c573e80` |

완료 실험의 수치는 main의 [SH1 사실 보고서](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/servers/server1/alpha-jv-sequential1000-layer-review-2026-09-07-v1/factual-report-ko.md)와 [GH mechanism 통합 분석](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)을 근거로 한다. 이 보고서는 그 완료 자료와 immutable source를 검토한 후속 설계이며, 새 GPU 재현 결과가 아니다.

### 1.2 현재 작업에서 하지 않은 일

- Server4 scheduler/process/GPU를 조회·수정하거나 live partial output을 읽지 않았다.
- 두 task의 실제 진행률, 완료 여부, process에 로드된 source의 동일성을 확인했다고 주장하지 않는다. 아래 설명은 **봉인된 실행 계약**의 비교다.
- 새 experiment, checkpoint evaluation, source patch, Slurm 제출을 하지 않았다. 이 디렉터리의 설계 문서만 새로 작성했다.
- 기존 dirty worktree와 미추적 보고서는 보존했다. 현재 local HEAD가 main보다 뒤에 있으므로 실행 준비 때 별도 worktree에서 source를 다시 pin해야 한다.

향후 병렬 실행도 별도 worktree/branch/process/output과 격리 GPU를 사용한다. Server4 두 task의 모델 객체·covariance·projector·history·checkpoint를 공유하지 않는다. 독립 자원이 없으면 `WAITING_FOR_ISOLATED_RESOURCE`로 두고 CPU 준비까지만 진행한다.

이번 신규 연구 범위는 **AlphaEdit 우선**이다. MEMIT용 새 history cache를 도입하거나 새 MEMIT 실험을 늘리지 않는다. 이미 진행 중인 ORBODE의 MEMIT arm은 그 계약 그대로 완료하도록 둔다.

### 1.3 「AlphaEdit JV sequential 1,000: GH 통합 분석 및 Sweep 후보」의 반영

사용자가 다시 지정한 이 문서의 **통합 v2 §13과 동일 CSV/정책 JSON까지** 기준으로 삼았다. 기존 10개 configuration/20개 model×configuration 후보를 폐기하거나 새로운 무관한 sweep으로 교체하지 않는다. 다만 이번 목적이 **Server4와 병렬인 Llama 개선 실험**이므로 실행 순서를 아래처럼 재배치한다.

| GH 후보 | 이번 병렬 계획의 처리 |
|---|---|
| JV-BASE | 신규 common-entry development의 공통 기준. 완료 1k를 중복 실행하는 뜻이 아님 |
| JV-LAM-001 / JV-LAM-1 | 정상 Llama의 우선 coarse operating-point 비교; Qwen은 대응 sensitivity/regression 확인 |
| JV-LAM-00316 / JV-LAM-0316 | 같은 raw state에서 CPU shadow는 유지. Actual trajectory는 coarse 결과와 budget에 따라 2차 배치 |
| JV-RES-N2 / JV-RES-N8 | 두 모델 필수 fixed-T 실험 E1로 승계 |
| JV-RES-N16 | GH와 같이 조건부. N8의 정보·비용·roundoff를 먼저 확인 |
| JV-HOR-T1 / JV-HOR-T4 | 두 모델 필수 fixed-h 실험 E1로 승계; 가능한 prefix 계산 재사용 |
| NRMS_ENTRY | GH의 별도 science ablation 정의를 그대로 L1/L2에 적용 |
| NNUM_OBSERVED_NOISE | Noise 측정 근거가 있을 때만 후순위. 임의 floor는 도입하지 않음 |

기존 후보 전체를 [gh-candidate-disposition.csv](gh-candidate-disposition.csv)에 ID 그대로 연결했다. GH의 완료 1,040개 CPU λ shadow는 재사용하며 새 GPU 결과로 세지 않는다. 5개 λ를 양 모델에서 모두 1k chain으로 돌리면 기존 runtime 단순 proxy로도 약40.9 GPUh라는 GH의 비용 경고를 유지한다.

변경점은 **해석이나 원본 target이 아니라 실행 우선순위**다. GH의 Llama B10 진단을 P0으로, 사용자 요청의 모델별 Euler 축을 필수 병렬 작업으로 올리고, key-only의 한계를 검증하는 functional-history observer를 별도 추가했다. History variant를 기존 native-only sweep의 일부처럼 기록하지 않는다.

이번 계획은 Server4 실험의 후속 task로 대기만 하는 구조가 아니다. L0/L1/L2/E0/E1/H0는 **두 live task의 종료가 선행 조건이 아니다.** 필요한 것은 각자의 sealed input·독립 자원·실행 budget이다. 완료 결과를 기다리는 것은 C0의 최종 join과 method/claim lock뿐이다.

### 1.4 두 모델 차이를 architecture 원인 하나로 확정하지 않는다

GH는 원래 AlphaEdit compute-z 설정부터 두 모델이 다름을 확인했다.

| 원본 고정값 | Llama | Qwen |
|---|---:|---:|
| v_num_grad_steps | 25 | 25 |
| v_lr | .1 | .5 |
| v_loss_layer | 31 | 27 |
| v_weight_decay | .5 | .001 |
| clamp_norm_factor | .75 | 4 |
| kl_factor | .0625 | .0625 |
| Native AlphaEdit L2 | 1 | 1 |

각 모델 안에서는 Official/JV가 같은 pinned target stage를 쓰므로 유효한 method 비교다. 그러나 Llama↔Qwen의 차이에는 architecture뿐 아니라 target optimization의 원래 operating point도 섞여 있다. 이번 병렬 run에서는 이 원본값, contexts/tokenization/padding/position, stock early-stop .05, P, native L2, history finalization을 **통일하거나 변경하지 않는다.** Sweep λ는 native L2나 compute-z regularization이 아니라 우리 response-controller 계수다. [GH 통합 v2 §13.5](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)

## 2. 현재 결과: 긍정적 성과와 해결할 문제

### 2.1 Qwen의 이득은 인정하되, 정확히 무엇이 좋아졌는지 구분한다

다음은 warm B10 pilot이 아니라 **B100×10 sequential chain의 최종 (W_{10})에서 모든 1,000 edit를 평가한 결과**다. RS/PS/NS는 두 정답의 teacher-forced NLL 선호이며 tie는 실패다. 자유 생성 정확도가 아니다.

| 모델 | Arm | RS /1,000 | PS /2,000 | NS /10,000 |
|---|---|---:|---:|---:|
| Llama | Official | 1,000 | 1,910 | 7,584 |
| Llama | JV | 923 | 1,690 | 7,259 |
| Qwen | Official | 992 | 1,887 | 6,978 |
| Qwen | JV | 997 | 1,894 | 7,377 |

Qwen의 가장 강한 결과는 NS **+399/10,000 = +3.99pp**다. 동일 neighborhood 문항에서 개선 1,289개, 악화 890개다. (W_0)에서 성공한 문항의 후속 손실은 1,838→1,390으로 448개 감소했다. Neighborhood true-answer 평균 NLL도 6.8350→6.3312로 개선됐다. 단순히 competing answer를 약화해 NS만 높인 현상으로 설명되지 않는다.

다만 (W_0) NS는 84.63%, JV 최종 NS는 73.77%다. 따라서 **Official보다 덜 잃었다**는 결과이지 원래 지식을 손실 없이 보존했다는 결과가 아니다. Rephrase-new 평균 NLL은 2.1800→2.2830, p90은 6.3888→7.3729로 악화한다. Qwen이 모든 품질 축에서 우월하다고 해석하지 않는다. [완료 분석 §7](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)

후속 개발에서는 현재 Qwen JV를 **고정 regression reference**로 보존한다. Llama를 개선하려고 공통 코드를 바꿀 때 Qwen 이득이 사라지는지 별도로 확인한다. 기존 Qwen final 결과를 반복해서 tuning에 사용하는 대신 새 development/audit stream을 분리한다.

### 2.2 Llama에는 서로 다른 두 문제가 있다

| 문제 | 확인된 증거 | 필요한 실험 |
|---|---|---|
| B10 신규 edit near-stall | 최종 RS 실패 77건 중 75건이 B10 at-write 실패. B1–B9는 at-write 900/900, final 898/900 | Exact-entry normalization/JVP/physical-write 진단 |
| 정상 상태에서도 PS/NS 열세 | B1–B9의 모든 batch에서 PS/NS가 Official보다 낮음. B1 PS 185→177, NS 869→858 | 정상 상태 운영점·layer support·기능적 history 제어 비교 |

B1–B9 current PS 합계는 Official 1,729/1,800, JV 1,663/1,800이며 NS는 7,300/9,000 대 7,041/9,000이다. **B10 rescue만으로 Llama method가 해결됐다고 판단하면 안 된다.** 반대로 최종 RS -77을 모두 historical forgetting으로 부르는 것도 틀리다. [완료 분석 §8](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)

### 2.3 왜 L8에 집중하면서 Qwen은 좋아질 수 있는가

현재 Φ는 final logits가 아니라 **deepest editable layer L8의 activation readout**이다. L8은 모델의 마지막 layer라는 뜻이 아니다. 이 readout에 대한 response–residual cosine 중앙값은 정상 Llama에서 L4 .251/L8 .9977, Qwen에서 L4 .235/L8 .9960이다. 현재 목적식이 L8을 선호하는 것은 자연스럽다.

실제 batch-net Frobenius energy의 L8 share는 Llama B1–B9 99.916–99.946%, Qwen 99.876–99.997%다. Qwen의 전체 batch-net energy는 **모든 batch에서 Official의 1.256–3.135배**인데, L4–L7 energy는 Official의 0.0060–0.5402%다.

따라서 현재 결과와 맞는 가설은 다음이다.

> Qwen에서는 early-layer action을 거의 피하고 target readout에 직접 작용하는 L8에 쓰는 경로가 유리했을 수 있다. 이 이득은 전체 write를 단순히 작게 만든 결과로 설명되지 않는다.

이것은 아직 원인 확정이 아니다. 작은 early-layer mixture, 현재 response 측정, 반복 실현, native metric이 함께 달라졌다. 또한 Qwen의 current NS는 첫 두 batch에서 Official보다 낮고 이후 좋아졌다. **L8 집중 자체가 언제나 locality를 개선한다**는 주장도 지지되지 않는다. [완료 분석 §4–7](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)

## 3. 두 Server4 task가 답하는 질문과 claim 변화

### 3.1 진행 중인 task는 서로 다른 실험이다

| 항목 | `odeedit_alpha_l8_s4_takeover` | `odeedit_orbode_cum_s4_r1` |
|---|---|---|
| 핵심 질문 | 현재 JV에서 L4–L7 support가 실제로 필요한가? | Ordered allocation/response/refresh가 retention에 어떤 차이를 만드는가? |
| 모델·방법 | Llama/Qwen AlphaEdit, L8-only 2 chains | Llama/Qwen × MEMIT/AlphaEdit × 5 arms, 20 chains |
| 길이 | Fresh (W_0,M_0)에서 B100×10 | Fresh entry에서 B100×10 |
| Arm | L8_ONLY_NATIVE; 완료 O/JV를 직접 비교군으로 사용 | O, QCL, NQFIX, ORBFH, JAC |
| 시간 | Joint (T=2,N=4,h=.5) | Ordered dynamic arms (T=1,N=4,h=.25) |
| Sample | 완료 Alpha JV/O와 동일 1,000개 | 별도 reserved stream; Alpha JV sampler가 이 inventory를 제외 |
| 평가 | Rewrite all-seen 매 batch; all-seen PS/NS W1/W5/W10 | 모든 W1–W10에서 all-seen RS/PS/NS |
| 한계 | Barrier-off가 아니며 stock L8 one-shot도 아님 | JV joint/native controller를 직접 ablate하는 실험이 아님 |

L8 task는 동일 sample·계약을 갖는 support restriction 비교다. 하지만 각 chain은 자신의 W/M/z를 누적하므로 **B2 이후에도 arm 간 동일 z라고 말하면 안 된다.** 동일 target을 공유하는 것은 별도 common-entry mechanism 실험에서만 보장한다.

ORBODE의 QCL↔NQFIX는 per-visit quota의 효과, NQFIX↔ORBFH는 current-response scalar의 효과, ORBFH↔JAC는 sweep 내부 state refresh의 효과를 주로 분리한다. ORBHit은 별도의 여섯 번째 sequential chain이 아니라 derived prefix observation이다.

두 실험의 data/order가 다르므로 ORBODE의 Official 점수를 Alpha JV의 paired baseline으로 대체하거나 분모를 합치지 않는다. 둘 다 1,000-edit 실험이지 10k lifelong 실험이 아니다. [L8 takeover source](https://github.com/hyunjun1127/ODE-edit/blob/44602a1a80554c67da0ef9646b43d843104785f2/project/run_scripts/alpha_native_response_ode_v31_sequential/server4_takeover.py), [ORBODE 측정 계약](https://github.com/hyunjun1127/ODE-edit/blob/8610faf0e114059a5e08116f1164baf31c573e80/project/run_scripts/ordered_response_barrier_ode/cumulative_claim_measurement_map.md)

### 3.2 결과별 claim 판정

| 완료 후 관측 | 남는 주장 | 줄여야 하는 주장 / 다음 조치 |
|---|---|---|
| Qwen L8-only≈JV, 둘 다 Official보다 좋음 | Concentrated iterative realization이 이득 대부분을 설명할 가능성 | Adaptive multi-layer mixing의 필수성은 낮아짐. L8 one-shot/반복/penalty 분리 필요 |
| L8-only<JV, 차이가 독립 sample에서도 반복 | 작은 early-layer mixture도 유용할 가능성 | Native history 또는 barrier의 단독 인과효과로 바로 귀속하지 않음 |
| L8-only>JV | 불필요한 support가 비용·성능 손해일 가능성 | Full joint mixing을 main으로 고집하지 말고 단순화 후보로 검토 |
| Llama L8-only도 B10 near-stall | Joint interference만으로 실패를 설명하기 어려움 | 공통 normalization/target/readout 후보를 우선. 원인 확정은 별도 counterfactual |
| Llama L8-only는 B10을 통과 | Support/trajectory 상호작용 후보 강화 | “N0 하나만이 원인”으로 단정하지 않음; 같은 W/M/z에서 비교 |
| ORBFH>NQFIX | Ordered setting에서 response amplitude 제어의 실용 가치 | JV native barrier의 증거로 대체하지 않음 |
| ORBFH>JAC | 해당 setting의 within-sweep state feedback 가치 | 단일 N의 결과로 continuous ODE 필요성·수렴을 주장하지 않음 |
| Current 개선, final/cohort retention 악화 | 신규 실현과 누적 보존 간 trade-off | Lifelong improvement claim 보류 |
| Layer debt와 후속 loss가 연관 | Request/cohort-aligned predictive evidence | Action/time/entry margin 등을 통제해도 observational evidence. 원인 증명은 아님 |
| 1k에서 L8 집중과 retention 이득이 유지 | 해당 길이·stream에서 집중이 즉시 붕괴하지 않음 | 5k/10k 안정성은 별도 확인 |

`≈`는 최종 숫자를 본 후 임의로 정하는 것이 아니다. 동일 request의 paired transition, NLL 분포, 독립 order와 사전 정한 practical tolerance를 함께 사용해야 한다. 현재 한 chain의 prompt 수를 독립 반복 수처럼 취급하지 않는다.

### 3.3 Barrier와 multi-layer claim은 같은 주장이 아니다

현재 JV는 다음 response regularization을 푼다.

\[
c^\star=\arg\min_{c\ge0}\frac12\|e-\Psi c\|^2+\frac\lambda2c^\top Gc,
\qquad F=\mathcal Tc.
\]

여기서 도출되는 (b=V_0-V-\lambda\int Q_N(F)dt)는 **native action과 target progress 사이의 dissipation certificate**다. 과거 정답이나 NS의 safe set을 직접 제약하지 않는다. 실제 코드에서 finite-step (b) 위반을 이유로 write를 거부하는 별도 veto도 없다.

정확한 current derivative와 고정 metric 아래의 연속시간 정리는 유효하지만, 기존 finite Euler 결과는 Qwen 40/40 node, Llama 38/40 node에서 실제 \(\Delta b\ge0\)였다. 이것을 “모든 Euler write의 functional safety가 보장됐다”로 바꾸면 안 된다. [완료 분석 §6](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)

L8-only에서도 certificate 자체는 사라지지 않는다. 다만 고정 state에서 하나의 유효 direction만 있으면

\[
F_\lambda(W)=a_\lambda(W)D_8(W),\qquad a_\lambda(W)>0
\]

이므로, 같은 entry의 연속시간 흐름은 regular segment에서 \(\lambda\)에 따라 **같은 geometric orbit을 다른 속도로** 지날 수 있다. 이 경우 λ 효과를 곧바로 “더 안전한 방향 선택”이라고 해석할 수 없다. 유한 Euler 오차, zero-field, endpoint exposure, 이후 batch의 W/M/z 분기는 별도 효과다.

또한 joint update라고 해서 cross-effect가 사라지는 것은 아니다. (H=\Psi^\top\Psi)의 off-diagonal은 target response의 중복·상쇄를 반영하고 다음 node에서는 변경된 state를 관측한다. 그러나 이것이 한 finite step 내부의 모든 nonlinear layer interaction을 정확히 모델링한다는 뜻도 아니다. Current-response/actual-response 차이와 fixed-T refinement가 필요한 이유다.

## 4. 병렬 작업 배치: 새로 무엇을 올릴 것인가

### 4.1 작업 패키지

| ID / 우선순위 | 책임 범위와 질문 | 초기 계산 범위 | 의존성 / 종료 조건 |
|---|---|---|---|
| L0 / P0 | 완료 Llama B10 entry의 복원 가능성 확인 | Sealed inventory/source CPU 검사 | Exact state 가능/불가를 구분; live 파일 금지 |
| L1 / P0 | Tiny-anchor가 실제 Gram/field를 지배하는가? | 공통 entry 1개, raw 5-direction JVP + CPU N0/NRMS | L0; 없으면 명시된 analogue |
| L2 / P0 | Weighting만 바꾸면 실제 near-stall이 바뀌는가? | 같은 entry의 N0/NRMS 2×4-node trajectory | L1; 원래 실패를 대체하지 않음 |
| E0 / P1 병렬 | (\lambda,T,N,h) 일관된 runtime config 준비 | CPU 단위시험, 기존 primitive 유지 | Live source와 별도 구현; source-exact 재현 |
| E1 / P1 병렬 | 두 모델의 resolution과 exposure 민감도 | 모델별 공통 B100 fixture 1개, 3 paths/18 nodes | E0, 격리 GPU; 정상 Llama fixture 사용 |
| H0 / P1 병렬 | 과거 margin/Jacobian이 native key 정보 외에 선택 가치를 갖는가? | Llama warm fixture, replay 8개, common-state observer | 신규 commit-reference 수집; history actuator는 추가하지 않음 |
| H1 / P2 | Functional metric/anchor가 실제 edit-retention trade-off를 개선하는가? | 공통 warm entry의 base/metric/anchor 각 4 nodes | H0, N0 판단; 후보 한 설정으로 제한 |
| S1 / P2 | 정상 Llama 품질 열세가 운영점 문제인가? | λ .01/.1/1의 same-state screen 후 제한된 actual runs | E1; dense full-factorial 금지 |
| C0 / P1 CPU 병렬 | 두 live task 완료 후 claim join 준비 | Sealed-table schema·paired cohort 분석 설계 | 결과 수신 전 결론 미리 채우지 않음 |
| A0 / P3 | 최종 후보 선택·독립 audit·ablation lock | 별도 audit + 짧은 sequential qualification | Llama 두 문제, E1, live 결과를 함께 판단 |
| LL / 보류 | 1k→5k/10k에서 L8/retention 유지 여부 | 승격 후보와 Official 중심 | A0 이후 별도 비용·sample·resume 계약 |

실제 GPU 큐 우선순위는 L1/L2 → Llama E1 → H0/H1 또는 정상 운영점 → Qwen regression/E1이다. 단, 별도 GPU가 있으면 서로 독립인 정상-state Euler 준비와 history observer 준비는 진행할 수 있다. **Server4 종료를 기다려야 하는 것은 L8/mixing claim의 최종 판정이지, Llama 진단의 시작이 아니다.**

### 4.2 지금 시작하지 않을 작업

- 동일 sample의 L8-only 1k 재실행: Server4에서 이미 진행 중이다.
- ORBFH/JAC/NQFIX/QCL의 별도 retention full run: 현재 cumulative task와 중복된다.
- λ×T×N×normalization×history-size의 full grid.
- Llama B10을 보기 위한 무조건 W0→B9 replay.
- 새 external Wikipedia/reference bank, causal layer scoring, learned router, dynamic-z.
- 과거 request별 output/history hard constraint, rollback/backtracking 추가.
- Native metric 대신 결과를 보고 임의 projector/identity/ridge로 교체.

## 5. Llama P0: 원인 진단 계약

### 5.1 현재 가장 강한 가설과 반증 가능성

B10 case4228의 entry residual norm은 \(3.5747\times10^{-5}\), 두 번째 최소값은 4.3894, median은 5.2684다. Baseline compute-z의 delta는 0이며 초기 loss .043이 stock early-stop .05보다 작았다. 그러나 별도 FP32 capture에서 tiny-positive residual이 남아 N0의 active row가 됐다.

N0는 대략

\[
e_i(W)=\frac{R_i(W)}{\sqrt{B_a}s_i},\qquad s_i=\|R_i(W_{\rm entry})\|
\]

이므로 이 row의 inverse weight는 약 2,797.4다. 이것이 실제 JVP energy까지 지배하는지는 **raw requestwise response가 없으므로 아직 미확정**이다. Inverse weight만으로 H 기여율을 대신 계산하면 안 된다.

B10 H의 eigenvalue는 약 \(1.66\times10^6\)–\(5.30\times10^6\), condition number는 3.19다. Ill-conditioned NNLS 실패가 아니다. 자유부호·λ0에서도 \(g^\top H^{-1}g/\|e\|^2\approx0.00446\%\)에 불과하다. 같은 state에서 λ .001–1의 13점이 physical field를 최대 \(2.548\times10^{-7}\)만 바꾼 기존 CPU 결과 때문에, **B10 원인 규명 대신 촘촘한 λ sweep을 돌리는 것은 우선순위가 낮다.** 다른 λ chain이 B10 entry 자체를 바꿀 가능성은 별개다. [완료 분석 §8–9](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)

### 5.2 Exact state를 당연히 확보할 수 있다고 가정하지 않는다

공개 checkpoint inventory는 W1/M1, W5/M5, W10/M10을 기록한다. Fixed-z는 hash만 있고 journal replay parity는 `NOT_TESTED`다.

1. 추가 **봉인된** W9/M9/z 또는 검증 가능한 journal이 실제 존재하는지 확인한다.
2. 없으면 W5/M5에서 B6–B9를 bounded replay하는 비용을 별도로 산정한다.
3. 중간 W/M/z와 B10 z의 원본 hash가 맞아야 `EXACT_STATE_REPLAY`다.
4. 다르면 `NEW_REPLAY_ANALOGUE`; 과거 B10의 원인 확정으로 보고하지 않는다.
5. W10 update가 작다는 이유로 W10을 W9로 대체하지 않는다.

기존 Llama JV B6–B9의 target+write 합은 약 5,678초(1.58 GPUh), full-batch 합은 약 6,299초(1.75 GPUh)였다. Model load, B10 target, 추가 JVP는 별도다. 이는 과거 장비의 측정치이지 신규 자원의 ETA나 승인 budget이 아니다.

### 5.3 L1: same-state raw response를 한 번 측정한다

공통 W/M/z/raw D/qref를 고정한다. Native writer의 RHS는 request-normalize하지 않는다. 동일 terminal capture 3회와 다섯 raw direction의 requestwise JVP를 얻고 다음 작은 FP64 항을 저장한다.

\[
g_i=\Psi_i^\top e_i,\quad H_i=\Psi_i^\top\Psi_i,\quad
\pi_i=\frac{\operatorname{tr}H_i}{\operatorname{tr}H},\quad
N_{\rm eff}=\frac{(\sum_i\operatorname{tr}H_i)^2}{\sum_i(\operatorname{tr}H_i)^2}.
\]

필수 관측은 case4228의 실제 H/g 기여, top1/top5 share, spectrum, response projection, raw/normalized field, 물리적 layer action이다. 전체 H/g만으로는 이 attribution을 재구성할 수 없다.

동일 raw 값에 N0와 다음 NRMS만 CPU에서 적용한다.

\[
s_{\rm RMS}^2=\frac1{B_a}\sum_{i\in A}s_i^2,\qquad
e_i^{\rm RMS}=\frac{R_i}{\sqrt{B_a}s_{\rm RMS}}.
\]

Active inventory와 exact-zero 처리는 source N0와 같게 유지한다. 두 방식 모두 entry \(V_0=1/2\)이므로 potential의 전체 scale confound를 줄인다. 그렇더라도 request weighting을 바꾸는 **science ablation**이며 단순 버그 수정으로 숨기지 않는다. Request4228을 제거하거나 임의 clip하지 않는다.

반복 capture 차이가 0이어도 tiny residual이 유의미한 signal임을 증명하지 못한다. Compute-z와 controller의 batching, context aggregation, token position, FP32 연산 순서 차이가 결정적으로 반복될 수 있다. Target log와 source readout을 함께 확인한다.

### 5.4 L2: 실제 intervention과 판정

N0/NRMS 각각 \(\lambda=.1,T=2,N=4,h=.5\)의 같은-entry trajectory를 한 번씩 실행하는 것이 첫 actual 비교다. 같은 raw writer, target, P/M, entry qref를 사용한다. 다음 연결을 확인해야 한다.

\[
\text{tiny-row response dominance}
\rightarrow\text{coefficient suppression}
\rightarrow\text{near-zero physical write}
\rightarrow\text{new-edit failure}.
\]

Weighting 변경 후 coefficient만 커진 것은 rescue가 아니다. 실제 activation change, materialized update, 전체100개 request의 NLL/margin/strict coverage를 확인한다. Old-edit/NS는 evaluation-only다.

- NRMS에서 dominance와 stall이 함께 해소되고 새로운 sample에서도 안정적이면 normalization 문제를 별도 수정 후보로 승격한다.
- Field는 커지나 semantic 성능이 나쁘면 target/direction 문제가 남는다.
- Raw JVP부터 numeric envelope 이하이면 N0 통계만으로 결론 내리지 않는다.
- Exact state를 못 얻으면 원래 anomaly의 원인은 `UNRESOLVED`로 남긴다.

NNUM floor, compensated overlay 등은 이 단계의 근거가 요구할 때만 별도 variant로 검토한다. 자동 gain/clip/target 재학습으로 실패를 덮지 않는다.

## 6. Euler sweep: 선행연구의 실제 결과와 우리의 설계

### 6.1 ODESteer와 ODE-M을 같은 근거로 뭉뚱그리지 않는다

ODESteer v2는 기본 10 Euler steps와 \(h=T/10\)을 사용한다. Table5의 (T) 범위는 Falcon7B 20–23, Mistral7B 3–4, Llama3.1-8B 4–6, Qwen2.5-7B 13–16(일부 detoxification 설정 48)로 크게 다르다. 반면 Appendix E.4의 고정-T step-count 1–20 실험은 초반 개선 후 포화로 설명한다. Figure4는 Falcon/Mistral/Llama 패널이며 Qwen의 별도 N curve를 제공하는 근거로 읽지 않는다. [ODESteer v2, Table5·Appendix E.4](https://arxiv.org/pdf/2602.17560v2)

ODE-M v3의 step-size Table4는 **CLIP ViT-B/32 한 모델**의 continual merging 결과다. (h=.001/.01/.05/.2\)에서 평균 정확도는 각각 80.2/79.8/79.4/78.7이며 기본값은 .05다. 이를 Llama/Qwen별 Euler 민감도 증거라고 인용하면 안 된다. 이 방법은 calibration loss를 사용하는 model merging이므로 native-only editing과 관측 입력도 다르다. [ODE-M v3, §4.3·Table4](https://arxiv.org/pdf/2605.19409v3)

따라서 사용자의 지적처럼 **모델별 시간 설정 검증은 반드시 필요**하다. 다만 (T\), (N\), (h\)를 구분해야 한다. 위 논문의 수치 범위를 우리 weight-space ODE에 그대로 이식할 근거는 없다.

### 6.2 우선 실행할 5 operating points

정상 common-entry B100 fixture를 모델별 하나씩 고정하고, 기존 N0/λ.1을 기준으로 아래를 비교한다. Llama B10 원인 실험과 정상-state numerical sensitivity는 별도다. N0 수정이 채택되면 최종 후보에서도 이 핵심 비교를 다시 audit해야 한다.

| 목적 | T | N | h |
|---|---:|---:|---:|
| 기준 | 2 | 4 | .5 |
| 고정-T coarse | 2 | 2 | 1 |
| 고정-T fine | 2 | 8 | .25 |
| 고정-h short exposure | 1 | 2 | .5 |
| 고정-h long exposure | 4 | 8 | .5 |

모델별 실제 paths는 3개면 된다: h1로 2 nodes, h.5로 8 nodes, h.25로 8 nodes. h.5 trajectory의 T1/T2 prefix를 읽어 총 **18 nodes/90 main target JVP**로 5 operating points를 얻는다. 단순 별도 실행은 24 nodes/120 JVP다. Entry compute-z/qref도 fixture 안에서 공유한다. Diagnostic FD, history response, endpoint evaluator는 이 수에 포함하지 않는다.

다른 resolution의 결과를 한 경로에서 얻을 수는 없다. Prefix 재사용은 동일 h/λ/normalization, autonomous field, first-hit OFF에만 적용한다. Prefix evaluation 중 history append를 한 뒤 continuation하지 않는다. Outer sequential에서는 T가 달라진 첫 batch 이후 W/M/z가 달라지므로 다음 batch를 공통 trajectory로 재사용할 수 없다.

### 6.3 무엇을 측정해야 sweep의 의미가 생기는가

- Fixed T: endpoint \(\Delta W\), \(\Phi\), residual vector의 N2↔N4 및 N4↔N8 거리, 실제 \(\Delta b\), model error, NFE/wall time.
- Fixed h: exposure 증가에 따른 actual progress, new-answer NLL/margin, action, held-out endpoint 변화.
- FP32: intended hF와 materialized \(\Delta W\), changed-element fraction, actual activation movement, capture envelope.
- Layer: physical field angle, 각 layer absolute action와 share. h가 작아질 때 stepwise angle만 작아지는 것을 “feedback 사라짐”으로 해석하지 않는다.

**N을 늘려도 finite-horizon edit 성능이 반드시 좋아지지 않는다.** 예를 들어 \(\dot R=-R\), T2에서 Euler residual ratio는 N4일 때 .0625, N8일 때 약 .1001이고 연속해는 약 .1353이다. Coarse step의 더 강한 contraction과 수치적 정확성을 분리해야 한다. 이는 수학적 예시이며 실제 모델 결과가 아니다.

B10처럼 actual/predicted step error가 8–38배인 상태에서 h를 줄이면 write가 FP32 rounding에 더 묻힐 수도 있다. FP64 NNLS가 이미 손실된 FP32 readout 정보를 복구하지는 않는다. N16/RK4는 첫 단계가 아니라 N8에서 유의미한 refinement가 보이고 비용이 허용될 때의 후순위다.

### 6.4 최소 runtime 변경

현재 runtime은 import-time T/N/H/λ 상수와 telemetry의 별도 λ binding을 사용한다. Lock JSON만 바꾸거나 monkeypatch하는 방식은 잘못된 sweep을 만들 수 있다.

새 worktree에서 immutable config를 trajectory/telemetry/manifest가 공유하게 한다. (h=T/N\), 실제 시간합, coefficient/energy에 h가 각각 한 번 적용됨, 모든 shadow의 λ 동일성을 검사한다. 기존 .1/T2/N4를 먼저 재현하고 native solve/JVP/NNLS는 재작성하지 않는다. [현재 runtime source](https://github.com/hyunjun1127/ODE-edit/blob/77358b1546d1baf83b3e251afcce663b08d7bfd7/project/run_scripts/alpha_native_response_ode_v31_sequential/trajectory.py)

## 7. Historical key 외에 어떤 정보를 더 줄 것인가

### 7.1 현재 M은 무엇을 알고 무엇을 모르는가

현재 AlphaEdit에는 history가 빠져 있지 않다. Frozen entry (M\)이 native direction과 native cost 양쪽에 들어간다. 실수 대수에서

\[
M_\ell=\sum_{r<k}K_\ell^{(r)}K_\ell^{(r)\top},\qquad
\operatorname{tr}(F_\ell M_\ell F_\ell^\top)=\sum_{r<k}\|F_\ell K_\ell^{(r)}\|_F^2.
\]

이는 과거 key에 대한 **해당 linear module의 변화 비용**이다. 지금 과거 정답이 맞는지, 얼마나 margin을 잃었는지, 그 변화가 final output으로 어떻게 전달되는지는 모른다. Qwen B10의 selected raw native work 중 history 항은 약69.79%였지만 cost-only M0 shadow와 field angle은 .27444°로 작았다. 비용이 포함된다는 사실과 실제 layer 재배분은 별개다.

또한 qref와 storage는 outer batch마다 새로 시작한다. M trace 증가가 normalized penalty의 절대 강도 증가를 보장하지 않는다. 전체 layer의 history key가 append되므로 M은 사용한 layer의 누적 작업량 ledger도 아니다. [완료 분석 §3·5](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)

### 7.2 최소 추가 기록: 기존 edit의 training margin과 commit anchor

새 external reference set 대신 **이미 commit한 edit의 원래 training context와 target**만 사용한다.

\[
m_r(W)=\overline{\operatorname{NLL}}_W(y_r^{true}\mid x_r)
-\overline{\operatorname{NLL}}_W(y_r^{new}\mid x_r),
\qquad m_r^{floor}=m_r^{commit}-\epsilon_H.
\]

Original edit record에 두 target이 모두 있는 경우의 정의다. 대안 target이 없으면 new-target NLL variant를 별도 정의해야 하며 임의 target을 생성하지 않는다.

최소 기록은 edit ID/fact version, 원래 training context/tokenization hash, 두 target, commit model/version, commit margin이다. Context마다 여러 prefix를 새로 만들지 않는다. 첫 pilot은 원래 training context 한 개를 사용한다.

Replay 제안은 **R=8**, hash-based deterministic selection, inner batch 동안 membership 고정이다. Uniform hash를 기본으로 하고 age-balanced sampling은 후속 별도 sensitivity로 둔다. Causal score나 결과 기반 layer weight는 도입하지 않는다. Replay size16은 비용 측정 이후의 사전 정의된 sensitivity이며 첫 full factorial에 포함하지 않는다.

- Commit reference를 다음 batch entry로 reset하지 않는다. 그러면 누적 손실을 새 정상으로 받아들이게 된다.
- Selection은 PS/NS, 현재 margin, 이후 성공 여부와 독립적이다.
- 알려진 overwrite는 명시한 fact/version policy로 inactive 처리한다. 식별되지 않은 conflict도 기록한다.
- 처음부터 실패한 commit을 사후 제외하지 않는다. Main denominator와 initially-successful retention strata를 분리한다.
- 첫 batch는 history가 없으므로 같은 설정의 native JV와 일치해야 한다.
- 과거 commit margin이 저장되지 않았다면 현재 margin을 대신 쓰고 commit anchor라고 부르지 않는다.

이 방법은 외부 bank를 추가하지 않지만 **native-only unchanged 방법은 아니다.** 기존에는 없던 historical training input과 최종-output 계산이 추가되는 별도 variant다. 이 실험을 선택하면 과거 RS와 일부 겹치는 training signal을 controller가 사용한다는 사실도 공개해야 한다. Old paraphrases, nonreplayed historical requests, neighborhood는 별도 evaluation으로 남긴다.

### 7.3 같은 5차원 controller로 비교할 세 가지

현재 native dictionary \(T_\ell\), current response \(\Psi\), target error (e\)는 유지한다. 과거 margin의 **현재** directional derivative를

\[
J_{r\ell}=Dm_r(W)[\mathcal I_\ell T_\ell]
\]

로 측정한다. (J\)는 R×5다. JVP 동안 T는 stop-gradient이며 다음 node에서 현재 W의 dictionary와 response를 다시 만든다.

#### H-BASE: 기존 key-native JV

\[
\min_{c\ge0}\frac12\|e-\Psi c\|^2+\frac\lambda2c^\top Gc.
\]

이 arm에도 observer-only J/margin을 측정하면 측정 자체의 개입과 계산량을 분리할 수 있다.

#### H-METRIC: 과거 output의 현재 sensitivity만 추가

\[
\min_{c\ge0}\frac12\|e-\Psi c\|^2
+\frac\eta{2R}\|Jc\|^2+\frac\lambda2c^\top Gc.
\]

Key-only block-diagonal cost에 없던 **functional cross-layer term (J^\top J\)**가 들어간다. 같은 current progress를 만들면서 과거 margin을 덜 움직이는 조합을 선호한다. 개선과 악화를 대칭적으로 벌주며 이미 잃은 margin deficit을 알지는 못한다.

KKT에서

\[
g^\top c=\|\Psi c\|^2+\eta\|Jc\|^2/R+\lambda Q_N(F),
\quad \dot V=-g^\top c.
\]

따라서 원래 storage의 연속시간 증가율은 \(\dot b=\|\Psi c\|^2+\eta\|Jc\|^2/R\ge0\)다. 개별 historical margin 보장은 아니다.

#### H-ANCHOR: commit 대비 deficit까지 추가

\[
d_r(W)=[m_r^{floor}-m_r(W)]_+,
\]

\[
\boxed{\min_{c\ge0}\frac12\|e-\Psi c\|^2
+\frac\eta{2R}\|d-Jc\|^2+\frac\lambda2c^\top Gc.}
\]

이는 stacked error \([e;\sqrt{\eta/R}d]\), stacked response \([\Psi;\sqrt{\eta/R}J]\)를 기존 NNLS에 넣는 형태다. Hard history constraints나 slack variables가 늘지 않으며 여전히 coefficient 5개다. (c=0\)은 항상 가능하고 \(\lambda G\succ0\)이면 유일한 minimizer가 있다. 하지만 target와 history를 동시에 원하는 만큼 개선할 수 있다는 보장은 아니다.

이때 potential은 반드시 바뀐다.

\[
U(W)=V(W)+\frac\eta{2R}\|d(W)\|^2,
\quad b_H=U(W_{entry})-U(W)-\lambda\mathscr E,
\quad \dot{\mathscr E}=Q_N(F).
\]

고정 replay/reference/normalization/native metric과 exact current derivative 아래,

\[
\dot U=-\|\Psi c\|^2-\eta\|Jc\|^2/R-\lambda Q_N(F),
\quad \dot b_H=\|\Psi c\|^2+\eta\|Jc\|^2/R,
\quad Q_N(F)\le\frac{U}{2\lambda}.
\]

Squared positive-part의 gradient는 boundary에서도 정의된다. d=0인 row도 J metric에는 남지만 derivative의 deficit 항에는 0으로 기여하므로 위 identity와 모순되지 않는다.

여기서 stacked response는 **design operator**다. Inactive row까지 포함했으므로 clipped residual 전체의 정확한 Jacobian이라고 부르면 안 된다. 정확한 관계는 \([e;\sqrt{\eta/R}d]^\top[\Psi;\sqrt{\eta/R}J]c=-\dot U\)라는 gradient pairing이다.

**H-ANCHOR에서는 U가 줄어도 current V는 증가할 수 있다.** 기존 current-only CLF/speed bound를 그대로 보고하면 안 된다. Replay membership이 outer batch에서 바뀌면 하나의 global lifelong certificate도 아니다. 유한 Euler에서는 실제 U/V/old margin/dissipation defect를 별도 측정한다.

Current target 비감소를 반드시 요구한다면 \(g^\top c\ge0\)라는 하나의 homogeneous constraint를 추가할 수 있다. c0은 여전히 feasible하다. 그러나 plain NNLS에서 constrained QP로 바뀌므로 초기 main에는 넣지 않고 별도 연구 선택으로 남긴다. Per-request barrier를 다수 추가하는 방향으로 돌아가지 않는다.

### 7.4 초기 설정과 scope

첫 observer는 \(\eta=0,.01,.1,1\), R8, \(\epsilon_H=0\)을 제안한다. 같은 J에서 CPU solve만 달리한다. 첫 actual H-METRIC/H-ANCHOR 후보는 \(\eta=.1\), R8, \(\epsilon_H=0\), T2/N4를 개발 설정으로 제안한다. 최적값·안전 보장으로 기록하지 않는다. 다른 값이 필요하면 development evidence와 변경 이력을 남기고 독립 audit 전에 lock한다.

Native λ와 historical η는 서로 다르다. Historical margin은 nat 단위이고 current e는 정규화되어 있으므로 η도 과학적 scale 선택이다. ε0은 commit-time training margin 손실을 그대로 deficit으로 보는 **soft** 기준이지 무손실 보장이 아니다.

과거 정보를 넣어도 actuator는 새 edit residual에서 만든 기존 다섯 native directions다. 과거 repair 전용 direction, output projector, z 재학습은 추가하지 않는다. 이 cone에 유용한 alternative가 없으면 감속·stall·new-edit sacrifice가 발생할 수 있다. 그것도 중요한 결과다.

### 7.5 “추가 정보가 layer 기여도를 바꿨는가”의 판정

Coefficient 숫자나 L8 share만 보지 않는다.

| 관측 | 해석 |
|---|---|
| Physical field가 회전하고 matched current progress에서 old-margin 악화가 감소 | 추가 functional geometry의 선택 가치 |
| L8 share만 줄고 전체 field/current progress도 크게 감소 | 감속 또는 under-edit; routing 개선으로 단정 불가 |
| H-METRIC 변화는 작고 H-ANCHOR가 deficit를 줄임 | 현재 sensitivity 외에 accumulated deficit 정보가 중요할 가능성 |
| Replay margin만 좋아지고 nonreplayed/old paraphrase는 그대로 | Replay coverage/overfitting 한계 |
| Current edit를 희생해 history만 개선 | Soft trade-off 확인, 전체 성능 개선은 아님 |
| Full5가 L8-only history shadow보다 좋음 | Alternative layer support의 실용 가치 후보 |
| 모든 η에서 유용한 field가 없음 | 정보 부족보다 dictionary/reachability 병목일 가능성 |

필수 telemetry는 physical native/Frobenius angle, layer별 absolute action, (g^\top c\), signed (Jc\), deficit, actual margin change, L8-only shadow다. Native quadratic action share와 batch-net Frobenius share는 구분하고, joint predicted target progress의 signed layer contribution은 cross-term 때문에 단순 양수 percentage로 강제하지 않는다.

L8-only로 후보를 미리 제한하면 historical penalty도 주로 scalar amplitude를 바꾼다. **이번 information-value 실험은 full5 dictionary를 유지해야 “다른 layer로 갈 수 있었는가”를 물을 수 있다.** 분산을 강제하는 anti-L8 penalty는 넣지 않는다.

### 7.6 비용을 먼저 재는 실제 pilot

Commit reference가 없는 기존 state에서 소급 anchor를 만들지 않는다. 새 isolated fixture에서 Official AlphaEdit로 결과 독립적인 B100 warm batch를 한 번 편집하고 commit margins를 저장한다. 같은 warm W/M에서 새 B100 target을 한 번 계산해 H-BASE/H-METRIC/H-ANCHOR가 공유한다. 이는 controlled short-history mechanism이며 원래 JV B10의 재현도, lifelong 성능 검증도 아니다.

먼저 Llama의 entry와 한 nonzero-step state에서 R8의 J만 측정한다. 단위시험과 짧은 actual probe로 예측·실현 일치를 확인하고 CPU η shadows를 계산한다. 의미 있는 기능적 선택 차이가 있을 때 세 actual paths로 확장한다. 그 다음 각 arm이 자신의 W/M/commit references를 누적하는 짧은 sequential audit를 수행한다.

J 계산은 history margins의 full-model teacher-forced path를 지나므로 L8 prefix JVP보다 비쌀 수 있다. 다섯 batched JVP 또는 scalar-margin reverse derivative의 실제 비용을 비교하되 처음부터 둘 다 full run하지 않는다. Full vocabulary tensor를 영구 저장할 필요는 없다. R8, 5 directions, node 수, NLL 두 target forward/JVP 비용을 별도 ledger로 기록한다.

J를 stale하게 재사용하면서 exact-current certificate를 주장하지 않는다. 초기 correctness에서는 current J를 재측정하고, 이후 reuse/analytic shortcut은 별도 parity를 통과한 optimization으로 둔다.

## 8. Llama 개선과 method closure

### 8.1 정상 상태의 운영점 탐색은 필요하지만 좁게 한다

B10에서 λ shadow가 무효라고 정상 상태 λ sensitivity까지 부정할 수는 없다. 정상 Llama B9에서는 λ .001/.1/1의 field가 유의미하게 달랐다. 따라서 정상-state 개발에서는 λ .01/.1/1의 coarse screen이 합리적이다.

그러나 순서는 다음과 같다.

1. N0 진단과 fixed-T/fixed-h 결과를 먼저 확보한다.
2. 정상 개발 state의 same-state λ screen에서 amplitude/angle/current progress를 본다.
3. λ .1 기준과 유망한 소수 후보만 actual trajectory로 비교한다.
4. Matched-progress 및 고정 계산 예산의 결과를 함께 확인한다.
5. Audit 전에 모델별 운영점과 normalization을 고정한다.

Llama/Qwen에 동일 최적 λ/T가 존재해야 한다고 가정하지 않는다. 반대로 final 점수를 보고 모델별 무제한 설정을 허용하지도 않는다. 모델별 설정을 쓸 경우 동일한 development 절차·후보 수·계산 budget을 공개한다.

### 8.2 데이터·선택 정책

- 기존 1,000 stream과 B10 사례는 retrospective diagnostic이다. 새로운 confirmatory sample로 취급하지 않는다.
- 두 live task의 reserved inventory와 기존 개발 inventory를 제외한 dataset 내 requests를 canonical ID hash 순서로 선정한다. 구체 sample hash와 seed namespace를 실행 전 봉인한다.
- 권장 1차 development는 두 모델 공통 ID의 B100 fixture, 독립 audit는 다른 B100 3-batch stream이다. History pilot warm/current inventory는 dev/audit와 역할을 분리해 기록한다.
- Compute-z는 동일 common entry 안에서만 공유한다. Sequential 분기 이후에는 각 arm의 현재 W에서 baseline 방식으로 새로 계산한다.
- Development에서는 별도로 지정한 validation 결과와 계산 비용으로 운영점을 선택할 수 있지만, 선택 규칙과 trade-off를 먼저 기록한다. Final/audit PS·NS는 controller, per-case stopping, rescue에 사용하지 않는다.
- Failure replacement, 성능이 좋은 prefix 선택, request별 T 증가는 금지한다.

모든 후보의 공식 비교는 전체 분모를 유지한다. Initially-successful subset은 forgetting 분석용 보조 strata이며 main table 대체가 아니다. Replay training margin은 controller가 직접 사용한 지표임을 표시하고 independent old-edit/paraphrase/locality 결과와 나눈다.

### 8.3 Method를 닫는 최소 조건

1. **구현:** 실제 W/M/z/state 전달, controller λ와 time ledger, materialization/JVP가 계약을 만족한다.
2. **Llama acquisition:** B10형 near-stall을 설명·진단할 수 있고, 수정 후보가 새로운 independent stream에서 과도한 실패를 만들지 않는다. 원인 미확정은 미확정으로 남긴다.
3. **Llama quality:** B10 이외의 PS/NS 열세가 개선되어야 한다. Rewrite RS만 회복한 후보를 완성본으로 부르지 않는다.
4. **Qwen regression:** 기존 locality 이득과 editability가 새 independent audit에서도 합리적으로 유지되는지 확인한다.
5. **Attribution:** L8 support, iterative feedback, native penalty 또는 추가 functional history 중 무엇이 실제 이득을 만드는지 최소 대조군으로 구분한다.
6. **Cost:** 추가 history/JVP/target 비용을 포함한 wall time과 main NFE를 공개한다.
7. **Selection:** Practical noninferiority margin, superiority target, budget을 audit 공개 전에 seal한다. 관측값을 보고 margin을 정하지 않는다.

이 조건은 두 모델 모든 metric의 무조건 우월성을 요구하는 것은 아니다. 충분한 editability 하에서 locality/retention을 개선하는 trade-off인지, 특정 모델에만 유효한지 명확히 결정하는 기준이다. 공통 규칙으로 Llama 개선이 안 되면 Qwen-specific 관측과 method 한계를 인정해야 한다.

History term이 선택 가치를 주지 않으면 main에서 제외한다. L8-only로 충분하면 full5를 유지하기 위한 인위적 분산 규칙을 만들지 않는다. 고정-z anchor 자체가 부족한 증거가 남으면 이번 scope의 제한으로 보고하고 MetaKE형 target 재학습을 조용히 섞지 않는다.

## 9. Method lock 이후의 최소 ablation

모든 조합을 동시에 돌리지 않는다. Server4 결과로 필요 없는 arm을 먼저 줄인다.

| 비교 질문 | 최소 대조군 | 해석 주의 |
|---|---|---|
| 여러 layer가 필요한가? | 봉인될 L8_ONLY_NATIVE 대 현재 full JV | 현재 running task 재사용; 새 최종 method가 바뀌면 해당 method에서 support audit 필요 |
| 반복 current response가 필요한가? | 선택 support의 one-shot/frozen-response 대 refreshed iterative flow | 같은 target/entry; stale response에는 current certificate 적용 금지 |
| Native penalty가 유용한가? | 동일 dictionary의 λ>0 대 λ0 또는 공통 ray/progress 비교 | λ0에는 λ>0 speed bound 없음; singularity를 자동 ridge로 숨기지 않음 |
| Direction selection인가, 속도 조절인가? | JV 대 ORB_RAY_N/동일 support ray | Fixed-T와 matched-progress를 함께 보고, 미도달은 그대로 표시 |
| 과거 정보가 추가 가치를 주는가? | H-BASE / H-METRIC / H-ANCHOR | 첫째는 key, 둘째는 functional sensitivity, 셋째는 deficit 정보 차이 |
| History replay의 일반화인가? | replayed/nonreplayed historical RS, old PS, NS | Replay training margin만 좋아진 결과를 lifelong improvement로 부르지 않음 |
| Numerical resolution에 강건한가? | 같은 T의 N4/N8 | N을 늘려 exposure를 같이 늘리는 비교는 별도 |

λ0 arm은 finite-horizon 대조군이지 안전하다고 가정한 deployed method가 아니다. Rerun의 상태나 설정을 바꾸는 방식으로 ablation하지 않는다.

L8-only가 충분하다면 analytic L8 response, grouped key capture, native factorization reuse를 이후 효율화 후보로 볼 수 있다. 현재 L8 takeover는 **모든5 directions를 만든 뒤 L8로 filter**하므로 JVP20→4/batch여도 native direction build는 줄지 않는다. “비용이 1/5”라고 미리 주장하지 않는다. Analytic readout 대체는 실제 architecture/token/context에서 parity를 확인한 후에만 적용한다.

## 10. Lifelong 검증과 claim 최종 분기

### 10.1 평가 clock

\[
R_{t,j}=\operatorname{Metric}(W_t,\mathcal B_j),\quad j\le t
\]

에서 current diagonal, old off-diagonal, final seen-prefix를 나눈다. Fixed cohort의 loss/recovery와 처음부터 실패한 edit를 구분한다. Overwrite된 사실은 유효한 과거 사실의 보존 문제와 분리하되 전체 요청/충돌 수를 공개한다.

Server4 ORBODE는 매 checkpoint의 all-seen matrix를 제공할 계약이고, Alpha JV/L8는 all-seen PS/NS가 W1/W5/W10에 한정된다. 관측하지 않은 중간 failure time을 보간해 survival curve를 만들지 않는다. 평가 간격에 따른 interval censoring과 recovery 가능성을 명시한다.

Paired prompt 분석은 request 단위 clustering을 기본으로 하고 relation/stream 의존성을 검토한다. 10,000 neighborhood prompts를 10,000개의 독립 editing runs처럼 취급하지 않는다. 한 stream의 bootstrap만으로 order-generalization을 증명할 수 없으므로 최종 후보에는 독립 order 확인이 필요하다.

### 10.2 추가 history가 pretraining locality까지 보장하는가

아니다. Historical training margin은 committed edit 보존에 관한 정보다. Unrelated/pretrained behavior는 기존 native P/M geometry의 간접 영향과 evaluation에서만 확인된다. NS가 개선돼도 sampled neighborhood 범위의 결과이며 전반적 capability preservation으로 확대하지 않는다.

따라서 main claim은 관측에 따라 다음처럼 달라진다.

- Native-only, L8 집중으로 충분: “현재 response에 따라 반복적으로 target을 실현하는 집중형 writer가 특정 sequential setting의 locality–efficacy trade-off를 개선한다.”
- Full joint support가 추가로 유용: “현재 transported response에 근거한 layer mixing이 단순 scalar rescaling보다 유용하다.”
- Functional history가 독립 retention을 개선: “Committed training behavior의 current sensitivity/deficit를 이용하는 response controller가 key-only history보다 retention을 개선한다.”
- Llama 개선 없이 Qwen만 유지: “Architecture-dependent benefit”; universal editor claim은 축소한다.
- 1k는 통과하지만 5k/10k에서 붕괴: 해당 horizon의 성과만 남기고 lifelong-safe 주장은 하지 않는다.

어느 경우에도 “realization debt가 forgetting의 원인” 또는 “barrier가 모든 output을 보존”한다는 문장은 자동으로 따라오지 않는다.

### 10.3 장기 run은 언제 시작하는가

Llama 후보의 독립 audit와 method lock 이후 Official/선택 후보 중심으로 1k qualification을 진행하고, 유효한 W/M/commit-reference resume identity와 새 stream을 봉인한 뒤 5k/10k로 확장한다. 기존 Qwen을 곧바로 대규모 tuning 대상으로 만들지 않는다.

ORBODE의 weight checkpoint는 evaluation-only이며 editing-resume cache가 보장되지 않는다. 이를 AlphaEdit history가 필요한 continuation의 시작점으로 임의 사용하면 안 된다. 현재 Alpha checkpoint도 source/version/reference 호환성을 확인한 뒤 재개한다.

## 11. 비용·산출물·실행 전 계약

### 11.1 비용을 아끼는 우선순위

완료 JV B100의 median 측정치는 다음과 같다. Endpoint 시간이 write 계측에 포함되므로 별도로 제시한다.

| 모델 | compute-z | write+endpoint | endpoint evaluator |
|---|---:|---:|---:|
| Llama | 794.89s | 619.52s | 80.20s |
| Qwen | 363.85s | 666.13s | 91.20s |

따라서 가장 먼저 줄일 중복은 **같은-entry의 compute-z 반복**과 이미 측정한 raw JVP에서 normalization/λ/η shadows를 다시 GPU로 계산하는 일이다. Prefix 재사용은 조건을 검증한 경우만 허용한다. History full-output JVP가 새 병목일 수 있으므로 작은 R8 observer의 wall time부터 측정한다.

자원 상한은 아직 지정되지 않았다. 이 보고서는 임의로 GPU-hour 승인을 만들지 않는다. 실행 전에는 작업별 예상비용/우선순위/최대 jobs/hours를 별도 resource lock으로 확정하고, 초과 시 후순위 arm을 **모델·case별 성능과 무관하게** 줄인다.

### 11.2 최소 산출물

각 새 package는 source/runtime/science/sample/resource manifest와 run registry를 별도로 갖는다. 이 보고서의 design manifest는 runtime science lock을 대신하지 않는다.

- L0: `artifact_availability.json`, `replay_identity.csv`.
- L1/L2: `request_response_contributions.csv`, `normalization_same_state.csv`, `physical_write_parity.csv`, `diagnostic_endpoints.csv`.
- E1: `time_resolution_grid.csv`, `endpoint_distances.csv`, `dissipation_nodes.csv`, `compute_accounting.csv`.
- H0/H1: `history_reference_manifest.json`, `history_same_state_fields.csv`, `history_margin_transitions.csv`, `replay_vs_nonreplay_metrics.csv`.
- C0/A0: `claim_evidence_matrix.csv`, paired fixed-cohort tables, `method_selection_record.md`, `audit.lock.json`.

Trajectory identity에는 model/method/source, W/M/z/context hashes, normalization/constants, support, λ/η/T/N/h, fixed qref, history replay/reference/version hashes, state_version을 포함한다. 실제 low-rank increments와 재현 가능한 checkpoint/reference는 local artifact로 보관한다. Raw prompt/tensor/cache는 Git에 올리지 않는다.

### 11.3 실패 정책

Source/metric/JVP 불일치는 technical hold다. Exact state 부재는 unavailable이며 근사 재현과 구분한다. Finite zero-field, near-stall, horizon miss, finite Euler defect, 성능 저하는 scientific observation으로 남긴다. No-action endpoint를 Official로 대체하거나 실패 row를 제외하지 않는다.

이번 단계는 fixed horizon, first-hit OFF, backtracking/rollback OFF를 유지한다. (\alpha=h u\)의 controller/integrator 구분과 (T=Nh\)를 보존한다. q excursion은 우선 telemetry이며, 별도 numerical guard를 추가하면 새 source/science contract로 명시한다. Semantic first-hit이나 dynamic-z는 현재 Llama 원인 진단과 섞지 않는다.

## 12. 최종 연구 방향

현재 가장 방어력 있는 출발점은 “다층에 반드시 분산해야 한다”가 아니다.

> Native writer가 만드는 실제 response와 보존 비용을 관측했을 때, 어디에 얼마나 써야 유효한 edit를 더 적은 기능 손실로 실현할 수 있는가?

완료 Qwen 결과는 이 질문의 유효성을 보여준다. L8 집중과 더 큰 전체 weight action에도 Official보다 locality가 좋아졌기 때문이다. 하지만 현재 Llama 결과는 **response weighting의 수치적 신뢰성**, **정상 상태의 목표–보존 trade-off**, **key-only history의 표현 한계**를 분리해서 해결해야 함을 보여준다.

권장하는 최종 순서는 **Llama near-stall 진단 → 두 모델 N/T 분리 → 필요할 때 최소 functional-history 추가 → Llama 개선 audit → Server4 결과와 함께 method 단순화·lock → 최소 ablation → 긴 lifelong**이다. Layer 분산이나 barrier라는 이름을 유지하기 위해 복잡성을 추가하지 않고, 실제로 남는 이득에 맞추어 claim을 정한다.

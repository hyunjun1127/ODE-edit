# JLZ v9 Ridge 두 arm과 exact pilot 실험

2026-10-03. 목적은 **공동 local-z의 계획 배분과 native ridge의 실현 배분을 구분하고, 평균 key 미실현이 actual 성능 차이의 어느 부분을 설명하는지 확인**하는 것이다. 사용자가 선택한 범위는 ridge 본선과 작은 exact 비교다. Exact를 본선으로 자동 승격하거나 absolute tracking을 추가하지 않는다. 이 문서는 설계이며 GPU 실행·기존 job 취소·GH 전달 완료 보고가 아니다.

## 1 고정 방법과 범위

[Method](../method-ko.md)와 [구현 계약](../implementation-ko.md)을 정본으로 사용한다. 두 main arm은 **A=층별 combined ridge 비용의 합, B=그 제곱합의 root**만 다르다. 모두 native subject 주입·손실·current pulse, replay0, native mean-key ridge/H, q Adam, 모든 eligible layer를 유지한다.

이번 실행 profile은 기존과 같은 Meta-Llama-3-8B-Instruct/CounterFact fixed10k, L4–L8, FP32 모델/FP64 geometry다. 각 arm은 **BS100×5=500 edit**, 독립된 W0/H0에서 시작한다. 후보25/갱신24, q LR.1, warm-up0, clamp.75, λ_alloc.1을 고정한다. 6번째 batch를 준비·fit·commit하지 않는다.

이 B·모델·benchmark·층 번호는 실험 profile이다. Method 코드는 runtime 실제 B/차원/context/target/partial batch를 사용한다. 각 layer에 별도 scalar quota나 사용 여부 gate를 두지 않는다. Exact probe의 rank 조건은 main layer 선택 조건이 아니다.

## 2 입력과 기준선 결속

기존 schedule `plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/case-schedule.csv`의 stream_index0 `[0,500)`을 순서 그대로 사용한다. 원 schedule SHA256은 `dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2`, 500 case ID compact JSON hash는 `0be7d88c759e7f65a690514035f40a18c5c19d8591ddd33f97b2fa6187b64a95`다. 처음3개16186/3743/1481, 마지막3개3419/8549/5628이다.

Native context/model/dataset/C0 자산 identity는 [v7 실험 receipt 명세](../../2026-10-02-jlz-causal-writer-v7/experiment-500/experiment.json)에서 **입력 identity만** 재사용한다. 그 문서의 raw-δ Adam, full-context writer/history, replay, 과거 objective/namespace를 상속하지 않는다. Baseline writer reference는 기존 리뷰에서 확인한 BLUE checkout의 `memit.memit_seq_main`의 `blue=False` 경로다. 실행자는 source commit·native functions hash를 새 receipt에 결속한다.

각 arm의 source/config/input/evaluator/seed를 실행 전에 고정한다. Native input group 평균과 NLL context 평균을 혼동하지 않는다. 실제 adapter/tokenization identity가 다르면 결과를 동일 실행으로 합치지 않는다.

새 baseline fit은 추가하지 않는다. 기존 MEMIT-H, AlphaEdit, BLUE, CAKE 및 v4/v5/v7의 동일 500개 raw가 있으면 비교한다. Attachment의 v7 A100/99.2/71.96은 provenance 확인 전까지 `review_reported`다. W0 N87.84%=4392/5000 역시 동일 case/token/metric을 검증한 뒤 사용한다. BLUE L4+L8 및 runtime 차이를 표에 명시한다. 서로 다른 GPU의 소요시간을 방법의 속도 배율로 직접 비교하지 않는다.

## 3 실행 단계

| 단계 | 실행 | 고정 범위와 산출물 |
|---|---|---|
| Q0 | CPU 수학과 정합 | Ridge/exact 식, merged cost, rank 반례, g/e 비식별, causal gradient. 실제 production/GPU PASS와 구분한다. |
| Q1 | 작은 native pilot | Main 밖 case541/6693/16935/17306으로 각 arm 독립 BS2×2, batch당25후보/24갱신. B2에서는 history 갱신을 검사하고 past forward0을 확인한다. |
| Q2 | BS100 본선 A의 B1 | Main의 cold W0/H0와 동결 config로25후보를 fit한다. Terminal D를 고정하고 아래 두 exact probe를 수행한다. 이는 별도 fit을 반복하지 않는 **main A 첫 batch 자체**다. |
| Q3 | Ridge main 완주 | A는 같은 B1 ridge 결과를 commit한 뒤 B2–B5를 이어간다. B는 독립 W0/H0에서 B1–B5. 각각500개 후 종료한다. |

Q1의 D/W/H/Adam/RNG를 main에 전달하지 않는다. Q2는 처음부터 main의 B1으로 등록하며 main trace/config/source가 동일해야 한다. Pilot을 본 뒤 config를 고치면 Q2의 prefix 재사용 자격을 잃는다. 버그 수정 후에는 새 attempt/source/hash로 cold restart하며 이전 성능과 합치지 않는다.

Q1에서 고정 D의 B1/B3 view와 unequal context/target 길이도 forward/gradient parity로 검사한다. 이들은 새 장기 training arm이 아니다. Main request를 교체하거나 실제 B100을 여러 독립 writer batch로 쪼개지 않는다. Microbatch는 memory 구현 단위이며 whole-B solve/gradient barrier를 유지한다.

## 4 같은 D를 쓰는 두 exact 비교

Q2의 ridge terminal25 D 및 batch-entry A/H/W를 고정한다. **추가 fit은0회**이며 official evaluation 전에 target D hash를 저장한다. Terminal weights를 보고 선택하거나 D를 확대하지 않는다.

### 같은 key에서 writer 식만 비교

Ridge builder가 계산한 κ를 그대로 두고 모든 eligible layer에서 ridge/exact operator를 계산한다. M, DM, diag/off-diagonal, self/cross action, context별 action, preservation energy, weight norm, equality residual, condition을 저장한다. 이 비교는 writer 식 자체의 차이를 분리한다. Full-model inference가 없어도 가능한 algebra/operator 단계다.

같은 frozen upper κ로 exact weight들을 한꺼번에 설치한 모델을 “causal exact”라고 평가하지 않는다. Lower writes를 바꾸면 upper incoming key도 달라지기 때문이다.

### 같은 D로 actual 경로를 각각 재구성

Ridge는 main terminal builder 결과를 재사용한다. Exact는 같은 W_entry/A/H와 D에서 낮은 층부터 실제 weight를 설치하고, 각 상층의 κ/X를 새로 계산한다. 각 층에서 full-rank/수치 residual 조건을 다시 검사한다. 조건을 모두 통과하면 실제 current native NLL/KL과 **B1 R100/P200/N1000**을 평가한다. Ridge B1의 같은 endpoint 평가는 main 평가와 재사용한다.

Exact가 한 층이라도 미지원이면 전체 exact endpoint를 `unsupported`로 남긴다. 부분 요청/층만 성공한 결과로 분모를 줄이지 않는다. Full-rank compatible 여부, rank-deficient compatibility residual, numerical failure를 구분한다. Pseudoinverse/ridge/jitter fallback은 없다. Frozen-K와 causal-K의 qualification은 별개며, 상층에서는 frozen-K 실패만으로 causal 시도를 생략하지 않는다. Exact 미지원만으로 qualified ridge 본선을 막지는 않지만, 복구/identity 실패는 main도 중단하는 기술 실패다.

두 writer가 upper key를 다르게 만들기 때문에 이 단계는 **writer와 그 인과적 후속 변화의 총효과**다. Fixed-key 효과와 분리해서 해석한다. 같은 D가 ridge 학습으로 얻어졌다는 한계도 기록한다. Independently trained exact method 또는 writer 간 최적 성능 비교라고 부르지 않는다.

Probe가 종료되면 W/H/RNG와 main candidate identity를 복원·검증한다. 저장해 둔 immutable ridge terminal weight를 main에 commit하고 native H를 한 번만 누적한 뒤 B2로 진행한다. 같은 run/candidate/D/weight hash를 사용하며 Q2를 독립 pilot과 main의 두 fit으로 이중 계상하지 않는다. Exact에는 main history update, 별도 optimizer, B2 이후 순차 편집이 없다. Ridge/exact의 first-layer κ/A는 공유할 수 있지만 P/G/E 및 first_geometry cache는 writer별로 분리한다.

## 5 학습과 관측의 경계

Official P/N은 optimizer, teacher, 배분 비용, reference set, candidate/arm 선택, stopping, coefficient 조정에 사용하지 않는다. Main A/B와 budget은 pilot 전에 고정한다. Exact 결과가 좋더라도 자동으로 본선을 교체하지 않는다. 모든 current native loss와 physical pulse는 method에 정의된 요청에서만 계산한다.

낮은 성능, 한 층 집중, 이후 clamp 포화, 미수렴, 높은 exact energy는 결과다. 이를 기술 실패로 바꾸거나 강제 분산·추가 fit·새 계수를 적용하지 않는다. Input/shape/finite/gradient/commit/rollback 오류는 기술 실패로 처리한다. Exact 수치 qualification은 equality 비교가 성립하는지의 검사이며 layer allocation gate가 아니다.

## 6 질문별 측정과 해석

| 검증할 질문 | 측정 | 허용되는 결론 |
|---|---|---|
| Native ridge의 평균 key 미실현이 실제로 큰가 | 직접 M, D M−D, gamma/norm/orthogonal, diag/offdiag | 계획과 mean action 차이를 직접 정량화한다. g/e scalar 역산은 사용하지 않는다. |
| Context나 lower trajectory 차이가 더 큰가 | Terminal canonical/모든 native rewrite context의 세 벡터 gap 분해와 교차 내적 | Mean error 제거 후에도 남는 원인을 구분한다. Norm 합을 원인 비율로 쓰지 않는다. |
| Exact가 어느 부분만 바꾸는가 | Fixed-K equality/energy 및 causal endpoint NLL/KL/R/P/N | 평균 실현 증가가 endpoint 개선으로 이어지는지 확인한다. Locality 개선은 보장되지 않는다. |
| 의도한 층별 배분이 달라지는가 | 계획 D share와 mean Y share, signed directional/self/cross action, write energy | 크기·방향·요청혼합 차이를 구분한다. Layer norm을 causal contribution으로 단정하지 않는다. |
| q Adam이 초기 포화를 교정하는가 | First step bound, pre/post clamp, step별 rho/virtual loss/current actual probes | First-step 수치 문제와 후반 목적/trajectory를 구분한다. 24-step 수렴은 주장하지 않는다. |
| 순차 locality와 PS가 함께 나아지는가 | Main W5 누적 R/P/N, W0 matched losses/gains, at-write→W5 cohort retention | 동일 cohort에서 tradeoff를 평가한다. New objective·writer·optimizer의 개별 인과 효과를 분리했다고 주장하지 않는다. |

## 7 평가 endpoint

Main W1–W4는 current100 전체를 평가해 R100/P200/N1000을 저장한다. W5는 first500 전체를 한 번 평가해 **R500/P1000/N5000**을 저장한다. W5 current100은 같은 raw에서 산출한다. W1–W4를 누적 점수로 표시하지 않는다.

NLL preference rate를 기존 기준으로 사용하고 strict target-token success, true/new NLL와 margin, token accuracy를 함께 저장한다. W0→W5 neighborhood lost/gained, at-write→W5의 cohort별 edit/paraphrase 유지도 산출한다. 500개 occurrence와 active fact version의 분모를 구분한다. Aggregate에 평가 누락이 있으면 count와 이유를 표시한다.

Writer probe의 B1 결과와 main W5 누적 결과를 같은 열에 혼합하지 않는다. Exact는 100개 endpoint만 있고 500-edit 결과가 없음을 명시한다. 동결된 결과로 native baseline과의 PS/NS 차이를 설명하되 품질 통과선은 두지 않는다.

## 8 비용 상한과 효율

Q1은 두 arm×2 batch×25후보=100후보,96갱신이며 B2 소규모다. Main은 두 arm×5 batch×25=250후보,240갱신이다. **Q2 B100의25후보/24갱신은 main A B1에 이미 포함**되어 추가 B100 fit이 없다. Exact probe는 terminal frozen-key algebra 한 번과 full causal builder/actual endpoint 최대 한 번이다. Exact backward와 solver 재학습은0이다.

Pulses/terminal/head/공식평가/checkpoint 재계산/geometry/CPU history는 별도로 계측한다. Prefix 재사용, native 입력의 불필요한 오른쪽 padding 제거, selected-position full-vocabulary head, first-layer geometry cache, A factor cache, 전체 weight gradient를 생략하는 qualified VJP를 재사용한다. Upper keys/P를 detach하거나 오래된 값으로 캐시하지 않는다.

Geometry telemetry는 이미 존재하는 작은 B×B 행렬에서 계산한다. 비싼 spectrum은 entry/terminal 및 exact qualification에서만 계산하고 모든 step에 full eigensolve를 추가하지 않는다. 저장은 detached batched transfer로 수행한다. 세 gap 분해는 기존 probe/terminal forward의 hooks를 이용한다. 동일 endpoint 평가를 반복하지 않는다.

최신 profile의 실측으로 native fit, builder, solve/backward, pulse, terminal, evaluator 시간을 분리해 ETA를 갱신한다. Main 총시간 예측은 A/B 각각 관측된 batch fit 및 부대비용을 사용하며, 기존 v7 초반 시간이나 다른 GPU baseline으로 가속 배율을 단정하지 않는다.

## 9 산출물과 종료

필수 산출물은 source/config/model/input/evaluator hashes, CPU/GPU qualification receipts, candidate logs, terminal D/K/P/M 및 effective weights binding, layer/request/context realization, exact supported/unsupported reasons, frozen-key와 causal probe 비교, W/H/RNG rollback, main exactly-once history, per-case raw, W5 aggregate/retention, runtime/GPUh/peak memory다.

종료 상태는 `design_ready`, `implementation_qualified`, `pilot_complete`, `main_running`, `main_A500_complete`, `main_B500_complete`, `partial_or_failed`를 구분한다. Exact는 별도 `qualified_endpoint`, `rank_unsupported`, `numerically_unqualified`, `algebra_exact_but_FP32_unqualified` 상태를 갖는다. 현재 문서 작성 완료를 실제 pilot 완료로 표시하지 않는다.

향후 전달 대상은 기존 담당 체계인 GH→SH4/server4로 정리할 수 있다. 이번 작성은 메시지 전송이나 기존 실험 중단을 수행하지 않았다. Main은 두 arm의 W5 평가까지이며 새로운 writer arm, budget extension 또는 추가 GPU baseline은 포함하지 않는다.

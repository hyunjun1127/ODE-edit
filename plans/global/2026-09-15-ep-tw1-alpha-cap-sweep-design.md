EP-TW-1 alpha cap sweep의 검토와 실행 설계

작성: 2026-09-15. 상태: 설계 제안. 이 문서 작성 중에는 기존 저장 tensor의 CPU 기하 계산만 수행했다. 새 모델 forward, GPU 실험 제출, 실행 코드 수정, 새 sequential chain은 수행하지 않았다.

**첫 sweep은 사용자가 제안한 `alpha_cap=1, 10, 100, 없음`의 네 설정으로 진행하는 것이 적절하다.** 기존 cap=1 결과를 대조로 재사용하고, cap=10/100/없음은 각각 **W0/M0부터 동일 1,000개 요청을 B100×10으로 독립 실행**하는 것을 기본안으로 둔다. 기본 신규 범위는 3 chains, 30 batches다. Cap 외의 방법 구성과 후보 검사 메뉴는 고정한다.

이 설계의 질문은 “cap=1에서 매우 작았던 실제 보정을 키우면, native current-quality 조건을 지키면서 순차 preservation 결과를 개선할 수 있는가?”다. Cap sweep이 모든 편집 단계에 통용되는 최적 상수나 barrier/ODE의 필요성을 증명하는 실험은 아니다. 현재 cap은 gradient의 단위에 의존하므로, 성능이 좋아져도 portable한 parameter-free 방법이라고 주장하지 않는다.

기준은 [완료 실험의 독립 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review-ko.md)와 실제 실행 source `6d317bdb2660d7e9919bc3a9fb878564e9729e37`이다. 새 [설계 contract](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-contract.json)와 [cell 목록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-cells.csv)을 함께 작성했다. 기존 실행 결과와 이전 contract를 소급 변경하지 않는다.

**사용자 표의 비율은 최대 후보 C1의 크기로 정의하면 타당하다.** 이번에는 선형 비례만 계산하지 않고, 기존 cap1 경로의 10개 저장 episode에서 native target ball, write trust, FP32 후보 materialization을 다시 적용했다. Native/route tensor 20개의 hash가 이전 감사와 같음을 확인했고, cap1의 C는 10개 모두 정확히 재현됐다. 계산한 후보는 40개 cap-episode 조합의 C1/C05/C025, 총 120개다. Loss forward는 0회다.

| 설정 | 역할 | Clamp 전 C1의 선형 크기 | Clamp·trust·FP32 후 C1/native | Ball에 걸린 요청 수 / B100 |
|---|---|---:|---:|---:|
| cap=1 | 완료 실험 기준 | 0.00861–0.02649% | **0.00861–0.02649%** | 0–19 |
| cap=10 | 작은 확대 | 0.0861–0.2649% | **0.0861–0.2649%** | 0–20 |
| cap=100 | 중간 확대 | 0.8608–2.6490% | **0.8608–2.6486%** | 0–41 |
| cap 없음 | norm 정규화만 적용하는 큰 변화 대조 | 약 25% | **24.31–24.63%** | 87–100 |

이 비율은 **기존 cap1의 저장 parent, proposal, gradient, A를 고정했을 때의 기하**다. 새 cap10/100/없음 chain의 B002 이후 모델과 target, gradient, map은 자기 경로에서 다시 계산한다. 따라서 위 범위는 새 sequential 결과의 예측값이나 약속된 action 크기가 아니다. 특히 “네 정책이 저장된 미래 방향을 공유한다”는 실험으로 구현하면 안 된다. [기하 재계산 요약](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-alpha-cap-geometry/summary.json), [batch별 값](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-alpha-cap-geometry/by-batch.csv)

선택 후 크기는 별도다. 기존 cap1에서 실제 선택된 nonzero 보정은 0.00675–0.02355%였고 2개 batch는 RAW였다. 위 cap1 행의 0.00861–0.02649%와 다른 이유는 표가 항상 최대 C1을 나타내기 때문이다. 새 sweep에서도 C1/C05/C025/RAW 중 무엇이 선택되는지에 따라 실제 크기가 바뀐다. 주표에는 **최대 C1, RAW=0을 포함한 실제 selected, nonzero selected**를 각각 표시한다.

**cap 없음은 숫자 상한 하나만 없앤다.** 식은

\[
\alpha_{norm}=\frac{0.25\|V_p-W_{entry}\|_F}{\|dA_t\|_F+10^{-12}},
\qquad
\alpha=
\begin{cases}
\min(c,\alpha_{norm}) & c\in\{1,10,100\},\\
\alpha_{norm} & \text{cap 없음}.
\end{cases}
\]

Native target ball, \(\zeta=0.25\)의 executable write trust, finite 검사, 현재 E·strict 조건은 그대로다. 그러므로 “cap 없음=아무 제약 없는 편집”이 아니다. Target projection과 실제 FP32 연산 후에는 보정이 정확히 25%일 필요도 없다.

이 arm은 좋은 성능을 예상해서 넣는 후보라기보다, 기존 unit cap의 영향을 제거했을 때 norm 규약이 실제로 어떤 결과를 만드는지 확인하는 대조다. 저장 상태에서 cap 없음의 C1에 1차 근사를 적용하면 `D(RAW)+〈gD,C〉`가 **10/10 batch에서 음수**가 된다. 실제 KL은 수학적으로 음수가 될 수 없으므로, 이 큰 step에 작은 step의 선형 감소율을 외삽하면 안 된다는 직접적인 신호다. 음수로 예측한 loss를 실제 평가 결과처럼 보고하지 않는다.

또한 cap 없음에서는 87–100개 요청이 target ball에 걸리고, clamp 후 `〈gE,C〉`가 10/10 batch에서 양수다. 이는 실제 E가 반드시 증가한다는 증명이 아니지만, 보정 전 tangent 조건을 큰 유한 step의 품질 보증으로 쓸 수 없음을 보여준다. 작은 cap에서도 일부 부호 변화가 있으므로 모든 문제를 큰 cap 하나로 설명하지 않는다. [후보별 기하·선형 예측](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-alpha-cap-geometry/candidate-geometry.csv)

**기존 코드에 `inf`를 전달하는 방식은 사용할 수 없다.** 현재 `NumericalPolicy`는 수치 항목에 `math.isfinite`를 적용하므로 `alpha_cap=inf`를 거절한다. Null도 기존 숫자 validation에 그대로 넣으면 처리되지 않는다. Cap=10/100은 이미 양의 유한 값으로 표현할 수 있지만, cap 없음은 `alpha_cap_mode="disabled"`, `alpha_cap=null`처럼 명시적으로 구분하고 식에서 min만 생략하도록 구현해야 한다. 숫자 항목·mode의 validation도 분리한다. 계산된 α, target과 candidate는 여전히 유한해야 한다. 0, NaN, 큰 유한 상수로 cap 없음을 흉내 내지 않는다. [실행된 config 검증](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-completed-2026-09-15/execution-source/project/run_scripts/bg_tw_reference/ep_tw/policy.py:27)

각 arm의 runtime policy는 EP-TW-1이고, cap 설정·output identity를 별도로 봉인한다. 현재 runner는 chain 하나의 dispatch를 받는다. 따라서 전체 sweep의 3-chain 설계 파일을 기존 single-chain validator에 그대로 전달하지 않고, 각 arm용 lock에 독립된 한 chain을 명시한다. Cap1/10/100의 계산 순서를 바꾸지 않는 최소한의 parameter 지원으로 구현한다. 기존 사용자 지시로 생략한 수치 진단을 이번 cap 비교의 필수 선행조건으로 다시 도입하지 않는다. 생략 상태를 numerical PASS로 바꾸지도 않는다.

**비교 범위는 다음으로 고정한다.**

| Arm | 시작과 길이 | 기본 실행 상태 | 핵심 역할 |
|---|---|---|---|
| CAP1 | W0/M0 → B001–B010 | job47962 완료 결과 재사용 | 현재 방법의 기준 |
| CAP10 | W0/M0 → B001–B010 | 신규 1,000개 sequential | 약 0.1%대 최대 후보의 효과 |
| CAP100 | W0/M0 → B001–B010 | 신규 1,000개 sequential | 약 1%대 최대 후보의 효과 |
| NORM_ONLY | W0/M0 → B001–B010 | 신규 1,000개 sequential | unit cap 없는 norm 규약의 결과 |

모든 arm은 동일한 1,000개 요청을 사용한다. 신규 처리 횟수는 3,000회이지만 서로 다른 새 요청 3,000개가 아니다. W50 또는 이전 cap1의 중간 checkpoint에서 시작하지 않는다. 처리 순서는 CAP10→CAP100→NORM_ONLY를 권장하되, 앞 arm의 성능을 보고 뒤 arm의 cap·beta·threshold를 바꾸지 않는다. 세 arm은 서로의 최종 모델을 이어받지 않는다.

CAP1 재사용은 초기 모델·M0·seed·context, teacher, native fitting, projection, materialization, selector, observer의 과학적 동작을 유지하는 경우에 적용한다. Shared runtime 변경이 이 경로를 바꿀 수 있다면 CAP1도 새로 1,000개 실행해 신규 4 chains로 맞춘다. 단순 경로 이름·cap metadata 변경만으로 같은 실험 전체를 무조건 반복할 필요는 없다. 첫 batch의 parity는 구현 확인에 유용하지만 새 method의 1,000개 성능 검증을 대체하지 않는다.

기존 N4와 사용자 baseline 다섯 종류는 결과표의 reference로 계속 사용한다. 이들은 ours를 만들기 위한 입력이나 KL allowance 산출용 대조가 아니다. 현재 cap sweep 때문에 다섯 baseline을 모두 다시 실행할 필요는 없다. 특히 N4를 먼저 1,000개 돌려 cap이나 KL 한도를 정하는 단계를 추가하지 않는다.

**Cap 이외의 과학적 구성을 고정해야 한다.** W0 revision, seed 20260915, sample order, parsed contexts, L4 hparams/P, dtype·kernel·microbatch, native target 최대24 Adam update/25 loss 평가와 early stop, native anchor/radius, fixed W0 C4 teacher/S64, mean E와 strict ID 규약을 완료 실행과 맞춘다. Teacher192는 이미 완료된 job47592 산출물을 재사용한다. 이전 문서의 오래된 PENDING 문자열을 현재 상태로 사용하지 않는다.

한 batch에서 native fitting 1회, current gradient sweep 1회, S64 gradient sweep 1회를 유지한다. 그다음 `RAW, C1, C05, C025`를 같은 Vp에서 구성해 실제 finite min-D 선택을 수행한다. 같은 bytes 후보 deduplication과 RAW 우선 tie 규약도 같다. `.5/.25`는 correction만 줄이는 값이며 native write 전체를 축소하지 않는다. Cap sweep과 동시에 target refresh, quality restoration, old replay, S64 확장, E 양의 tolerance, gradient 업데이트 횟수 증가를 넣지 않는다.

앞선 리뷰에서 제안한 후보 검사 조기 종료도 **이 첫 sweep에는 함께 넣지 않는다.** Cap이 커지면 D가 scale에 대해 단조롭지 않을 수 있고, 최초 feasible 후보와 finite min-D 후보가 달라질 수 있다. Cap 변화와 search 변화가 동시에 들어가면 결과의 원인을 분리하기 어렵다. 현재 추가 비용의 상당 부분이 screen인 사실은 남겨두고, cap 비교 후 별도 비용 ablation으로 진행한다.

**현재 메뉴는 cap 간에 포함 관계가 없다.** 같은 parent에서도 cap10의 C025는 대략 cap2.5의 크기이고 cap1 C1을 포함하지 않는다. Cap100 메뉴도 cap10의 좋은 후보를 자동으로 포함하지 않는다. Ball projection 후 보정 방향까지 달라질 수 있으므로 단순히 숫자 α를 맞춘 것과 같은 후보가 된다고 가정하지 않는다.

특히 저장 상태의 NORM_ONLY 후보는 C1 약24.3%, C05 약12.2%, C025 약6.1%다. 작은 cap에서 가능했던 0.1% 또는 1% 보정을 그 메뉴가 포함하지 않는다. 따라서 NORM_ONLY가 RAW를 자주 선택하면 “검사한 큰 후보 세 개가 통과하거나 개선하지 못했다”는 결론은 가능하지만, 모든 중간 크기의 보정이 불가능하다거나 보존 공간이 없다는 결론은 불가능하다.

이 한계 때문에 첫 sweep에 cap1000이나 많은 beta를 미리 추가하지는 않는다. 네 설정으로 크기 제한이 주된 문제인지부터 본다. 결과에서 cap100과 NORM_ONLY 사이의 비어 있는 구간이 실제 해석을 막을 때만, cap1000 또는 작은 beta를 포함하는 별도 메뉴를 후속 개발 실험으로 선언한다. 중간 결과를 보고 기존 arm의 메뉴를 조용히 확장하지 않는다.

**모든 arm은 성능이 좋든 나쁘든 10개 batch를 완료한다.** B001만 보고 승격·탈락시키지 않는다. 보정 후보가 모두 실패하면 native RAW를 commit하고 history를 한 번 등록한 뒤 다음 batch를 계속한다. 많은 RAW 선택은 정상적인 과학 결과이지 실행 중단 조건이 아니다. Native 자체의 nonfinite, 실제 resource failure, 잘못된 state 연결 등 기술 실패는 별도로 기록하고 미완료 실행을 full1000으로 보고하지 않는다. 재개할 때 finalizer를 중복 실행하거나 cap을 바꿔 같은 chain으로 이어붙이지 않는다.

기존 관측 schedule을 유지하면서 다음 정보를 원시 값으로 남긴다.

- 각 후보: cap mode/값, αnorm/실제 α, cap 활성 여부, pre-ball norm, post-ball norm, 실제 materialized correction/native, ball hit, trust retraction, `〈gE,C〉`, `〈gD,C〉`, 예측 ΔE/ΔD와 actual ΔE/ΔD, strict lost ID, 선택·탈락 사유.
- 각 batch: 같은 요청에서 entry→RAW→selected R/P/N, canonical/P TF-strict, 두 P 모두 strict, accepted-old 결과. Official P/N은 observer로만 사용한다.
- 순차 endpoint: W10 전체 요청 원분모, at-write→W10, 첫500의 W5→W10, ACTIVE/SUPERSEDED, true/new NLL과 paired tail, acceptance coverage.
- Reference: S64는 각 batch, Dev128은 W5/W10의 observer. Fixed original teacher를 batch마다 교체하지 않는다. Report256/Audit128/MMLU68/FutureN 완료를 이번 sweep의 선행조건으로 두지 않는다.
- 비용·상태: native fitting, map, E/D F/B, teacher read, 모든 rejected probe, observer, history/finalization, 저장/I/O와 실제 보존 파일. RAW=0을 포함한 selected norm 분포와 RAW 선택 횟수를 반드시 보고한다.

집계는 cap별 C1의 기대 크기만 비교하지 말고, **실제 선택된 보정 크기와 선택 빈도**를 함께 본다. 큰 cap에서 후보의 D 감소가 커도 매번 RAW가 선택된다면, 그 정책의 실행된 보정은 0이다. 반대로 cap 활성 여부가 바뀌어 norm 상한에 도달한 batch에서는 명목 cap의 비례 확대를 가정하지 않는다.

Endpoint preservation 판단은 S64 local 개선만으로 내리지 않는다. C4 S64와 Dev128, N true/new NLL·NS, active old R/P, PS·strict와 비용을 함께 비교한다. Cap 설정 선택에 이 결과를 사용하면 해당 요청 순서의 실험은 개발 데이터다. 독립 확인은 선택을 고정한 뒤 다른 order/full10k와 reporting panel에서 한다. 단일 order의 문항 수를 독립된 순차 반복 수로 바꾸지 않는다. 이번 설계에서는 새로운 KL allowance나 임의의 PS 허용 손실선을 추가하지 않는다. Trade-off가 있으면 그 상태로 보고한다.

결과별 후속 방향은 다음과 같이 제한한다.

| 관측 | 허용되는 해석과 다음 선택 |
|---|---|
| cap10/100에서 selected action이 커지고, editing을 유지하며 endpoint 보존이 개선됨 | cap1의 작은 action이 개선을 제한했다는 가설이 강해짐. 해당 규모를 중심으로 다음 검증을 좁힘 |
| 큰 후보의 D는 좋아지지만 E 때문에 대부분 RAW | 관측 메뉴의 finite quality 문제가 남음. 중간 action 공백·target projection·curvature를 로그로 구분하고, 필요할 때 quality restoration을 별도 비교 |
| 큰 후보에서 actual D가 나빠짐 | 작은 step의 보존 gradient 방향을 큰 유한 step까지 외삽할 수 없음. Norm-only가 자동으로 올바른 operating point인 것은 아님 |
| 선택된 보정과 local D 이득은 커지지만 N/old retention이 개선되지 않음 | 크기 외에 surrogate·반복 간 영향의 문제를 검토. 더 큰 cap만 계속 늘릴 근거가 약함 |
| 네 설정 모두 최종 결과가 비슷함 | 더 큰 실제 action도 비슷했는지, 모두 RAW/작은 후보로 수렴했는지 먼저 구분. 이를 구분하지 않고 amplitude 가설을 반박하지 않음 |
| NORM_ONLY만 좋아짐 | cap 제거를 후보로 남기되, 한 개발 order의 결과로 ODE/barrier·25% 상한의 일반적 최적성을 주장하지 않음 |

**예상 신규 비용은 현재 계측을 기준으로 약 6.4 GPUh다.** Cap1의 실제 allocation 7,694 GPU초=2.1372 GPUh를 신규 3 chains에 선형 적용한 값이다. CAP1 재실행까지 필요하면 약 8.55 GPUh다. 동일 하드웨어·평가/저장 schedule을 가정한 예상이며 실제 시간이 아니다. Teacher 준비 비용은 재사용하므로 반복 계수하지 않는다. Native early stop, branch 변화, 자원 경합과 I/O에 따라 실행 시간은 달라질 수 있다.

산출물은 각 arm의 W10 L4 weight와 W0/model/token/context identity를 적어도 유지해 후속 inference 평가가 가능해야 한다. Full resume을 주장하려면 M/history/RNG/ledger까지 실제 보존해야 한다. 모든 중간 checkpoint를 영구 보존한다는 가정으로 보고서를 작성하지 말고, native/route tensor 및 checkpoint의 생성 당시 inventory와 현재 retention을 구분한다. 기존 cap1의 W4는 남은 tensor로 재구성할 수 있지만 전체 resume checkpoint는 현재 경로에 없다는 이전 감사 결과를 그대로 유지한다.

이 첫 sweep의 권고는 **cap 네 설정, 신규 3개의 W0→SEQ1000, cap만 변경, 기존 finite menu 유지**다. 실제 개선이 관측된 action 규모를 확인한 뒤에 검사 비용이나 quality restoration을 단계적으로 추가하는 편이 결과를 해석하기 쉽다.

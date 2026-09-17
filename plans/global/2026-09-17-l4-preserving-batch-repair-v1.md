# L4-preserving batch-wise repair: full native write 뒤의 조건부 L8 복구

2026-09-17. 사용자 방향을 반영한 v1 설계. 모델 runner 구현·GPU 제출·효능 검증은 아직 아니다. CPU 수식 fixture와 계약 검사를 수행한다. 기존 실험 source와 산출물은 수정하지 않는다.

사용자 후속 지시 반영: **paraphrase를 target으로 하는 별도 세트는 만들거나 사용하지 않는다.** 공식 paraphrase는 평가 observer로만 사용한다. 온라인 편집 보호는 현재·과거 canonical rewrite로 한정한다.

## 1. 방법의 정체성과 증거

목표는 L4-only의 편집 성능을 유지하면서 original-knowledge preservation 비용을 줄이는 것이다. L4는 target writing, L8은 그 뒤의 preservation repair를 담당한다. Layer 사용 수·norm 균등 분담·gate 합1을 목적에 넣지 않는다.

Cold7 LD는10batch 중9회(.75,.5)를 택했고 N4보다 NS+2.84pp, PS−1.25pp였다. 이는 다층 조합에서 유용한 신호가 반복됨을 보여주므로, static(.75,.5) 추가 결과를 기다려야만 새 방법을 설계할 수 있는 것은 아니다. 다만 LD의 L8 적용9회 모두 partial L4 대비 S64 KL은 증가했다. 이번의 full-L4 후 repair는 그 관측에서 자동으로 도출되는 성공 사례가 아니라, 역할을 바꾼 새 가설이다.

관련 내부 선행 B-OS는 이미 full L4 동결·L8 보정을 시도했다. Middle에서 Base KL .034895→.113661, Current NLL .047468→.062879였고 두 PCG RHS가20회 안에 수렴하지 않았다. 일차 equality는 만족해도 실제 endpoint는 악화했다. 그러므로 역할 분담 자체를 새로운 발견으로 포장하지 않는다. 이번 변경은 **고정 W0 목적, 명시적인 작은 응답 행렬, 현재·과거 rewrite guard, 실제 endpoint acceptance와 zero fallback**이다.

근거: [cold7 독립리뷰](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/global/2026-09-17-local-z-seven-arm-independent-review.md), [기존 B-OS 설계](2026-09-11-multilayer-joint-edit-and-compensation-design.md), [B-OS 실측](/mnt/raid5/janghj/.codex/worktrees/odeedit-local-z-independent-audit-20260917/experiment-reports/servers/server2/multilayer-damage-compensation-b-2026-09-11-v1/partial-recall-r1/diagnostic-report-ko.md).

## 2. 상태, target, 실제 수정 범위

각 batch t에서 상태를 다음과 같이 정의한다.

    We = 직전 batch의 최종 모델; 과거 L8 repair 포함
    WN = NativeL4(We, current requests)
    W(c) = WN with W8 := WN8 + Σ_j c_j Q_j

1. Native local-z4를 L4에서 계산하고 full native update를 적용한다. L4 gate=.75, terminal-z 배분, target 강도 축소를 넣지 않는다.
2. WN을 임시 anchor로 봉인한다. 모든 repair 후보는 동일 WN에서 시작한다. W4와 모든 비L8 parameter는 후보·최종상태에서 WN과 동일해야 한다.
3. **L8에서는 target_new compute_z를 호출하지 않는다.** L8 방향은 실제 출력 preservation gradient 및 edit 보호 response에서 구한다. Repair에 activation target을 별도로 fit한 뒤 write하는 단계도 없다.
4. Repair 실패 시 WN을 선택한다. 이때 이번 Δ8=0이며 이전에 누적된 L8 repair를 제거하는 것은 아니다.
5. 다음 batch의 z4는 자기 branch의 실제 We에서 계산한다. L4-preserving은 한 batch의 native W4를 유지한다는 뜻이다. 독립 N4 lifelong chain과 모든 W4 bytes가 같다는 뜻은 아니다.

이 업데이트는 L8 down-projection weight 자체에 적용하므로 모든 token에 영향을 준다. 일부 subject 위치에만 hook한 효과를 실제 write로 대신하지 않는다. L4 weight 고정만으로 편집 출력이 고정되는 것은 아니다.

## 3. 온라인 입력과 관측 분리

| 패널 | 용도 | teacher / target |
|---|---|---|
| S64 | repair 목적·gradient·곡률·acceptance | 고정 W0 full-vocabulary teacher |
| Current R100 | canonical 편집 보호 | current target_new; 비교기준 WN |
| Past R64 | 이전 active edit의 canonical rewrite에 대한 추가 손상 보호 | 최신 유효 target_new; 비교기준 WN |
| Dev128 | 독립 generic observer | 고정 W0 teacher; 선택에 사용 금지 |
| 공식 Current/seen R·P·N | 성능 평가 | 기존 정본 정의; P/N은 제어기에 입력 금지 |

별도의 paraphrase 생성·저장·target 학습·guard를 두지 않는다. 공식 P를 분할하거나 이름을 바꿔 온라인 입력으로 사용하는 것도 금지한다. Native L4 writer의 기존 context 절차는 그대로 유지하되, 이를 새로운 paraphrase 보호 세트로 재해석하거나 확장하지 않는다. 공식 P/N은 선택 봉인 이후 observer에서만 평가하며 후보 선택이나 step 크기에 반영하지 않는다.

Past64는 received active-fact ledger에서 current overwrite를 제외하고 기존 hash-priority 규칙으로 선정한다. 성공한 edit만 모으지 않는다. S64와 Dev128은 기존 Teacher192의 token/model 연결을 확인해 재사용하며, teacher를 WN 또는 직전 We로 바꾸지 않는다.

S64에 편집 fact와 충돌하는 내용이 있을 가능성을 지우지 않는다. 그런 충돌은 edit 제약을 우선하고 repair off로 귀결될 수 있다. C4 복원 자체를 요청한 edit보다 우선하지 않는다.

## 4. 목적과 실제 품질 제약

목적은 B(W)=mean_x KL(p_W0(.|x) || p_W(.|x))다. 기존 S64의 vocab sum → scored128-position mean → document mean을 유지한다. 목표는 B(WN+Δ8)<B(WN)이며, W0 지식을 완전히 복구하거나 N을 반드시 올린다는 보장은 아니다.

온라인 품질 기준은 WN에서 한 번 봉인한다.

- L_R, L_H: 각각 Current canonical과 Past canonical의 target-new 평균 NLL. token→context→request 순서로 평균하고 현재·과거를 하나의 평균으로 혼합하지 않는다.
- S_R, S_H: WN에서 성공한 정확한 rewrite ID 집합. Target-new 전토큰 teacher-forced strict 성공 ID를 보존한다.
- 편집 request에 old target이 제공되는 경우, 현재·과거 canonical rewrite의 new/old NLL 선호 성공 ID도 별도 보존한다. Old target이 없는 경우 이 항목을 NOT_AVAILABLE로 기록하며 argmax와 동일시하지 않는다.

실제 후보 acceptance는 각 평균 L_k(W)≤L_k(WN)+ε_L, 위 성공 ID subset, finite state, W4 고정, B(WN)−B(W)>ε_B를 모두 요구한다. **max(E_N4,.05) plateau는 제거한다.** NLL mean은 분리해 보호하지만 모든 요청의 NLL 무악화를 보장하지는 않는다. 요청별 악화수·p95/p99·margin tail도 기록한다.

Past 기준은 We가 아니라 WN이다. We→WN에서 native L4가 이미 낸 손실과 WN→Wfinal의 추가 손상을 분리한다. 이 선택으로 zero가 항상 품질-feasible하며, native 단계에서 잃은 모든 과거 edit를 복구한다고 주장하지 않는다.

온라인 조건은 현재·과거 rewrite만 보호하며 공식 PS 유지까지 수학적으로 보장하지 않는다. **RS·PS 유지라는 최종 평가 목표는 그대로 유지**하고, 공식 PS 비열화는 독립 observer 결과로 판단한다. PS가 떨어지면 방법의 목표를 충족하지 못한 결과로 기록하며, NS 이득으로 상쇄하거나 P를 사후 controller에 편입하지 않는다. 평균 RS/PS 동점이어도 paired lost/gained를 반드시 보고한다.

## 5. 무엇을 적응적으로 계산하는가

layer weight softmax가 아니라 **이번 L8의 update 방향·크기·사용 여부**를 response에 맞춰 계산한다. 시작 시점의 primary 신호는 다음과 같다.

    D_N = B(WN)                         누적 W0 drift
    delta_D_native = B(WN)-B(We)        이번 full L4의 추가 손상
    g_B = ∇W8 B(WN)                    가능한 repair 방향
    edit response Jacobian             어떤 repair가 edit을 얼마나 바꾸는가
    constrained predicted gain          품질 조건 아래 남는 복구 여력
    actual gain / predicted gain        이 batch에서 근사가 믿을 만한가

큰 D_N이나 큰 z residual만으로 L8를 켜지 않는다. 실제 feasible repair gain이 있어야 한다. delta_D_native≤0이어도 과거 누적 손상을 더 줄일 가능성은 있으므로 D_N>0이면 proposal은 허용한다.

## 6. 작은 방향 공간: v1은 최대3개

WN에서 L8에 대해서만 g_B, g_R, g_H를 구한다. 각각 Base KL, Current canonical NLL, Past canonical NLL의 gradient다. Past가 비면 g_H는 없어 B1은 최대2방향이다. 전체 W8 gradient의 span을 FP64 Gram/QR로 정규직교화하여 Q1..Qm을 만든다. m≤3이며 zero·선형종속 방향은 제거한다. g_B가 finite numerical zero이고 검증된 descent가 없으면 그 batch는 repair off다.

**v1은 별도 key basis V8나 native P8 null-space를 추가하지 않는다.** 이중 압축으로 복구 방향을 미리 제거하는 교란을 줄이기 위해서다. Native P4는 L4 writer에서 그대로 유지한다. L8 repair는 edit response 제약으로 보호한다. Native P8 적용은 별도 후속 ablation이며, 그 안에서 실패했다고 전체 L8 repair가 불가능하다고 주장하지 않는다.

Q는 full W8 shape이며 CPU에 보관할 수 있다. m=3·FP32이면 약0.70GB 추가 방향 저장량이다. 후보는 WN+ΣcQ로 materialize한다. Random LoRA 초기화나 update norm 재확대를 하지 않는다. Gradient span 밖의 가능한 repair는 v1에서 탐색하지 않는다.

후속으로 active guard gradient를 추가해 m≤8로 확장할 수 있으나 **v1 결과에 소급 혼합하지 않는다.** 작은 공간에서 zero가 나온 것은 현재 basis·제약·reference의 결과다.

## 7. 측정된 response와 명시적인 작은 최적화

각 방향별 JVP로 gradient b_j=⟨g_B,Q_j⟩와 guard response Jacobian A를 만든다. Guard scalar에는 위2개 mean NLL과 보호 성공 rewrite의 target-token margin, 사용 가능한 new/old preference margin이 들어간다. Past가 비면 해당 mean·margin 행은 없다. Margin 부호를 뒤집어 모든 행을 A c≤s로 통일한다. Mean NLL의 모델상 slack은0, 성공 token margin의 slack은WN에서의 현재 비음수 margin이므로 c=0은 feasible하다. Argmax tie에서 target token이 선택된 strict 성공은 margin0일 수 있다. New/old NLL 선호의 성공 margin은 양수다. Numerical tolerance는 실제 검증에만 별도 적용한다.

토큰 margin의 경쟁 token은 WN의 최상위 non-target로 고정하여 미분한다. 새 경쟁 token이 등장하는 비선형 효과는 실제 full-vocabulary strict 검사에서 잡는다. 모든 logit의 exact equality나 KL(WN||candidate)의 first-gradient equality는 쓰지 않는다. 후자는 anchor에서 gradient0이라 첫 step을 보호하지 못한다.

Base의 작은 GN 행렬은 다음과 같다.

    H_B[j,k] = mean_x J_z,j^T [diag(p_N)-p_N p_N^T] J_z,k

Teacher는 p0지만 여기의 곡률 확률은 p_N이다. Gradient에는 p_N−p0가 들어간다. H_B는 PSD GN 근사이며 정확한 weight Hessian이 아니다.

Full-vocabulary logit JVP는 문서별로 누적하고 폐기한다. 전체 S64×모든 방향×vocab tensor를 상주시켜서는 안 된다. m회 directional sweep에 각 context의 scalar response를 함께 수집하고 작은 m×m H_B만 남긴다. 행렬이 작으므로 거대 matrix-free PCG를 호출할 이유가 없다.

수치 whitening에는 Htilde=H_B+λnum I를 사용한다. λnum은 condition number가 최대10^6이 되도록 필요한 최소 nonnegative diagonal shift다. 양의 hmax가 있고 hmin≤hmax일 때 λnum=max(0,(hmax−10^6 hmin)/(10^6−1)). 유의한 음의 고유값은 PSD/derivative 기술 실패로 분리한다. H≈0이면서 g_B가 유의하면 임의 floor로 진행하지 않고 수치 연결을 점검한다.

R=Htilde^(−1/2), c=R u로 두면 작은 QP는 다음과 같다.

    min_u  (R^T b)^T u + 0.5 ||u||²
    s.t.   A R u ≤ s
           -r ≤ u_j ≤ r

λnum은 수치 regularization이며 강한 과학적 보존 가중치를 학습한 것으로 부르지 않는다. QP solver는 double precision에서 primal/dual feasibility, stationarity, complementarity residual을 기록한다. 순서가 같고 수치 tie이면 작은 norm, zero를 우선한다.

최초 r0=sqrt(2 D_N/m)로 둔다. m은 고정3이 아니라 zero·종속 방향 제거 후의 실제 유효 방향 수다. m=0이면 radius 계산 없이 zero를 선택한다. 따라서 최초 box 안에서는 0.5 c^T Htilde c≤D_N이다. 현재 관측한 drift를 수치 탐색 scale로 사용하는 선택이며 실제 KL 감소 보장이나 capacity 상한은 아니다. D_N≤ε_B이면 zero. 현재 손상보다 큰 근사 변화부터 시험하지 않는 engineering 규칙임을 공개하고 sensitivity는 별도로 다룬다.

## 8. 실제 acceptance와 batch-wise adaptive control

한 batch에서 최초 r0를 포함해 최대6개의 QP endpoint를 검사한다. 실패 시 r←r/2로 줄여 같은 response model의 QP를 다시 풀며 축소는 최대5회다. Rejected 후보는 WN으로 완전히 rollback한다. Warm-start는 작은 QP 내부에서만 허용하고 모델 state는 공유하지 않는다.

    predicted = -[(R^T b)^T u + 0.5||u||²]
    actual = B(WN)-B(WN+Σ(Ru)_j Q_j)
    agreement = actual/predicted

predicted>ε_B, actual>ε_B, agreement≥0.1 및 모든 finite 품질 조건을 만족하는 최초 후보를 commit한다. 이것은 radius가 큰 순서의 first-acceptable controller이며 전체 비선형 최적해나 검사후보 최솟값을 보장하지 않는다. 최대6회 안에 없으면 zero를 commit한다. 한 batch에서 accepted repair는 최대1회다.

ε_L=1e−4 nats/token, ε_B=1e−6, QP scaled KKT tolerance=1e−8을 첫 numerical contract로 고정한다. Post-N4 반복 관측 오차가 각 tolerance의 절반 이내임을 실제 모델에서 먼저 확인한다. 실패하면 tolerance를 품질 결과에 맞춰 자동 확대하지 않는다. 수치 원인을 분리하고 새 버전을 봉인한다. ε_L은 .05 plateau 같은 과학적 품질 완화가 아니다.

Guard 실패, Base 악화, 예측 불일치, 작은 실제 개선, QP residual 실패, basis zero, FP32 materialization zero를 각각 다른 사유로 기록한다. 기술 오류는 정상 품질 off와 합치지 않는다.

이 과정에서 layer-on은 nonzero accepted Δ8 여부이며 강도는 최종 ||Δ8||·Fisher 크기·선택 radius로 정량화한다. Softmax나 수동(.75,.5) menu는 필요 없다. 다만 basis 크기, trust 축소 규칙, numerical threshold라는 engineering 선택은 여전히 존재한다.

## 9. 누적 상태·history·관측 budget

L4 native M4는 최종 commit에서 이번 Current keys를 정확히1회 append한다. 후보 생성/평가에서는 append0회. L8 repair에는 native solver가 없으므로 operational M8/P8를 사용하거나 임의로 append하지 않는다. L8 current/past response는 현 WN에서 다시 계산하며 예전 batch의 key/Jacobian을 그대로 사용하지 않는다. M8가 필요하면 별도 diagnostic으로만 명명한다.

각 batch의 checkpoint는 W4/W8, M4, received active ledger, native context/token identity, Current/Past rewrite 기준값, Q/b/A/H/λnum, solver 결과, probe 상태·실패 사유·선택을 연결한다. Full backbone pointer/version guard와 selected weight byte hash를 구분한다. L8 off여도 과거 누적 A8가0이라고 보고하지 않는다.

Budget에는 세 값을 구분한다.

1. **사용량:** 이번/누적 layer delta norm, path/net/energy, W0-S64/Dev drift.
2. **현재 repair 여력:** 같은 trust box의 unconstrained gain Γfree와 constrained gain Γsafe, ratio Γsafe/Γfree, 실제 repair gain, active guard 및 dual multiplier. 최적 QP를 정확히 풀었다면0≤Γsafe≤Γfree. Γfree가 수치적으로0이면 ratio는NA다. Ratio는 현재 basis의 편집 충돌 정도이며 남은 사실 수가 아니다.
3. **근사 신뢰성:** predicted/actual gain과 agreement, radius 축소 횟수, 유효 후보 비율.

Base·guard는 token/문서 수가 다르므로 mean 단위·row scaling을 기록한다. Dual multiplier를 다른 단위의 row끼리 그대로 비교하지 않는다. Physical norm share는 functional contribution share가 아니다.

## 10. 비용 사양과 재사용

각 batch의 native z target는 N4와 같은100회이며 L8 target call은0이다. 추가로 Base/Current-R/Past-R 최대3개 gradient sweep, 최대m=3 JVP sweep, 작은 QP solve≤6, 실제 후보 평가≤6이 있다. Past가 빈 B1은 gradient·JVP가 각각 최대2개다. Paraphrase 생성·온라인 학습·guard 비용은0이며 공식 P observer 비용은 별도로 남는다. 한 sweep이 한 model forward라는 뜻이 아니며 문서·microbatch·token·backward/JVP·teacher I/O를 각각 기록한다.

FullVocab Fisher streaming은 구현 복잡도와 비용을 갖는다. 과거 B-OS의 두 RHS×20 PCG와 구조가 다르지만 실제 속도 우위를 미리 주장하지 않는다. 초기 numerical pilot에서 단1batch의 wall/peak memory를 재고10batch 비용을 추정한다. 더 큰 budget이 필요하다고 method tolerance를 바꾸지 않는다.

재사용 가능한 것은 cold7 snapshot/restore/observer 거래, W0 Teacher192, native L4 fitter/finalizer, saved-NLL evaluator, source/state hash 계측이다. EP의 target-space build_correction/edit-z ball은 L8 preservation repair의 main parameterization으로 가져오지 않는다. B-OS의 SelectedView는 참고 가능하나 거대 PCG 경로는 재사용하지 않는다.

## 11. 실험 계획: 모든 main chain은 W0에서

공통 capsule은 cold7의 Llama3-8B-Instruct revision·FP32/eager·TF32off·fixed10k order·context와 seed20260916을 우선 유지한다. Repair source와 rewrite 보호 규약을 별도 봉인한다. 실제 실행 전에 config와 기술 검사 결과를 lock해야 하며, 이 문서는 제출 지시가 아니다.

**단계0: first100의 bounded numerical/mechanism pilot.** W0→native L4를 한 번 수행하고 QP model 및 finite 후보를 검사한다. 두 FD scale에서 derivative 부호/크기, parameter vs functional materialization, L4 고정, rollback/history0, repeat noise, full-vocab GN PSD/whitening/KKT를 확인한다. Tiny mathematical fixture PASS를 actual-model PASS로 바꾸지 않는다. Official P/N은 선택 봉인 후 raw/repair 후보에 observer로만 기록한다.

**단계1: 신규 cold B100×10, 5 arms.** N4, REFIT4, 기존 LD, R-GD, R-QP. 각 arm은 자체 W0→W10 trajectory를 갖는다. R-GD는 같은 L8 Base descent 한 방향에서 guard 없는 scalar quadratic proposal을 구하고 동일한 curvature normalization/trust 축소/실제 품질 acceptance를 적용한다. R-QP는 proposal 단계부터 guard response를 사용하여 방향을 바꾼다. 따라서 R-GD도 실패한 품질 후보를 commit하지 않으며, R-QP의 최대3방향·제약 응답 모델이 실제 추가 이득을 내는지 분리한다. 각 방법의 방향 수에 따른 비용 차이는 그대로 기록하고 dummy backward로 비용을 맞추지 않는다. 모든 arm을 지표를 보고 중도 탈락시키지 않는다. 기술 실패만 별도 종료한다.

| 핵심 비교 | 질문 |
|---|---|
| R-QP−N4 | full L4 편집 성능을 유지하며 locality를 개선했는가 |
| R-QP−R-GD | 단순 추가 보존 최적화보다 제약 방향 설계가 유효한가 |
| R-QP−REFIT4 | 같은 층 추가 최적화보다 추가 layer의 repair가 유효한가 |
| R-QP−LD | edit 분담에서 발생한 PS 비용을 줄이며 locality를 얻는가 |

매batch We/WN/selected의 current R/P/N, W5 seen500, W10 seen1000, 각 cohort at-write→W10, W0-success 조건부 N유지를 평가한다. B1/B5/B10에는 선택을 봉인한 뒤 검사한 모든 실제 후보의 Current official P/N과 고정 early cohort를 관측한다. Model-level candidate forward 수를 비용에 별도 포함한다. 후보 N을 이용한 사후 가상 chain은 만들지 않는다.

**승격 기준:** ‘RS·PS 유지’의 과학적 목표는 N4 대비 손실 허용폭0pp다. point estimate뿐 아니라 paired lost/gained, request-cluster 불확실성, Current/old cohort를 함께 보고한다. 작은 NS 이득으로 PS손실을 상쇄해 통과시키지 않는다. 한 seed의 우연한 양의 차이는 보편적 비열화 증명이 아니므로 “이번 fixed stream에서 관측”으로 한정한다. Off가 많으면 실패 사유를 그대로 보고하며 품질 허용량을 사후 늘리지 않는다.

이후 locked method를 W0→fixed10k로 확장하고 long-horizon N forgetting·PS유지·비용을 검사한다. 이미 본1000은 개발구간이며 나머지 구간도 역사적 연구에서 노출됐다면 blind라고 명명하지 않는다.

## 12. 실패 결과도 무엇을 알려주는가

- Γsafe≈0: 현재 response basis에서 edit 보호와 repair가 충돌함. intrinsic capacity 소진이라는 결론은 아님.
- Γsafe>0이나 actual≤0: 근사·step·수치 모델 문제. 역할 가설의 즉시 반박 아님.
- S64 개선, Dev/N 미개선: reference 전이 문제. Layer repair 성공을 locality 전체로 확대하지 않음.
- Rewrite 유지, official PS 손실: rewrite 보호만으로 paraphrase 일반화가 유지되지 않은 결과. RS·PS 유지 목표를 충족했다고 부르지 않으며 공식 P를 controller로 옮겨 같은 결과를 재평가하지 않음.
- current는 유지, old P/N 손실: sampled Past 범위/누적 native 손상/이전 repair와 새로운 L4 상호작용을 분리.
- R-GD와 동일: 작은 QP 방향 제어의 추가 필요성은 미확인.
- 모두 zero: 강한 N4 fallback이 유지된 유효한 음성 결과. L8를 쓰기 위해 threshold를 바꾸지 않음.

## 13. 관련 연구와 주장 범위

[DOW-KE](https://arxiv.org/html/2608.16932v1)는 실제 update를 직접 최적화한다. [LOKI](https://arxiv.org/html/2606.19679v1)는 projected gradient와 norm 제약을 사용한다. [LyapLock](https://aclanthology.org/2025.emnlp-main.327.pdf)은 누적 preservation 제약의 동적 제어를 다룬다. [KLOD](https://arxiv.org/html/2608.27839v1)는 target 증폭과 비대상 출력분포 보존을 분리한다. 따라서 dynamic, gradient projection, 출력 KL, 후속 최적화 자체의 최초성을 주장하지 않는다.

검증할 주장은 **강한 단일 layer writer를 유지하고, 추가 layer의 자유도를 편집량 분배가 아닌 출력 손상 복구에 사용하며, batch별 기능 제약과 실제 복구 이득을 만족할 때만 적용한다**는 것이다. RS/PS 보존과 NS 향상의 동시 실증 및 기존 B-OS 대비 계산·수치 신뢰성 개선이 필요하다.

구조화 사양: [contract](2026-09-17-l4-preserving-batch-repair-contract-v1.json), [실험 cells](2026-09-17-l4-preserving-batch-repair-cells-v1.csv). 수식 검산: [CPU 결과](../../audits/global/2026-09-17-l4-preserving-repair/math-checks.json).

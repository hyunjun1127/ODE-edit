# BG-TW 파이프라인 재검토: 사전 보존 한도 없이 무엇을 최적화할 것인가

작성: 2026-09-15 KST. 상태: 원문·기존 설계·Server4 보고·구현 source 검토 및 설계 수정. 새 controller 구현, GPU 실행, 원격 지시 변경 결과가 아니다.

## 1. 판단

**N4 순차 결과로 보존 한도를 정하는 단계를 주 방법에서 제거한다. 외부에서 정당한 한도가 주어지지 않은 현재에는, 편집 품질을 유지하는 후보 중 고정 W0 대비 출력 KL이 작은 후보를 선택하는 문제로 돌아가는 것이 타당하다.**

이것은 PDF를 포기하는 변경이 아니다. 원문 p.4 식 (13)의 endpoint frontier와 p.5 식 (14)의 endpoint preservation 최소화에 더 가깝다. 식 (15)의 log barrier와 식 (17)의 CBF는 선택적 구성이다. 이전 리뷰가 선택적 barrier를 첫 구현으로 승격하고, 후속 계약이 N4의 10-batch 최대 KL을 필수 입력으로 만든 것이 불필요한 의존성의 직접 원인이다.

고정 한도 자체가 잘못된 수학이라는 주장은 하지 않는다. 외부 요구가 `D≤b`로 주어졌다면 자연스러운 제약 최적화 문제다. 현재 문제는 **그 요구가 없는데 .9×N4, TV=.10 등의 선택을 방법에 내장한 것**이다. 반대로 한도를 없애도 편집과 보존 사이의 우선순위까지 자동으로 정해지는 것은 아니다.

첫 후보는 **EP-TW-1: edit-quality-preserving target–write refinement**로 구분한다. 이름은 가칭이며 신규성 주장이 아니다. 기준은 별도 N4 경로가 아니라 **자기 batch의 native proposal이 실제로 달성한 canonical 편집 품질**이다. 한 번의 보정, 실제 후보 검사, native 복귀부터 시작한다. 상세 사양은 [v3 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-edit-quality-preserving-tw-design-v3.md), [후보 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-edit-quality-preserving-tw-contract.json)에 둔다.

## 2. 재검토 범위와 실제 실행 상태

- [ICLR_2027.pdf](/mnt/raid5/janghj/ODE-edit/plans/global/ICLR_2027.pdf) 13쪽 전체. SHA256 `9ac95303ddeaabcd57e0b83e3a298d7f24cd0571d89ee830c3194f9a33cc728f`.
- [첫 PDF 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pdf-method-review-ko.md), staged v2, reference 계약, 철회된 TV 예산안, 기존 no-budget 논의안. 기존 논의안의 방향을 채택하되 actual native endpoint의 좌표, 유한 step의 실패, 원격 admission 변경 범위를 보충했다.
- Git `05a9c11e16edee3d43e18e66bc0072c5472fc440`의 Server4 G0 보고와 `bg_tw_reference/{control,gate,correction,native_map,operations}.py`. 현재 local checkout은 `ddc178584ef14efd5d4e1271b3c324e3ebd3e443`이며 사용자 작업을 덮어쓰지 않고 fetch한 source를 읽었다.
- [출처·이전 문서 hash manifest](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-pipeline-reset-2026-09-15/source-manifest.json). Server4 파일은 해당 Git blob의 보존 사본이다. 원격 GPU·tensor를 독립 재실행한 감사가 아니다.

[Server4 G0 사실 보고](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-pipeline-reset-2026-09-15/source/experiment-reports/servers/server4/bg1-c4-ours-first-2026-09-15-v1/g0-factual-report-ko.md)의 관측은 다음과 같다.

| 항목 | 보고된 사실 | 이번 설계에 미치는 영향 |
| --- | --- | --- |
| N4 calibration | B1·B5·B10만 보존, B2·B3·B4·B6·B7·B8·B9 없음 | 기존 `CALIBRATION_MISSING`은 실제 실행 장애다 |
| 새 BG 과학 실험 | 0 edits, 첫 B100 미실행 | 기존 BG 성능을 관측한 것으로 쓰지 않는다 |
| C4 원문·token | 두 전체 shard, 768문서, int64 `[768,257]` 구축 보고 | 재구축 대신 sealed identity와 자산을 재사용한다 |
| Teacher | job 47592, 마지막 관측 PENDING | 완료 여부는 이 보고로 확정할 수 없다 |
| 구현 | CPU fixture 39 PASS; 실제 모델 adapter·persistent runner 미완성/미검증 | 새 방향의 실제 gradient·복원·history 검사가 남는다 |
| 운영 | `G0_BLOCKED / WAITING_USER_RESUME` | 본 설계문이 기존 원격 지시를 자동 변경하지 않는다 |

Reference identity는 `f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0`, member root는 `0c6aa4e2ddb350c61999580fe3efabaae880ec8e17aa208a8f7d85c026a6f1aa`다. Teacher payload 11.7422 GiB는 계획 산술이지 완료 산출물 확인이 아니다. 기존 제출·봉인된 V1 지시와 source lock은 역사적 원문으로 보존한다.

## 3. PDF의 정의를 처음부터 다시 분리한다

| 원문 | 유지할 내용 | 수정·제한할 내용 |
| --- | --- | --- |
| Abstract·§1 p.1 | 실제 write 후 feedback을 연구할 동기 | REFIT4를 full10k·short+full·세 지표 동시 개선으로 기술하지 않는다. 확인된 후보 적용은 W50→W60 신규1000, 상한 `[24,24]`, scale `[.75,1]`; P strict 손실도 있다 |
| 식 (1), p.2 | 요청별 편집 feasibility의 정의 | 요청별 조건과 평균 loss 조건은 다르다. 첫 EP-TW는 평균 NLL+성공 ID 보존만 요구하는 제한된 근사다 |
| 식 (2), p.2 | fixed-original response와 accepted-old desired label의 역할 분리 | Original KL을 old accepted edit 정답 보존으로 대체하지 않는다 |
| 식 (3), p.2 | hook target objective는 proposal 생성에 유용 | 실제 모든 token에 작용하는 weight update의 효과와 같다고 보지 않는다 |
| 식 (4)–(5), p.2 | 고정 writer의 선형 residual map | 이상적 orthogonal projector 식을 실제 native solve로 무검증 대체하지 않는다. 이 quadratic writer 자체는 nonconvex barrier의 증거가 아니다 |
| 식 (6)–(8), p.3 | target/write 교대의 hybrid 표현, 고정 A의 endpoint 등가 | 유한 Adam·reset·write를 엄밀한 ODE 적분 효능으로 부르지 않는다. optimizer state·clock이 필요하다 |
| 식 (9), pp.3–4 | frozen target에서 residual filter의 해석 | `H(W+Δ)-H(W)=ΔK`가 성립하는 matching readout/key 조건에서만 적용한다 |
| 식 (10)–(12), p.4 | path peak와 path family 간 barrier 비교의 정의 | 임의 높이 b를 method의 필수 입력으로 만들 필요가 없다. sampled peak가 낮다고 연속 경로 우회나 최적 barrier 차이가 증명되지 않는다 |
| 식 (13), p.4 | **품질 조건에서 endpoint KL 최소화** | 이번 주 목적의 출발점. 사전 D 한도를 요구하지 않는다 |
| 식 (14), p.5 | endpoint D와 optional peak의 목적 | 첫 버전은 peak 항을 제거한 `ξ=0`에 해당하는 endpoint 문제만 근사한다 |
| 식 (15), p.5 | actual/virtual write를 통한 target feedback | **log barrier는 선택 사항이다.** b가 없는 첫 방법에서 이 penalty를 유지하지 않는다 |
| 식 (16), p.5 | realization gap 진단 | 입력·scoring·reduction이 같은 경우만 비교하며 gap을 landscape barrier로 부르지 않는다 |
| 식 (17), p.5 | feasible 초기 상태·정확한 동역학 등에 의존하는 연속시간 조건 | 유한 후보 검사·relaxed barrier에 invariance 보증을 옮기지 않는다 |
| §6, pp.6–8 | 대표·위험·witness·old의 역할 분리, finite-menu loss bound | S64부터 시작한다. finite-menu bound는 새 gradient 후보·이후 drift의 gradient 보증이 아니다 |
| §7, pp.8–10 | iteration·virtual write feedback·보존 제약의 선행 연구 | projection, barrier 용어, ODE 표현, medoid 각각의 신규성을 주장하지 않는다 |
| §8, pp.10–11 | endpoint·compute·one-shot·scalar·state 비교 | 구현 전 전수 gate가 아니라 필요한 주장에 따른 W0 sequential 비교로 배치한다 |
| 참고문헌, pp.12–13 | 관련 분야 연결 | 원문에 나열됐다는 사실만으로 논문별 정리의 조건이 이 방법에 적용되지는 않는다 |

I2는 N4보다 paraphrase 성공과 strict가 높지만 NS가 낮은 trade-off를 보였다. 따라서 **“step 분할이 비합리적임이 판명되어 ODE만 정당하다”는 논증은 사용할 수 없다.** 현재 동기는 시험한 unguided 분할이 원하는 편집–보존–비용의 결합을 아직 달성하지 못했다는 것이다.

## 4. 한도 제거가 해결하는 것과 남기는 것

### 4.1 N4 최대값의 .9배

이 수치는 독립 baseline의 특정 operating point에 맞춘 비교 규약으로는 정의할 수 있다. 그러나 online 방법의 필수 입력으로 쓰면 현재 요청보다 뒤의 batch 결과·저장 상태를 요구하고, corpus/core/order에 따라 다시 산출해야 한다. 이번에는 10개 checkpoint 중 7개가 없어 admission이 막혔다. 저장되지 않은 중간 weight를 metric/hash만으로 재구성할 수는 없다.

### 4.2 TV를 사전에 정하고 KL로 변환

Pinsker와 Jensen을 사용하면 같은 가중 평균에서 `mean TV ≤ sqrt(mean KL/2)`이므로 `D≤2ρ²`는 평균 TV≤ρ의 충분조건이다. 이는 정당한 관계지만 **어떤 ρ가 이 편집 문제에 적절한지는 결정하지 않는다.** `ρ=.10 → b=.02`는 10% accuracy 손실 허용이라는 뜻도 아니다. 해당 초안의 수학과 한도 선택의 근거를 분리해야 한다.

### 4.3 자동 b·자동 λ도 선호를 자동 발견하지 않는다

예를 들어 `E(w)=(w-1)²`, `D(w)=w²`이면 `[0,1]`의 점들은 서로 다른 Pareto trade-off다. 데이터만으로 유일하게 옳은 점을 정할 수 없다. Adaptive budget에는 update rule·reference·단위·목표가, primal-dual 방법에는 제약 자체가 필요하다. Plateau나 quantile로 b를 바꾸는 것도 새로운 운영 선택이다.

두 loss를 MGDA로 동시에 줄이면 해결된다는 주장도 조심해야 한다. **W0에서 고정 teacher KL은 최소값 0이고 gradient도 0이다.** 두 gradient convex hull의 최소 norm은 0을 고를 수 있어 편집을 시작하지 않는 Pareto stationary 상태가 생긴다. 이는 MGDA의 오류가 아니라 목적의 충돌을 보여주는 예다. [Sener–Koltun, NeurIPS 2018](https://arxiv.org/pdf/1810.04650)

`D_t≤D_{t-1}`를 W0부터 강제하면 D0=0이므로 보호한 입력의 분포를 정확히 유지하는 매우 강한 문제가 된다. 다른 입력에 대한 모든 편집이 불가능하다는 뜻은 아니지만, 한도 없는 부드러운 대체물이 아니다.

따라서 첫 방법의 명시적 우선순위는 **현재 proposal의 편집 품질을 보존한 뒤 그 안에서 D를 줄이는 것**이다. 이를 “hyperparameter-free” 또는 “선호 없는 최적화”라고 부르지 않는다.

## 5. 한도 없는 첫 문제와 실제 native anchor

현재 자기 branch의 entry를 W, native target/write가 실제 FP32로 만든 provisional endpoint를 Vp라 한다. 별도 N4 stream을 실행하지 않는다. 현재 B100의 canonical teacher-forced desired loss를 요청별 token 평균 후 요청 평균한 값을 E, Vp에서의 값을 Ep라 한다. Ap는 Vp에서 desired answer의 모든 token이 기존 strict 판정을 통과한 **요청 ID 집합**이다.

후보 집합에 Vp를 반드시 포함하고 다음 유한 문제를 푼다.

\[
\min_{V\in\mathcal C_t}D_{64}(V)
\quad\text{s.t.}\quad E(V)\le E_p,\qquad
\mathcal A_p\subseteq\mathcal A(V).
\]

여기에 D의 허용량 b, TV ρ, base penalty μ는 없다. Ep는 그 batch에서 이미 계산한 품질 anchor다. 성공 수가 같아도 성공 ID가 바뀌면 실패 처리한다. 평균 NLL을 맞추기 위해 요청을 분모에서 제외하지 않는다.

실수 연산에서 고정 native writer를 `S_t(R)=R A_t`로 쓴다. 실제 source의 context 반복 T를 포함하면

\[
Q=P(KK^\top+M_t)+\lambda I,\quad
A_t=T(Q^{-1}PK)^\top.
\]

Native source는 residual RHS를 포함한 solve를 수행한다. FP32에서 `W+R_p A_t`가 저장된 native Vp와 byte-identical하다는 보장은 없다. 따라서 새 후보 좌표는

\[
V(C)=V_p+C A_t,\qquad C=0\text{은 실제 native endpoint}.
\]

로 둔다. `C=0`은 native bytes를 직접 복원한다. Native target Zp를 중심으로 `Zp+C`를 native anchor/radius 안에 유지한다. Trust 크기의 분자는 근사 `||R_p A||`가 아니라 **실제 `||V_p-W||`**다. 수학적 선형 map과 rounded forward는 별도 검증한다. 이 anchor 보정은 기존 no-budget 논의안을 실행 가능하게 만들기 위한 추가 수정이다.

## 6. Gradient projection은 어떤 보장을 주는가

Vp에서 `gE=∇C E(V(C))|0`, `gD=∇C D64(V(C))|0`를 별도로 구한다. 고정 A에서는 각각 `∇V loss · Aᵀ`다. 다음 문제는 residual 좌표의 Euclidean half-space projection이다.

\[
d=\arg\min_u\tfrac12\|u+g_D\|_F^2
\quad\text{s.t.}\quad\langle g_E,u\rangle_F\le0.
\]

`q=〈gE,gD〉`, `gE≠0`일 때

\[
d=\begin{cases}
-g_D,&q\ge0,\\
-g_D+\dfrac{q}{\|g_E\|_F^2}g_E,&q<0.
\end{cases}
\]

정확한 산술에서는 `〈gE,d〉≤0`, `〈gD,d〉≤0`다. 충돌 시 보존 방향에서 현재 품질을 악화시키는 성분을 뺀다. 같은 식의 dual multiplier `λ*=max(-q,0)/||gE||²`는 **현재 두 gradient로 결정되는 KKT 값**이며 사용자가 정하는 손실 혼합 가중치가 아니다.

이 projection 원리는 알려져 있다. A-GEM은 episodic reference gradient를 이용한 평균 비간섭 제약을 사용했다. 여기서는 current quality가 제약이고 fixed-W0 KL이 목적이며 executable residual 좌표에서 계산한다. 이 차이가 projection 정리 자체를 새롭게 만들지는 않는다. [Chaudhry et al., ICLR 2019](https://arxiv.org/pdf/1812.00420)

또한 이 방향은 weight Frobenius 공간이나 native metric에서의 최적 projection이 아니다. `d A_t`가 실행 방향이므로 metric 선택의 효과가 남는다. 첫 버전은 복잡한 metric solve를 추가하지 않고 그 선택을 명시한다. `gE=0`이면 선형 제약은 비어 있지만 실제 nonlinear 품질 제약까지 없어진 것이 아니다. 수치적으로 작은 norm은 별도 기술 기준과 skip 사유로 처리한다.

### 6.1 가장 중요한 한계: 접선 방향은 실제 feasible step이 아닐 수 있다

국소 예를 직접 계산한다. `E(x,y)=1+x+y²`, `D(x,y)=((x-1)²+(y+1)²)/2`, native preview `(0,0)`이다. 이때 `gE=(1,0)`, `gD=(-1,1)`이므로 projection은 `d=(0,-1)`이다.

모든 양의 α에 대해 직선 후보 `(0,-α)`는 `E=1+α²>Ep=1`이므로 **backtrack을 아무리 줄여도 정확한 품질 검사를 통과하지 못한다.** 반면 곡선 `(-α²,-α)`는 E=1을 유지하고 `0<α≤.5`에서 D를 줄인다. 따라서 raw 복귀가 많다는 사실만으로 실행 가능한 개선 방향이 없다고 판단하면 안 된다.

이는 nonlinear 제약의 곡률을 선형화가 놓치는 문제다. 고전적 Maratos effect와 second-order correction 논의가 참고가 되지만, 이 예 자체를 완전한 SQP 수렴 분석으로 주장하지 않는다. [Leyffer–Mahajan, §4.4](https://www.mcs.anl.gov/uploads/cels/papers/P1767.pdf)

후속 확장이 필요하다면 NLL tolerance를 임의로 넓히기보다 trial의 품질 위반 `v=[E-Ep]+`를 줄이는 **quality restoration**을 우선 검토한다. Trial gradient gE′가 유효할 때 `-v gE′/||gE′||²`는 선형화된 위반을 상쇄하는 후보 보정이다. 이것도 실제 검사·clamp·trust가 필요하고, nonlinear feasibility나 수렴을 보장하지 않는다. 첫 SEQ1000에는 넣지 않고 실패 사유가 확인된 뒤 별도 버전으로 비교한다.

### 6.2 Target ball과 finite selection

정규화한 d로 `Zp+αd`를 native per-request ball에 투영하고 C를 구한다. 필요하면 C를 줄여 `||C A||≤ζ||Vp-W||`, 최초 ζ=.25를 유지한다. ζ는 solver trust parameter이며 허용 KL량이 아니다. Ball 투영 후에는 1차 두 부등식조차 유지된다고 가정하지 않고 다시 기록한다.

후보는 `Vp`, `Vp+C A`, `Vp+.5 C A`, `Vp+.25 C A`다. **Native write 전체를 .5/.25로 줄이는 기존 메뉴와 다르다.** Raw와 실제 quality screen을 통과한 후보 중 D가 가장 작은 것을 고른다. KL 개선이 불확실하거나 후보가 모두 탈락하면 native를 commit한다. Native 자체가 비유한 값·실행 오류이면 기술 실패로 처리한다.

## 7. 이 방법이 실제로 보장하지 않는 것

1. 유한 후보의 정확한 측정에서는 `D(selected)≤D(own raw preview)`이고 선언한 current 평균·성공 ID 조건을 만족한다. 모든 후보 중 전역 최적이라는 보장은 없다.
2. `D(Wt)≤D(Wt-1)`나 `D(Wt)≤D(independent N4_t)`는 따라오지 않는다. B2부터 두 policy의 weight entry와 새 target trajectory가 다를 수 있다. Key/history까지 반드시 달라진다는 뜻은 아니다. 단일 L4 down-projection과 고정 입력에서는 upstream key·동일 등록 history가 같게 유지될 수 있으므로 실제 규약·hash로 구분한다.
3. 예를 들어 W0=0, D=w², 첫 raw=2에서 ours=1.5를 선택하고, 다음 native map을 `T2(w)=4-w`로 두면 독립 native는 2→2, ours는 1.5→raw2.5→선택2.25가 될 수 있다. 매번 own raw보다 D가 작아도 terminal D는 ours 5.0625, native 4다. 적절한 current loss를 두면 두 quality screen도 만족한다. 이는 논리 반례이지 LLM 예측이 아니다.
4. 평균 NLL과 성공 ID 보존은 모든 요청의 NLL, margin, 공식 paraphrase 성공을 보장하지 않는다. 성공 ID가 유지되어도 confidence가 낮아질 수 있다.
5. S64에서의 fixed-prefix KL은 원 모델의 정답 지식·자유 생성·전체 C4·old accepted knowledge를 보증하지 않는다. Dev128, Report256, NLL/NS, active-old를 따로 평가한다.

### 7.1 Old retention에 남아 있는 임의 한도도 자동 승계하지 않는다

기존 `old loss at acceptance + .1 nats`를 generic b 제거 후에도 기본값으로 남기면 같은 근거 문제가 남는다. 첫 방법은 old ledger를 저장·평가만 한다. Old protection을 구현했다고 쓰지 않는다.

후속 old 버전은 현재 entry에서 여전히 성공하는 active accepted ID를 지키는 task-level 조건 등을 검토한다. 이 경우 old 기준은 raw preview가 아니라 entry여야 이미 raw가 잊은 지식을 보호할 수 있다. 그러나 `현재 품질≥raw`와 `old 성공≥entry`는 동시에 불가능할 수 있다. Native fallback도 더는 old-feasible하지 않을 수 있다. 공동 제약 불가능 시 defer/부분 성공/우선순위와 원분모를 **새 정책으로 먼저 정의해야 하며**, old 보호라고 쓰면서 raw로 조용히 복귀하지 않는다.

## 8. Barrier·ODE 주장은 어떻게 정리할 것인가

**첫 방법의 이름과 핵심 주장에서는 log barrier·CBF invariance·ODE 필요성을 제외한다.** 고정 한도를 없앤 뒤 `-log(1-D/b)`를 유지할 수 없고, 대신 `-log(Ep-E)`를 쓰면 시작점 Vp의 slack이 0이라 정의되지 않는다. 임의 ε를 더하면 새로운 완화 선택이다.

PDF의 landscape barrier는 다른 개념이다. 같은 edit quality·경로 family·계산 제약에서 `B_ray(ε)-B_controlled(ε)` 또는 endpoint를 분리한 segment peak excess를 사후 진단할 수 있다. 그때도 몇 개 sampled segment의 peak는 해당 연속 최대값의 하한이다. 탐색한 최선 경로가 family 전체의 최적 경로라는 보장도 없다. 실제 controller는 endpoint만 고르므로 경로 peak 최소화 방법이라고 부르지 않는다.

CBF는 외부에서 정당한 안전 집합과 초기 feasibility가 주어지는 후속 적용에서 다시 쓸 수 있다. 연속 dynamics·regularity·feasible control 조건 등이 필요하다. [Ames et al.](https://arxiv.org/abs/1903.11199)

ODE는 여전히 target/write의 hybrid 상태를 표현할 수 있다. 그러나 작은 유한 step 또는 한번의 projected correction만으로 ODE가 필요하다는 증거는 생기지 않는다. 고정 A에서는 모든 다단계 endpoint가 `Wentry+B_eff A`로 표현되므로, multistep 이득을 주장하려면 같은 executable endpoint family의 강한 one-shot 대조가 필요하다. Provisional native preview를 실제로 거쳤다면 그 계산 경로에서 발생한 peak를 숨겨 barrier bypass라고 쓰지도 않는다.

Target-through-write feedback은 MetaKE의 virtual post-edit target optimization과 특히 가깝다. 검증할 차이는 실제 native endpoint를 anchor로 한 유한 후보 검사, fixed-original response, 명시적 current quality 우선순위, 비용 및 lifelong 결과의 조합이다. [MetaKE v3, §4.3](https://arxiv.org/html/2603.12677v3)

내부 2026-09-10 [EP-Free 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-10-edit-progress-preserving-damage-removal-design.md)와도 edit progress를 지키면서 위험을 줄인다는 계보가 이어진다. 다만 covariance/native-metric 위험과 fixed-W0 full-vocabulary KL, rate 조건과 실제 finite quality screen은 구분한다.

### 8.1 사전 D 한도 없이 남길 수 있는 연속시간 해석

ODE를 연구 설명에서 전부 삭제할 필요는 없다. 현재 episode의 고정 Vp·A·Ep를 사용해

\[
\mathcal F_t=\{C:E(V_p+CA)\le E_p,\quad
\|Z_{p,i}+C_i-a_i\|\le r_i\ \forall i\}
\]

라는 **편집 품질·target-ball feasible set**을 정의하고, 그 안에서

\[
\dot C\in\operatorname*{argmin}_{u\in T_{\mathcal F_t}(C)}
\tfrac12\|u+\nabla_C D_{64}(V_p+CA)\|^2
\]

인 projected gradient dynamics를 이상적 해석으로 둘 수 있다. 여기서 T는 실제 feasible set의 tangent cone이다. **보존 loss D에는 임의 한도가 없다.** Generic KL을 줄이는 목적과 task feasibility를 구분하는 표현이다.

충분한 regularity와 적절한 해 개념 아래에서 이러한 projected dynamics의 존재·viability를 다룰 수 있다. 특히 닫힌 convex tangent cone에 대한 정확한 Euclidean projection이고 미분 가능한 실제 경로라면 projection의 직교 관계로 `dD/ds=-||Cdot||²≤0`가 성립한다. 이는 고전적 projected-gradient 이론의 특수한 해석이며 새 ODE 정리가 아니다. 비정규 집합·불연속 field에서는 해의 정의와 존재/유일성 조건을 더 따져야 한다. [Hauswirth–Bolognani–Dörfler, SIAM J. Control Optim. 2021, §§4·8](https://arxiv.org/pdf/1809.04831)

이 이상 흐름은 §6.1의 예에서 변화하는 접선을 따라 `x+y²=0`인 곡선으로 움직일 수 있다. 처음 구한 접선으로만 유한 Euler step을 내디디면 이 곡률을 놓친다. 따라서 후속 predictor–corrector/quality restoration의 연구 질문은 **“분할 횟수를 늘리면 좋은가”보다 “변하는 품질 경계를 추적하면 실제 허용 후보를 더 잘 찾는가”**로 구체화할 수 있다.

그러나 첫 EP-TW-1이 이 흐름을 구현했다는 주장은 하지 않는다. 실제로는 한 지점의 평균 E half-space를 사용하고, ball을 사후 투영하며, 작은 유한 메뉴를 평가한다. 위 F에는 discrete strict-ID 조건을 포함하지 않았고, 그 조건은 실제 screen에서 추가 확인한다. Neural feasible set의 regularity·유일성도 검증하지 않았다. 이 설명으로 native jump를 포함한 전체 lifelong D 단조 감소를 주장할 수 없다.

PDF Proposition 1의 rate-versus-route 구분도 유지한다. 같은 autonomous field에 양의 scalar 속도만 곱하면 연속 궤적은 시간 재매개화 관계다. 여기서 가능한 경로 변화는 품질 조건에 따른 **field 방향 변경**에서 오며, 작은 step 자체에서 오지 않는다. Native proposal/target reset을 포함하면 전체는 jump와 refinement가 있는 hybrid 과정이다.

실험에서 이 연결을 주장하려면 같은 quality 조건·endpoint family·실제 비용 아래 one-shot과 refinement를 비교해야 한다. 낮은 Euler truncation error 또는 sampled peak만으로 편집 효능·ODE 필요성을 주장하지 않는다. Log barrier가 없는 현 버전에는 “barrier-guided” 대신 “quality-constrained refinement”가 더 정확한 명칭이다.

## 9. 데이터부터 실행까지 다시 정리한 파이프라인

| 단계 | 새 설계 | 재사용·상태 |
| --- | --- | --- |
| 데이터 | C4-WebRef-v2 문서·token·분할 고정 | 구축 완료 보고된 768 자료 재사용; 표본을 재선정하지 않는다 |
| Teacher | fixed original W0의 full-vocab logp0 | 기존 teacher job의 봉인 산출물·완료 여부를 실행 재개 시 확인; 무조건 재생성하지 않는다 |
| 목적 | current quality 조건 아래 D64 최소화 | N4 calibration, TV→KL 설정, μbase/τ 제거 |
| Proposal | 자기 branch의 native target/write 1회 | N4 독립 chain이나 미래 endpoint 없음 |
| 보정 | 두 gradient 분리, half-space projection 1회 | 기존 mixed barrier gradient 함수와 다른 policy |
| 검사 | correction-only 후보, current quality screen, min D | 기존 KL 한도 screen와 whole-write 축소 메뉴 교체 |
| Commit | correction이 유효하면 선택, 아니면 native | preservation 한도에 따른 parent rejection 없음; native 실패는 기록 |
| History·ledger | B100 finalizer 1회, accepted desired label 별도 기록 | inner append0; rollback·resume identity 검사 필요 |
| 개발 실험 | W0 B100×10, ours 신규 1 chain | 단일 batch는 기술 검사 용도; 성능 판정은1000 sequential |
| 비교·확장 | 기존 baseline 자산의 검증 가능한 범위 재사용 | 다섯 baseline+REFIT4는 비교 목록이지 자동 추가 제출 목록이 아니다 |

### 9.1 JSON의 b만 지우면 실행 의존성이 없어지는 것은 아니다

확인한 실제 source의 [control.py](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-pipeline-reset-2026-09-15/source/project/run_scripts/bg_tw_reference/control.py)는 dispatch calibration 필드를 요구한다. [gate.py](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-pipeline-reset-2026-09-15/source/project/run_scripts/bg_tw_reference/gate.py)는 B1–B10 전체를 검증해 `calibrated_budget`을 먼저 호출한다. [correction.py](/mnt/raid5/janghj/ODE-edit/local/reviews/bg-tw-pipeline-reset-2026-09-15/source/project/run_scripts/bg_tw_reference/correction.py)는 barrier slope, whole-write backtracking, `D≤ceiling` 후 min E, 전부 실패 시 PARENT를 구현한다.

따라서 후속 구현은 **새 policy ID와 새 admission 규약**이어야 한다. `b=∞`, partial maximum, calibration PASS 위조로 기존 gate를 통과시키지 않는다. V1 dispatch·teacher source lock은 보존하고, 새 instruction을 실제 전달할 때 새 version/source manifest에 연결한다. 이번에는 원격 전달·제출·모니터링 재개를 하지 않았다.

Acquire/builder/tokenizer/teacher/native-map 자산은 재사용 가능하다. 별도로 필요한 것은 실제 모델 adapter, gE/gD 분리, raw-anchor 복원, 새 selector, 새 admission, persistent transaction이다. 기존 CPU 39 PASS를 새 방법의 실제 LLM gradient·resume 검증 완료로 계산하지 않는다.

## 10. 단계적 실험과 판정

첫 개발 실험은 W0, B100×10, 동일 순서1000요청의 EP-TW-1 한 경로다. 모든 batch에서 own native preview와 selected endpoint의 E/D/strict-ID/비용을 함께 기록한다. 이는 동일 entry에서 보정의 직접 효과를 보여주지만 독립 native lifelong 대조를 대체하지는 않는다.

Native 대비 matched sequential 수치는 identity가 맞는 기존 자산에서 가능한 endpoint만 재사용한다. 없는 B2 등의 C4 값을 보간하지 않는다. Terminal W10이 있으면 terminal 비교를 하고, 없는 시계열은 미측정으로 둔다. 과학적 비교를 위해 나중에 별도 baseline 실행이 필요할 수 있어도 그것을 방법의 시작 조건으로 만들지 않는다.

후속은 원인별로 하나씩 추가한다.

- 유효 correction이 남음: unprojected `-gD`+동일 quality screen과 비교해 projection의 추가 효용을 확인한다.
- 방향 기여를 주장함: 동일 current-quality 조건에서 scalar native update 후보와 비교한다. 그 비교도 W0 B100×10이다.
- Tangent correction이 품질 screen에서 반복 탈락함: bounded quality restoration을 별도 버전으로 비교한다. “feasible 개선 불가능”으로 결론 내리지 않는다.
- Current 개선 뒤 old active 손상이 남음: old retention 정책의 불가능 상태·우선순위를 정의한 뒤 추가한다.
- 그 뒤에만 target refresh·두 committed stage·native metric·core 확대를 검토한다. 여러 모듈을 동시에 바꾸지 않는다.

핵심 평가는 원 requested 분모의 R/P/N·canonical/P strict·NLL tail, at-write→W10, first500 W5→W10, active/superseded old, S64/Dev128/Report256 KL 및 자연 token NLL, wall·teacher·probe 비용이다. W0 시작이므로 기존5000 성능표를 그대로 쓰지 않는다. Request가 cluster이므로 paraphrase/neighborhood를 독립 표본처럼 bootstrap하지 않는다. 한 order의 paired uncertainty는 order 반복을 대체하지 않는다.

Audit128/MMLU68, 전체 E01, landscape sweep는 구현 선행조건이 아니다. 독립 Report256은 policy lock 후 확인하며 이를 보고 tuning하면 개발 자료로 역할을 바꿔야 한다. 최종 lifespan 주장은 W0부터 고정 정책으로 full10k를 실행하고 사용자 지정 다섯 baseline과 비교할 때 판단한다.

현재 허용되는 주장은 **“native 품질을 보존하려는 실제 출력 피드백의 저비용 구현을 시험한다”**다. 편집–보존 효능, old 보존, barrier bypass, ODE 필요성은 아직 결과가 없다.

## 11. 이번 수정의 검증 범위

[수학·계약 점검 기록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-ep-tw-design-checks.json)은 projection 부호·KKT, 유한-step 반례, local/trajectory 비교 반례, reference 데이터 규약 불변, 새 계약의 N4 비의존성과 원분모를 확인한다. 이는 CPU에서의 산술·문서 일관성 점검이며 실제 모델 성능이나 numerical adapter의 검증이 아니다.

기존 V1 실행 계약 및 전달문은 원문 그대로 남긴다. Reference 계약은 corpus identity를 유지하면서 method 연결·진행 상태·실험 순서를 수정했다. 기존 PDF 리뷰와 staged v2에는 현행 설계로 가는 표지를 추가하여 이전 N4 한도를 최신 기본값으로 읽지 않게 했다.

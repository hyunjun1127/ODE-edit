# REFIT4 후속 확정 설계 — target 분할 최적화·반복 fitting의 B100×10 비교

작성일: 2026-09-14. 상태: **설계 확정, 신규 모델 실행·서버 지시 전송·GPU 제출은 수행하지 않음.**

최신 사용자 요구를 반영해 방법 선택의 주 실험을 **공통 W50/M50에서 같은 신규 1,000개 요청을 처리하는 정책별 B100×10 순차 경로**로 확정한다. 이전 G1/G1-R의 단일 batch endpoint는 구현 확인과 국소 해석에만 사용한다. 단일 batch 성능을 보고 후보를 선별하거나 본 실험 진입을 막지 않는다. Audit128/MMLU68도 선행조건이 아니다.

이 문서가 이전 [G1 중심 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-mechanism-next-gate-design.md)와 사용자 보충안의 **실행 순서·주 비교 범위**를 대체한다. 기존 수치·원본 source 및 좁은 claim 허용 원칙은 유지한다. 추가 layer, ODE 명칭, 모든 지표 비악화, CI 하한 양수, online 비용 1.5배 이하를 결합한 AND gate를 만들지 않는다.

## 1. 첨부 보충안의 검토 결과

검토 원문: [분할 target 최적화와 write-refresh 보충 설계](/mnt/raid5/janghj/.codex/attachments/0a5a1bd4-a5a6-4b60-8b55-f340409fcf0c/pasted-text.txt), SHA256 `b30d54de059191ab6f45904c5817f1ef34b883d4333598b9d9eac4aa4a4d5514`.

수치와 수식의 핵심은 타당하다. 다만 **I2에 대응하는 2-write frozen control**을 명시적으로 포함하고, target 좌표·loss·optimizer 상태를 구현 가능한 형태로 고정하며, 단일 batch 선택을 1,000요청 순차 비교로 바꾼다.

### 확인된 관측

- 완료 CSV를 직접 재집계하면 REFIT4 second-fit Adam 분포는 **0회853개, 21회1개, 24회146개**다. Positive147개 평균은23.97959회다. 첫 fitting까지 합하면146개는48회, 1개는45회, 853개는24회다. 평균27.525회/request와 per-request 최대48회를 구분한다.
- 신규 R TF strict는 N4 993→REFIT4 998/1000으로 증가한다. 신규 P strict는1423→1405/2000으로 감소한다. Strict 전체가 일률적으로 악화됐다고 쓰지 않는다.
- 신규 active R은 양쪽994/994이며 old active에서는 N4 대비 R−1/P−6이 남는다. ALL과 ACTIVE_TARGET을 함께 보고한다.
- 0-Adam target이 곧 residual 0 또는 실제 write 0은 아니다. 모든 key를 포함한 공동 solve와 canonical/target-init readout 차이를 유지해서 분석한다.

근거는 검토 snapshot `7e67befa1c73b67768d16ab98029468c945f2ac0`의 [완료 CSV·소스 manifest](/mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source-manifest.json), [six-arm 상세 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-14-lowcost-seq10-review-ko.md), [이번 재계산 기록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-design-checks.json)이다. 원시 신경망 tensor의 새 재구성이나 GPU 재평가는 수행하지 않았다.

### 수식에서 유지할 해석

고정 target의 repeated fitting은 residual 방향에 matrix-polynomial filter를 적용할 수 있다. 따라서 비평행 update를 관측하더라도 fresh target의 효과라고 곧바로 부를 수 없다. 2회·4회 write 각각에 frozen control이 필요하다.

첨부의 Euler scalar 예시도 맞다. 고정 target에 h=1/J로 J번 접근하면 J=2에서75%, J=4에서68.359375%, 극한에서63.212056%를 실현한다. 이 예시는 horizon이 다른 native one-shot과 under-write가 섞일 수 있음을 보일 뿐이다. 이번 I2/I4는 **고정 Adam budget을 분할한 target–write 교대 최적화**이며 엄밀한 Euler 수렴이나 ODE 고유 기여를 주장하지 않는다.

## 2. 본 실험의 여섯 정책

공통 초기 상태는 기존 six-arm의 **L4 W50/M50 및 동일 context/RNG**다. 데이터·순서는 fixed10k의 B51–B60, ordinal [5000,6000), 총 unique1000이다. 각 정책은 자기 모델·history를 이어가며 다음 batch의 target을 새로 계산한다. 모든 정책의 최종 full-seen 분모는 old5000+new1000=6000이다.

| 정책 ID | Target Adam 최대 budget/request | Write 계수 | Target 처리 | 역할 |
|---|---|---|---|---|
| N4 | [24] | [1] | batch entry의 native target | 원 기준; N24와 같은 reference |
| REFIT4 | [24,24] | [.75,1] | 각 단계 fresh reset | 관측된 개선·비용 reference |
| FROZEN2 | [24,0] | [.75,1] | 처음24-step target을 두 write에서 고정 | REFIT4와 I2의 2-write 대조 |
| I2 | [12,12] | [.75,1] | target/Adam 상태를 유지하며 중간 write | 최대24회를 두 구간으로 분할 |
| FROZEN4 | [24,0,0,0] | [.75,.75,.75,1] | 처음24-step target을 네 write에서 고정 | I4의 4-write 대조 |
| I4 | [6,6,6,6] | [.75,.75,.75,1] | target/Adam 상태를 유지하며 중간 write | 최대24회를 네 구간으로 분할 |

FROZEN2는 이전 G1의 **FZ4(같은 L4에서 두 번 fitting)**와 같은 정책 의미다. FROZEN4는 보충안의 F4다. Layer 번호와 fitting 횟수의 혼동을 피하려고 새 실행 ID를 구분한다. FROZEN 계열 표의0은 추가 target optimization 없음이며 추가 residual solve가 없다는 뜻이 아니다.

주 contrast는 **REFIT4−FROZEN2, I2−FROZEN2, I4−FROZEN4**다. 각 정책−N4로 순차 실용 이득을 확인하고 I4−I2는 refresh 빈도에 따른 전체 정책 차이로 읽는다. I4−I2만으로 빈도의 순수 인과효과를 주장하지 않는다.

기존 N4/REFIT4의1,000요청 경로를 같은 entry/source/precision/evaluator 조건에서 재사용하면 **신규4경로×10=40 batch executions**다. 기존 두 경로를 포함한 과학적 비교는6경로/60 batches다. 재사용이 비교 성립 조건에 맞지 않으면 같은 실행 환경에서 두 reference를 포함해60 batches를 수행한다. 기존 S75/S875/RES8/FULL8 결과는 참고 자료로 유지하며 다시 제출하지 않는다.

## 3. Target-stepper의 정확한 상태와 loss

I2/I4는 기존 fresh `compute_z`를 여러 번 호출한 구현이 아니다. 요청별 state를 **한 B100 내부에서만** 이어가는 별도 stepper를 사용한다. 다음 B100의 새 요청은 그 정책의 새로운 batch entry에서 초기화한다. 서로 다른 요청 사이와 batch 사이에 Adam moments를 넘기지 않는다.

요청 i의 batch entry target-init anchor를 a0_i라 하고, optimizer의 leaf parameter는 **entry 기준 offset u_i**로 둔다. 초기값은0이고 absolute target은 Z_i=a0_i+u_i다. Chunk j의 변경된 모델에서 같은 target-input/lookup으로 얻은 unhooked anchor를 aj_i라 하면 additive hook은

    delta_hook_i = u_i + (a0_i - aj_i)

로 정의한다. 이 표현은 absolute target을 유지하되 optimizer parameter를 매번 빼고 더하지 않는 방식이다. Chunk0에서 aj=a0이면 원 native delta=u가 된다. 큰 absolute Z를 매 step 갱신했다가 a0를 빼는 다른 FP32 계산 경로로 native parity를 가정하지 않는다.

재기준화할 때 u_i 자체와 Adam moments를 바꾸지 않는다. a0_i와 aj_i는 detach하고, 과거 weight write나 solver를 통한 meta-gradient를 계산하지 않는다. 같은 additive hook을 native와 동일한 rewrite/KL contexts와 lookup 위치에 적용한다. 첫 clean target-init anchor와 canonical writer readout Y를 임의로 동일시하지 않는다.

### 고정할 objective

I2/I4의 batch-entry 기준은 다음처럼 고정한다.

- KL teacher: 해당 branch의 B100 entry에서 native와 동일한 KL prompt로 얻은 분포.
- Regularizer: **v_weight_decay × ||u|| / ||a0||²**. Native 식은 numerator가 제곱 norm이 아니다. AdamW 또는 일반적인 squared L2로 바꾸지 않는다.
- Clamp: **||u|| ≤ clamp_norm_factor × ||a0||**. Projection 뒤 moments는 native와 동일하게 유지한다.
- KL 방향: 원 source의 `kl_div(entry_log_probs, current_log_probs, log_target=True)`와 reduction을 유지한다. 수학적으로 current→entry KL이며 인수 순서를 조용히 뒤집지 않는다. 이는 [실행 버전에 대응하는 PyTorch 2.9 정의](https://docs.pytorch.org/docs/2.9/generated/torch.nn.KLDivLoss.html)로 확인했다.
- Rewrite NLL: source의 target tokenization·context averaging·loss layer를 유지한다. 일반적인 최종 logits cross-entropy로 임의 교체하지 않는다.
- Adam: 동일 learning rate/betas/epsilon, m/v와 실제 update counter를 chunk 사이에 유지한다. Counter는 실제 Adam update 때만 증가한다.

Adam의 moments와 bias-correction counter가 optimizer 상태라는 점은 [원 논문 Algorithm1](https://arxiv.org/pdf/1412.6980)에 따른다. Local target loss·regularizer·KL 방향은 [실제 compute_z source](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/compute_z.py:83)를 따른다.

REFIT4 reference는 기존대로 partial 모델에서 delta/Adam/teacher/clamp 기준을 **fresh reset**한다. I2/I4의 carry 규칙을 REFIT4에 소급 적용하지 않는다. I2/I4와 REFIT4의 차이는 예산·target/optimizer reference가 함께 다른 정책 차이다.

### Stopping convention

Chunk마다 초기 loss 평가→필요 시 Adam update→업데이트 뒤 loss 재평가를 수행한다. Source와 같은 total loss<0.05에서 해당 chunk의 요청 최적화를 종료한다. 다음 write로 모델이 바뀌면 다음 chunk에서 같은 요청을 다시 평가한다. 0-step 요청을 영구 제거하지 않는다.

해당 chunk에서 안 쓴 quota는 다음 chunk나 다른 요청으로 이월하지 않는다. Optimizer step 수와 loss evaluation 수를 따로 센다. Early stop이 없으면 I2는12+12 updates와13+13 loss evaluations, I4는6×4 updates와7×4 loss evaluations다. 총24 updates가 같아도 실제 계산량은 같지 않다.

## 4. B100 동기화와 writer

각 chunk에서 **100개 요청 모두 같은 W_j에서 target chunk를 수행한 뒤** 하나의 batch write를 한다. 요청 하나의 target을 계산할 때마다 weight를 쓰는 B1×100 정책으로 바꾸지 않는다. Target 호출 동안 model weight는 고정이고, write 뒤 다음 chunk에서 변경된 full model의 loss/gradient를 계산한다.

각 write는 현재 readout 기준 residual을 쓴다.

    R_j = Z_j - Y(W_j)
    D_j = native_writer(R_j; K, P, M_entry, L2=1)
    W_(j+1) = materialize_native_style(W_j, D_j, gamma_j)

FROZEN2/FROZEN4는 각 **자기 branch의 현재 batch entry**에서 구한 최초 native Z를 그 batch 안에서만 고정한다. 이후 batch의 Z를 N4 경로에서 빌리지 않는다. 최초 residual 재사용, 누적 endpoint를 incremental update처럼 더하기, alpha=.75를 매번 W50 기준으로 적용하는 것은 모두 다른 실험이다.

Materialization은 기존 six-arm 정의를 따른다. 먼저 현재 W_j에서 native full candidate를 FP32로 만들고, gamma=1이면 그 endpoint를 exact copy한다. gamma=.75이면 저장 FP32 candidate−entry 차이에 .75를 곱해 entry에 더한다. 단순 W_j+gamma×이상적인 solve delta와 FP32 연산 순서가 같다고 가정하지 않는다. Requested solve update와 실제 subwrite를 구분해 기록한다.

한 B100의 모든 inner write 사이 M4 append는0회다. 최종 endpoint에서 native finalizer를 정확히 한 번 호출해 M4를 한 번 append한다. 다음 batch는 그 endpoint와 M4/context/RNG에서 시작한다. 모든 다른 layer는 고정한다. M8 재구성은 필요 없다.

## 5. Frozen 반복 writer 수식과 cache 선택

W가 d_out×d_in, K/Kc가 d_in×100일 때 writer matrix를 G, residual-space response를 H로 구분한다.

    G = P(K K^T + M_entry) + I
    B = solve(G, P K)^T
    D = R B
    H = B Kc

고정 teacher-forced inputs, eval 모드, L4 down_proj만 변경, history 불변이면 L4의 K 및 canonical Kc는 정확산술에서 고정이다. Local block readout은 affine이므로 frozen target에서

    R_(j+1) = R_j (I - gamma_j H)

가 된다. 최종 logits나 loss의 선형성을 뜻하지 않는다. Context 평균 K와 canonical Kc가 달라 H의 대칭성·spectrum[0,1]·단조 수축을 보장할 수 없다. 첨부의 같은 수식을 소형 합성 행렬에서 검산했으며, 그것을 모델 FP32 재현 증거로 쓰지 않는다.

첫 과학 비교에서는 native의 `solve(G, (P K) R^T)` 경로와 FP32 순서를 유지한다. P/K/PK/G의 같은 batch 재사용은 identity를 확인하고 적용·타이밍을 기록한다. **RHS map B 선계산, factorization 재사용, Kc로 forward 대체는 첫 비교에 추가하지 않는다.** 연산 재배열이 수치와 비용 비교에 섞이는 것을 줄이고, 유망 정책 선택 후 별도 구현 최적화로 검토한다. G가 대칭 양의 정부호라고 가정해 Cholesky를 적용하지 않는다. Dense inverse는 필요 없다.

## 6. 단일 batch 검사의 역할과 source 연결

단일 batch는 성능 선별 gate가 아니다. Native 연결과 chunk 의미를 확인하는 bounded 기술 검사로 사용한다.

1. I1=[24]·gamma=[1]에서 원 N4의 target/loss/actual endpoint/history와 비교한다. Native source는 [NativeSingletonFitter](/mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source/project/run_scripts/low_cost_write_donor_pilot/fitting.py)로 유지하고 새 stepper가 덮어쓰지 않는다.
2. Weight를 고정한 pause/resume target continuation이 같은 총 Adam update의 단일 호출과 이어지는지 확인한다. 이는 별도 과학 arm이 아니라 optimizer carry 구현 검사다.
3. FROZEN2 첫 .75 materialization이 기존 S75와 연결되는지, 첫 frozen residual이 Z1−현재Y인지 확인한다. History append와 branch 격리도 검사한다.
4. 새 정책들은 구현이 유효하면 초기 P/N 성적과 관계없이 예정된1,000요청을 완료한다. 수치 오류/NaN/정의 불일치는 typed failure로 기록하고 해당 비교만 복원한다. 조용한 native fallback이나 성능에 따른 skip은 하지 않는다.

구현 범위는 기존 `project/run_scripts/low_cost_write_donor_pilot/` 안의 별도 target-stepper, sequential policy adapter, logger/evaluator 확장이다. Fresh native fitter는 reference로 유지한다. 과거 cached target에 moments/teacher가 기록되지 않았다면 I2/I4의 continuation state가 있다고 가정하지 않는다.

기존 N4/REFIT4 재사용은 전체 comparison capsule에 결속한다. E01의 비정확 replay를 현재 reference의 noise floor로 차감하지 않는다. 재사용 비교가 성립하지 않으면 같은 source/host/precision에서 해당 reference 경로를 새로 만든다. 이는 Audit 점수 통과와 별개의 기술적 비교 조건이다.

## 7. 1,000요청에서 평가할 것

### 매 batch와 중간 checkpoint

- 각 batch 직후 Current R100/P200/N1000, R/P TF strict, 두 paraphrase request-strict, true/new NLL·margin을 저장한다.
- 기존 고정 Historical128의 R/P/N과 active 구분을 매 batch 평가한다. 현재100은 batch마다 바뀌므로 이 곡선을 고정 cohort retention으로 읽지 않는다.
- B55에서 suffix500을 재평가하고 B60에서도 **동일 첫 suffix500**의 변화를 산출한다.
- B51/B55/B60 checkpoint와 매 batch commit→다음 entry identity를 저장한다. 진단을 위해 다른 branch의 미래 state를 주입하지 않는다.

### B60 주표

| 모집단 | 주 보고 |
|---|---|
| 신규1000 | R/P/N 성공 수·분모·N4 및 matched control 대비 차이, R/P strict, request-strict, NLL·margin 분포 |
| 신규1000 유지 | 각 요청의 자기 at-write→W60 lost/gained와 NLL 변화; B60의 미래 노출0을 분리 |
| 기존5000 | 전체 및 ACTIVE_TARGET/SUPERSEDED/UNKNOWN의 최종 R/P/N과 paired lost/gained |
| 전체6000 | 총점 및 old/new 분해의 합계 일치; 신규 품질의 대체 지표로 쓰지 않음 |
| 고정 first-suffix500 | W55→W60 변화를 같은 원분모·공통성공/공통실패 집합과 함께 보고 |
| 일반능력 개발 패널 | Wiki128/MMLUdev32를 최소 terminal에서 동일 reference와 비교; method 선택의 단독 gate로 쓰지 않음 |

Old active는 six-arm의 raw(subject,relation)의 마지막 관측 event와 target_new 문자열이 같은 요청으로 정의하며 same-target 재발행도 포함한다. 과거 entry 평가가 없는 범위에서 최종 cross-policy net 차이를 직접 W50→W60 forgetting으로 부르지 않는다. Known overwrite의 완전한 semantic 판별이라고도 쓰지 않는다.

NLL은 평균·중앙값과 **신규/과거 각각의 paired desired-target NLL 악화 p95/p99**를 남긴다. PS·NS의 두 후보 선호와 strict를 구분한다. Marginal true/new NLL을 별도로 보아 competing target만 약화해서 성공률이 오른 경우를 구분한다.

1000요청은 누적 유지와 cohort 차이를 읽기 위한 기본 범위다. 한 fixed order의10 batch는 독립10회 반복이 아니다. CI를 산출하면 request cluster와 batch별 차이를 함께 제시하되 통계적 유의성이나 다른 order 일반화를 자동 보장하지 않는다. PS2개·N10개를 독립 요청으로 늘려 표본 수를 부풀리지 않는다.

Audit128/MMLU68/FutureN은 이번 실행에서 조회하지 않는다. 정책 선택에 공식 P/N·general·audit를 online feedback으로 사용하지 않는다. Audit 미측정을 통과로 기록하지 않고, 향후 확인 때 fullseen과의 중복까지 고려한 범위로 주장한다.

## 8. 계산량과 저장 범위

아래 값은1,000요청당 **early-stop이 전혀 없을 때의 최대치**다. 실제 사용량을 별도로 보고한다.

| 정책 | 최대 Adam updates | Target loss evaluations | Writer solves | History appends |
|---|---:|---:|---:|---:|
| N4 | 24000 | 25000 | 10 | 10 |
| REFIT4 | 48000 | 50000 | 20 | 10 |
| FROZEN2 | 24000 | 25000 | 20 | 10 |
| I2 | 24000 | 26000 | 20 | 10 |
| FROZEN4 | 24000 | 25000 | 40 | 10 |
| I4 | 24000 | 28000 | 40 | 10 |

REFIT4의 기존 실측은 Adam27525/loss29525다. I2/I4는 N4 및 해당 FROZEN control과 **최대 Adam budget**을 맞춘 것이지 실제 FLOPs·wall time·요청별 수렴 정도를 맞춘 것이 아니다. REFIT4보다 R/P가 낮으면 refresh 무효, 부족한 budget, 다른 anchor/teacher 중 하나로 바로 단정하지 않는다.

기존 online reference는 N4 304.8911초/B100, REFIT4 372.7526초/B100이다. 새 I2/I4/FROZEN runtime은 미측정이다. 최대 update수가 같다는 이유로 같은 비용이라고 선기록하지 않는다. GPU-hour hard cap은 사용자가 정하지 않았으므로 만들지 않는다.

Ledger는 준비, target forward/backward, matrix 준비/solve, readout, materialization/finalization, 진단, 평가, I/O, 실제 allocation을 분리한다. 중첩 timer를 더하지 않는다. 기존 reference 재사용 비용과 이번 신규 지출을 따로 보고한다. 일부 cache 최적화를 한 정책만 사용했다면 순수한 refresh 계산량 차이로 해석하지 않는다.

Request/chunk별 u/Z/a0/aj, canonicalY, residual, actual Adam count, m/v/t carry, loss 성분·stop reason, teacher ID, clamp 기준·hit를 재검산 가능하게 보존한다. 모든 inner subwrite의 actual delta와 필요한 entry/terminal snapshot을 남긴다. 새 전체 모델 checkpoint를 매 chunk 복제하지 않는다. 큰 raw/tensor는 ignored local 영역, Git에는 manifest·작은 집계·보고만 남긴다.

## 9. 결과 해석과 다음 단계

| 순차 결과 | 다음 판단 |
|---|---|
| FROZEN2가 REFIT4의 주요 이득을 유지 | Fresh second target 필요성은 약해짐. 두 번의 residual fitting을 더 단순한 후보로 평가 |
| I2가 FROZEN2보다 유리 | 같은 최대24 updates·같은2 writes에서 target을 intermediate state에서 이어간 구성의 추가 가치 |
| I4가 FROZEN4보다 유리 | 같은 최대24 updates·같은4 writes에서 분할 target 최적화의 추가 가치 |
| FROZEN4가 I4와 비슷하거나 우세 | Repeated fitting의 효과를 우선 설명으로 검토; fresh/interleaved target 고유 기여를 보류 |
| I4가 I2보다 낫지만 비용도 큼 | refresh 횟수의 정책 trade-off. I4 자동 채택 대신 이득·추가 비용을 함께 판단 |
| Split 계열이 R/P를 덜 달성 | 최대24회 운영점의 결과로 제한. REFIT4의 일부48회 사용과 다른 reset 규칙을 함께 보고 |
| PS/NS 이득과 strict/old-active 손실 공존 | 좁은 trade-off claim으로 허용 가능. 모든 지표 동시 개선을 요구하지 않음 |

최종 endpoint의 I2−FROZEN2/I4−FROZEN4 차이는 B52 이후 각자 달라진 trajectory까지 포함한 **정책 차이**다. 동일 state의 단일 구성요소 인과효과로 확대하지 않는다. B51 공통 entry와 저장된 같은-entry geometry 분석은 이를 보조한다.

첫 본 실험에 NM4 oracle chain, S875/S75 재실행, I4-reset, 8-refresh, Euler, 모든48-step 변형을 추가하지 않는다. NM4는 필요한 batch의 저장된 같은-entry native/REFIT update를 이용한 진단으로만 남긴다. 큰 비평행 성분만으로 기능적 유용성을 주장하지 않는다.

Split 계열의 budget 부족 여부가 핵심 쟁점으로 남을 때만 **I2-B48=[24,24], carry/teacher/clamp 동일, gamma=[.75,1]** 한 운영점을 추가할 수 있다. 이때도1,000요청 순차로 비교하며, 기존 REFIT4와의 차이를 budget 하나의 효과로 단정하지 않는다. 첫 campaign에는 자동 포함하지 않는다.

Middle 결과에서 구현 가능한 후보 하나를 고정한 뒤 **Late 공통 W90/M90→B91–B100**을 N4/REFIT4/선택후보로 비교한다. 후보가 REFIT4이면 두 정책만 수행한다. 각 정책은 다시1000요청이다. Terminal 분모가10000이어도 warm-entry 마지막1000개에 적용한 방법이며 처음부터full10k 정책을 적용한 증거는 아니다.

Intrinsic durability가 핵심 논문 claim이 될 때만 기존 W55 state origin×향후 updater hybrid를 별도로 수행한다. 이 hybrid와 E01 20-cell 완결은 이번 본 실험의 선행조건이 아니다.

## 10. 실행자가 남길 산출물과 판정 책임

- [정책별60개 batch cell 목록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-seq1000-cells.csv)
- [Machine-readable 실험 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-seq1000-contract.json)
- Entry/source/evaluator/capture 재사용 manifest, 기술 검증 receipt, policy lock.
- 정책별 terminal fullseen6000 및 old/new/active 성능, at-write/first-suffix500 전이, NLL tails.
- Request/chunk optimizer·residual·subwrite 집계와 실제 비용/메모리/저장 ledger.
- 누락·수치 실패·계획 변경을 포함한 SH 사실 보고, 이에 근거한 별도 global claim 판정.

허용 판정은 ALLOW / ALLOW_WITH_LIMITED_CLAIM / NEEDS_TARGETED_CHECK / NOT_SUPPORTED / INVALID_COMPARISON을 유지한다. 처음부터 특정 refresh 수나 ODE가 이기도록 결론을 고정하지 않는다. 사용자에게 보고할 최종 결과는 **어떤 대조에서 어떤 개선이 남았는지, 품질·과거 active·비용의 대가는 무엇인지, 다음 후보를 왜 선택했는지**다.

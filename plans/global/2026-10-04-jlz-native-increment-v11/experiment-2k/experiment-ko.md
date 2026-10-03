# JLZ v11 native increment: 2,000-edit 실험 설계

2026-10-04. 사용자 지시: method와 실험 설계를 GH에게 전달하고 SH4가 구현·실험한다. 범위는 **500개가 아니라 고정 첫 2,000개, BS100 × 20회 순차 편집**이다. 이 문서는 실행 설계이며, 수식과 구현 의미의 정본은 [method](../method-ko.md)와 [계약](../contract.json)이다. 기계 설정은 [experiment.json](experiment.json), 입력 증거는 [input-reference.json](input-reference.json), 순서는 [case schedule](case-schedule-first2000.csv)에 둔다.

현재 상태는 설계 완료·구현 및 technical pilot 대기다. 이 문서가 GPU 실행 완료나 GH/SH4 수신을 뜻하지 않는다. 사용자는 technical pilot 통과 후 아래 본실험까지 승인했다. 성능 결과를 보고 추가 확인을 요청하거나 과학적 gate를 새로 만들지 않는다.

## 1. 비교 대상과 고정 범위

| 체인 | 목적 함수 | 초기 상태 | 순서 |
|---|---|---|---|
| **MAIN** | native NLL/KL + 모든 층 delta norm + frozen-entry G+E 배분 비용, λ_alloc=0.1 | 독립 cold W0, H0=0 | 1 |
| **NOALLOC** | MAIN과 동일하되 λ_alloc=0 | 독립 cold W0, H0=0 | 2 |
| **MEMIT-H** | 원본 native MEMIT-H 경로 | 독립 cold W0, H0=0 | 3 |

NOALLOC은 **강제 균등 배분이 아니다.** 두 JLZ 체인 모두 모든 eligible layer의 delta를 공동 최적화하며, 차이는 명시적 배분 비용의 유무뿐이다. 과거 A/B의 다른 root 결합 방식을 재실행하는 실험도 아니다. 특정 layer subset, exact writer, reference set, replay, tracking, T′, pulse, warm-up, 강도 보정 및 coefficient sweep은 포함하지 않는다.

Native MEMIT-H는 동일 입력·환경의 비교 체인으로 둔다. baseline의 single-L8 z 계산, 요청별 native Adam/stop/clamp, remaining-layer residual writer를 v11 방식으로 바꾸지 않는다. v11은 **native ridge solve·actual key 갱신·history 의미를 유지하면서 local delta 증분을 쓰는 방법**이므로 전체 writer 절차가 native baseline과 같다고 표기하지 않는다.

기본은 세 체인을 순서대로 새로 실행한다. 다만 GH가 아래 identity와 모든 필수 endpoint를 충족하는 기존 MEMIT-H 2k 결과를 확인하면 baseline만 재사용할 수 있다. 모델·revision·dtype·입력 순서·B100·native source·context·C0·history·runtime·평가기·metric 분모 및 raw 행이 모두 같아야 한다. 이전 500개 집계나 다른 runtime의 결과는 참고치이며 matched baseline을 대체하지 않는다. 검증 근거가 부족하면 예정된 fresh baseline을 실행한다.

## 2. 현재 인스턴스와 이식성

| 항목 | 고정값 |
|---|---|
| 데이터 | `counterfact-fixed-10k-v1`의 `records[:2000]`, 순서 유지 |
| 모델 | Meta-Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| 편집 층 | L4, L5, L6, L7, L8 모두 |
| delta 위치 | 각 transformer block **전체 출력**의 subject-last token |
| writer | 해당 층 `mlp.down_proj` weight; key는 down_proj 입력 |
| NLL readout | L31 |
| native 입력 | 요청당 canonical 1 + prefix 5 rewrite, generic KL 1 |
| NLL context 가중치 | rewrite 6개 각각 1/6 |
| writer key 평균 | native nested mean: canonical 1/2, prefix 각 1/10 |
| λ_KL / λ_norm / λ_C | 0.0625 / 0.5 / 15000 |
| dtype | model FP32, geometry FP64, native history storage/append CPU FP32 |
| 실행 | eval mode, eager attention, autocast/TF32 off, seed 20261002 |

L4–L8, hidden dimension, context 수, B=100 또는 P/N panel 수를 알고리즘 내부 상수로 만들지 않는다. adapter가 실제 B_t, 층 목록, d_l, readout과 lookup, native row group/가중치, metric schema를 제공한다. 마지막 작은 batch에도 같은 정규화와 history 규칙을 쓴다. 현재 실험만 100으로 나누어 떨어지는 2k profile이다. 다른 benchmark/model에 적용할 때는 adapter identity를 별도로 잠그며 이 실험의 성능 결과를 자동 일반화하지 않는다.

## 3. fit과 commit 계약

배치 진입 모델을 W_entry로 고정한다. 각 층의 native 전체 배치 평균 key K_l을 한 번 얻고 A_l=15000 C0_l+H_l 및 ridge P_l, G_l, M_l을 계산한다. 이 **full-batch geometry는 fit 동안 고정**한다. microbatch마다 작은 writer를 새로 풀거나 off-diagonal을 버리지 않는다.

각 요청 r의 delta_lr는 해당 요청 rewrite와 KL 문장에 동일하게 주입된다. 모든 eligible layer가 한 forward 안에서 공동으로 작용한다. fit 도중 실제 weight writer/builder를 호출하지 않는다. 손실은 다음과 같다.

\[
 L=\frac1{B_t}\sum_r\left[\mathrm{NLL}_r+0.0625\,\mathrm{KL}(p_r\Vert p_{r,\mathrm{entry}})
 +0.5\sum_l\frac{\|\delta_{lr}\|_2}{a_{lr}^2}\right]
 +\lambda_{\mathrm{alloc}} C_{GE}^{\mathrm{entry}}.
\]

\[
 C_{GE}^{\mathrm{entry}}=\sum_l\sqrt{\frac{
 \mathrm{tr}(D_lG_lD_l^\top)+\|D_l(M_l-I)\|_F^2}
 {\sum_r a_{lr}^2}}.
\]

anchor와 matrix 정의는 method 계약을 따른다. root를 임의 epsilon으로 부드럽게 바꾸지 않고 0에서 명시한 subgradient 0을 사용한다. G+E는 **frozen-entry write proxy**다. 실제 순차 commit에서 갱신한 key의 비용과 같다고 보고하지 않는다.

최적화는 delta_lr = a_lr q_lr / sqrt(d_l m), q=0 초기화, Adam lr=0.1, betas=(0.9,0.999), eps=1e-8, weight_decay=0, foreach=false다. 각 update 후 delta_lr/a_lr를 0.75에 개별 clamp한다. 전 층 합 norm과 per-layer clamp는 single-layer native의 전역 편집 예산과 같은 양이 아니다. 기록·stop 판정은 request-mean J로 하되, optimizer에는 기존 JLZ의 request-sum 관례에 따라 `q.grad = B_t * s * ∂J/∂D`를 정확히 한 번 전달한다. B_t 또는 s를 중복 적용하지 않는다.

한 배치에 최대 **25 candidate 평가, 24 Adam update**다. 평가한 **배치 전체 total mean loss(배분 항 포함)가 0.05 미만**이면 다음 update 없이 종료한다. 요청별 NLL<0.05 freeze나 독립 early stop을 추가하지 않는다. 마지막 finite candidate를 반환하며 best-loss 후보나 P/NS 우수 후보로 사후 교체하지 않는다. finite·shape·dtype·state 정합 실패는 기술 실패로 기록하고 invalid candidate를 commit하지 않는다.

fit 후 delta 증분 D_l은 고정한다. 낮은 층부터 다음을 수행한다.

1. 앞선 낮은 층이 **실제로 write된 모델**에서 현재 층 native 평균 key K_l을 다시 읽는다.
2. 같은 native ridge 식으로 P_l=(A_l+K_l K_l^T)^(-1)K_l을 다시 계산한다.
3. U_l=D_l P_l^T를 weight에 반영한다.
4. 다음 층으로 진행한다. 모든 층 완료 후 final model의 native key로 history Gram을 한 번 append한다.

`absolute virtual z − actual h` 피드백, 남은 층 수 divisor, 역행 보정, exact writer로의 fallback은 쓰지 않는다. batch 동안 H는 고정하며 success·중복 claim 여부로 append를 필터링하지 않는다. final-key 재캡처를 기존 capture로 재사용하려면 해당 모델 구조에서 동일하다는 pilot parity와 identity 증거가 필요하다.

## 4. 고정 입력과 중복·version 처리

모델 로딩 전 `scripts/fixed_counterfact.py verify`와 prefix 검증을 수행한다. server4 데이터 경로 힌트는 `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json`이다. 경로 존재만으로 검증을 대체하지 않으며 같은 폴더의 receipt/sample lock도 확인한다.

- 데이터 SHA256: `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`.
- 첫2k case ID 배열 SHA256: `0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`.
- schedule SHA256: `dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2`.
- 첫/마지막 case ID: 16186 / 19132. 2,000개 case ID는 모두 서로 다르며 각 record는 P 2개, N 10개다.

동일 claim은 `(NFC(공백 정규화 subject), relation_id)`로 판별한다. 더 늦은 다른 target이 오면 앞 occurrence는 superseded다. target이 이후 원복되어도 과거 occurrence를 재활성화하지 않는 기존 native observer 규칙을 사용한다.

| Endpoint | 전체 occurrence | unique claim | conflicting claim | active occurrence | superseded |
|---|---:|---:|---:|---:|---:|
| W5 | 500 | 500 | 0 | 500 | 0 |
| W10 | 1,000 | 999 | 1 | 999 | 1 |
| W15 | 1,500 | 1,491 | 8 | 1,491 | 9 |
| W20 | 2,000 | 1,983 | 13 | 1,984 | 16 |

학습·history·primary 평가에서는 모든 occurrence를 유지한다. active/superseded는 보조 분모로 각각 보고한다. `active_at_W20` 열을 W5 등의 과거 label로 사용하지 않는다. 완전히 같은 panel prompt/target의 중복 수를 raw identity로 세고 unique-prompt 보조 집계를 명시하되, primary의 2,000/4,000/20,000 행을 조용히 dedup하지 않는다.

## 5. Technical pilot와 본실험 진입

pilot은 고정 배열 index 2000..2003의 **541, 6693, 16935, 17306**을 사용한다. 이번 설계에서 데이터로 다시 검산한 값이며 main과 case ID가 겹치지 않는다. 기존 native target cache를 불러오지 않고 fresh target을 계산한다. MAIN, NOALLOC 각각 cold W0/H0에서 **BS2×2회 실제 순차 commit**을 수행한다. MEMIT-H도 같은 4개로 BS2×2 소형 parity를 수행한다. pilot 상태·target·history를 main으로 넘기지 않는다.

필수 확인은 다음과 같다.

- native token/mask/lookup/context 가중치 및 single-layer NLL·KL·norm reference parity.
- full-batch geometry의 고정, off-diagonal 보존, root 0 및 비영 candidate의 finite gradient.
- reference와 최적화 경로의 loss/gradient parity. microbatch 변경으로 B_t나 가중치가 바뀌지 않음.
- q-scale/Adam/clamp, 최대 25평가·24update, total-loss whole-batch stop의 정확한 순서.
- 증분 D와 실제 upper-key 갱신, divisor/absolute feedback 부재, final history의 정확히 한 번 append.
- raw R/P/N 행·분모·version label, endpoint 재사용, 체인 상태 분리와 noCP.

허용오차는 구현 설계 §3과 같다. FP32 scalar/tensor는 `atol=2e-5, rtol=2e-4`, D/q gradient는 `||차이|| / max(||reference||, 1e-8) ≤ 2e-3`이며 reference gradient norm≤1e-8이면 차이의 max-abs≤1e-6이다. FP64 stable cost는 `atol=1e-8, rtol=1e-6`, relative solve residual≤1e-8, CPU 대수 항등식 오차≤1e-10, finite-difference는 `atol=1e-7, rtol=1e-5`다. 입력 identity는 exact다. clamp 상대 slack은 5e-6이며, commit weight는 선택한 dtype·연산순서로 materialize한 update와 정확히 일치해야 한다. stop 경계 0.05 근처에서 reference/최적화 경로의 판정이 갈리면 reference loss로 판정한다. 실패 시 한 번 reference 재확인하고 원인·fallback을 기록한다. 과학적 결과를 보고 허용오차를 늘리지 않는다.

pilot에는 PS/NS, layer 집중도, loss 수렴 여부를 이용한 과학적 합격선이 없다. **정의된 technical PASS 후 MAIN→NOALLOC→MEMIT-H 본실험을 자동 진행**한다. reference parity 실패, nonfinite, 잘못된 상태/데이터만 해당 실행을 막는다. 배분 모양이나 locality가 기대와 다르다는 이유로 coefficient를 고치거나 2k 전에 종료하지 않는다.

## 6. 평가 일정과 지표

| 시점 | 평가 대상 | R/P/N occurrence 분모 |
|---|---|---|
| W0 | 첫2k 전체 | 2,000 / 4,000 / 20,000 |
| 모든 batch 직전 | 해당 current 100개 | 100 / 200 / 1,000 |
| 모든 batch 직후 | 해당 current 100개 | 100 / 200 / 1,000 |
| W5 | 첫500 all-seen | 500 / 1,000 / 5,000 |
| W10 | 첫1,000 all-seen | 1,000 / 2,000 / 10,000 |
| W15 | 첫1,500 all-seen | 1,500 / 3,000 / 15,000 |
| W20 | 첫2,000 all-seen | 2,000 / 4,000 / 20,000 |

그 외 endpoint의 all-seen은 실행하지 않고 `NOT_MEASURED`로 남긴다. W5/10/15/20의 current 행은 동일 all-seen 평가 결과에서 재사용한다. W0도 모델 상태·입력·runtime·평가기 identity가 같을 때만 체인 간 공유한다. 같은 raw 행을 가리키는 current/all-seen view를 두 번 평가해 비용을 늘리지 않으며, 서로 다른 endpoint를 대체하지 않는다.

모든 R/P/N에 다음을 함께 기록한다.

- **NLL preference:** R/P는 `new_nll < true_nll`, N은 `true_nll < new_nll`; 동률 실패.
- **Strict ACC:** teacher forcing에서 desired target의 모든 token이 argmax 정답. R/P desired=new, N desired=true. 자유 생성 exact match와 구분한다.
- **Token ACC micro:** 정답 target token 수 합 / 전체 target token 수 합.
- **Token ACC prompt macro:** prompt별 token 정확도의 산술 평균.
- new/true NLL, 각각의 token 수와 target ID hash. N에서는 new/true NLL 및 margin의 pre→post·birth→checkpoint 변화도 남긴다.

각 요청의 편집 직후 current를 birth로 두고 이후 W5/10/15/20에 같은 raw identity로 retention을 연결한다. current 직전→직후, W0→checkpoint, 첫500/1000/1500 cohort를 각각 제공한다. N의 correct→incorrect 및 reverse 전환은 **각 방법의 own-at-birth-correct 분모와 공통 correct 분모**를 구분해 비교한다. endpoint마다 active/superseded label과 prompt 중복 정책을 함께 제공한다. 관측되지 않은 중간 checkpoint를 보간하지 않는다.

평가 P/N 문장, label, NLL, PS/NS는 fit·stop·후보 선택·계수 결정에 들어가지 않는다. 기존 결과와 PS/NS 수치만 같다는 이유로 evaluator identity를 같다고 판단하지 않는다.

## 7. 배분·실현·비용 기록

후보별 total/NLL/KL/norm/allocation, candidate/update 번호, 반환/정지 사유를 기록한다. 요청·층별 delta/a와 planned share, clamp 횟수·상한 도달률·분위수를 남긴다. frozen-entry G/E/root와 actual writer의 갱신 geometry 비용은 다른 이름으로 기록한다. fit-entry와 actual key 차이, subject 위치의 실제 증분과 delta 차이, context별 차이, planned/realized share도 남긴다.

NLL/KL/norm/allocation의 성분 gradient norm은 후보2/9/terminal(실제 방문한 후보만, 중복 제거)에서 측정한다. 진단 backward는 실제 optimizer update를 수행하지 않고 별도 call ledger로 구분한다. MAIN과 NOALLOC의 global-step 후보 수가 early stop 때문에 다르면 그대로 보고한다. 동일 step 수로 강제로 맞추지 않는다.

entry geometry, fit, writer/key refresh, history, evaluation 각각의 wall time, valid/padded token 수, 실제 forward/backward 수, peak GPU/host memory를 기록한다. 논리 candidate 수와 실제 호출 수를 구분한다. 최대 JLZ 본실험은 40 batch, 1,000 candidate, 960 update이며 실제 early stop 수를 함께 제공한다. 속도 향상과 ETA는 아직 측정하지 않았다.

평가 예산은 체인당 pre-current 2,000 request, post 6,600 request다. native prompt pair 기준 pre 26,000 + post 85,800이며, 세 fresh 체인과 공유 W0의 논리 상한 합은 361,400 pair다. pair는 prompt 하나에서 new/true 두 target을 비교하는 단위이며 모델 forward 횟수가 아니다. W0와 B1 pre-current의 동일 raw 재사용 등 물리 실행 절감은 별도로 기록한다.

## 8. 자원, 중단, 산출물

SH4가 server4에서 실행한다. **이 task의 동시 GPU cap은 1**, job당 GPU 1개, CPU 기본 8개다. 현재 server4/project의 더 엄격한 cap과 실제 기존 allocation을 우선한다. GH/SH4가 새 admission에서 GPU 점유, 명시적 host memory 요청과 현재 repository ceiling을 확인한다. 설계문에서 과거 59GiB 등의 자원값이나 ETA를 추정해 고정하지 않는다. 사용 중이면 scheduler queue/dependency로 기다리며 다른 실험을 취소·수정하지 않는다.

기본 `save_checkpoints=false`를 유지한다. full W/U, 복원 가능한 R+P, H, optimizer/RNG bundle을 저장하지 않는다. 한 체인은 같은 process/RAM에서 진행하며 in-process snapshot은 process 종료 뒤 사용할 수 없다. **정확한 resume은 불가능**하다. 중간 중단 시 완료 commit/endpoint까지의 raw와 상태를 보존하고 PARTIAL로 보고한다. 재실행이 필요하면 새 attempt에서 cold W0/H0로 시작하며, 과거 attempt의 raw를 하나의 연속 체인처럼 이어 붙이지 않는다.

대형 raw/log는 server-local `local/` 아래에 보관하고 provenance·작은 집계·SHA256 manifest·한글 보고서만 Git에 둔다. GH envelope에 따른 정상 artifact broadcast와 검증을 수행하되 checkpoint를 생성해 전송하지 않는다.

필수 산출물은 다음과 같다.

1. source/config/model/tokenizer/runtime/input/evaluator를 잠근 `execution.lock.json`과 frozen source manifest.
2. technical pilot의 모든 정의된 PASS/FAIL, parity 오차, exception, resource receipt.
3. 체인별 20개 commit/history ledger와 candidate/배분/실현 telemetry.
4. W0·각 pre/post current·W5/10/15/20 raw R/P/N, strict/token/preference/NLL 집계.
5. paired retention, N leakage, active/superseded 및 명시적 중복 보조 집계.
6. component time/token/call/memory, archived baseline reuse 판정 또는 fresh baseline 실행 증거.
7. SH4 한글 사실 보고서와 raw 위치·bytes·SHA256 manifest.

SH4는 사실·수치·산술 차이·기계적 gate·경로·hash·typed failure만 보고한다. 과학적 의미와 다음 method 권고는 GH의 별도 리뷰에 둔다. MAIN·NOALLOC 각20 commit과 필수 raw, matched MEMIT-H 2k 비교까지 갖춰야 전체 완료다. 한 체인이나 endpoint 누락을 DONE으로 표기하지 않는다.

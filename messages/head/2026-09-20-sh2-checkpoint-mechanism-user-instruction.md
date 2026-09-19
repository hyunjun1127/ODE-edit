Server2에서 회수한 AlphaEdit L4-only lifelong checkpoint의 기전 분석을 구현하고 실행하라. 설계 재작성이나 검토만으로 끝내지 말고, 입력 준비 → runner 구현 → 수치 검증 → 단계별 분석 → 최종 보고까지 진행하라.

이번 작업의 목적은 다음 연결을 실제 자료로 확인하는 것이다.

고정 key와 누적 history의 관계
→ 해당 방향의 target 요구량
→ 실제 native write의 부담
→ 과거 편집 및 neighborhood margin 변화

이 연결이 성립한다고 전제하지 말고, 각 연결이 지지되는지 분리해서 판정하라.

1. 반드시 읽을 문서

설계:
 /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-server2-checkpoint-mechanism-audit-design-v1.md

실행 contract:
 /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-server2-checkpoint-mechanism-audit-contract-v1.json

분석 단위·의존성:
 /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-20-server2-checkpoint-mechanism-audit-cells-v1.csv

Server2 preflight:
 /mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-server2-checkpoint-mechanism-design/server2-preflight.json

평가 source 감사:
 /mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-server2-checkpoint-mechanism-design/functional-source-audit.json

기존 checkpoint inventory:
 /mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-server2-geometry-gss-history-review/server2-checkpoint-inventory.json

설계·contract·CSV를 함께 적용하라. 수치 기준은 contract의 분모·집계 방식·zero-reference 처리까지 그대로 따른다. 실제 충돌이 발견되면 영향을 받는 부분과 해결 근거를 기록하고, 독립적으로 유효한 작업은 계속 진행하라.

현재 문서는 설계 상태이며 새 분석 runner가 구현되어 있다고 가정하지 마라. CSV의31개 row는 분석 단위이지31개 editing arm이 아니다.

2. 실행 범위와 작업 공간

모든 모델·행렬·평가 분석은 Server2에서 실행한다.
Server4는 원본 target·평가·source를 읽고 가져오는 용도로만 사용한다.

Server2 출력 root:
 /mnt/raid5/janghj/ODE-edit/local/checkpoint-mechanism-audit/20260920-v1/

새 attempt 디렉터리에 입력 manifest, 실행 source, cache, logs, 결과를 분리해 저장하라. 기존 작업과 원본 checkpoint를 덮어쓰지 마라.

새 runner와 필요한 검증 코드는 이번 작업 전용 경로에 구현하라. 기존 native writer/evaluator의 수치 의미를 유지하고, 다른 진행 중인 작업의 코드나 환경을 변경하지 마라.

이번 범위:
- 저장 평가의 종단 분석
- actual W/M geometry
- 고정 key 추출
- 저장 z를 이용한 native write 재구성
- 소수 사례의 activation–margin 기전 분석

이번 범위에 넣지 않을 것:
- 새로운 local-z 최적화
- 10k lifelong replay
- EN correction이나 GSS 실행
- 새로운 layer allocation arm
- 공식 PS/N을 이용한 방법 최적화
- 기존 reference512/G256의 교체·축소·재생성

3. 입력 staging과 독립적인 결속 검사

Server2 checkpoint root:
 /mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/payload/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/

Checkpoint:
 B001,B005,B010,B020,B030,B040,B050,B060,B070,B080,B090,B100

Server4 companion root:
 /data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/

필요한 자료:
- seen-full.json: 위12경계
- current.json: B001–B100 전체
- commit.json, entry.json, native-observation.json, contexts.json: batch별 기록
- native-targets.pt: B001,B002,B006,B011,B021,B051,B091
- 봉인된 sample/order, dataset, runtime, source, hparams, dependencies

S2에 이미 있는 W/M을 다시 복사하지 마라.
Source→destination hash와 state·request identity를 검증하라.
약449MB의 평가·선정 target 외 source/dependencies 크기도 기록하라.

입력 성공 여부는 다음 component별로 관리하라:
- archival evaluation
- sealed order
- checkpoint W/M
- W0 L4
- original source
- model runtime
- P4
- 각 batch의 target

Late target 하나가 없거나 실패했다고 CPU 저장평가 분석까지 막지 마라.
큰 파일의 hash/finite 검사는 최초 검증 후 유효한 receipt를 재사용하라. 반복 호출마다 전체 파일을 다시 검사하지 마라.

4. 원 실행 환경을 재현하라

핵심 조건:
- Model revision: 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2
- 편집 weight: model.layers.4.mlp.down_proj.weight
- Weight shape: [4096,14336]
- 모델 실행: FP32/eager/autocast=false
- Torch: 2.9.1+cu128
- Transformers: 4.44.2
- TF32 matmul=false
- TF32 cuDNN=true

S2 기본 Transformers4.57.1 환경을 그대로 사용하지 마라.
원 dependencies를 확인한 격리 환경을 준비하라.

원 safetensors의 L4는 BF16이다. 원 실행처럼 FP32로 변환한 뒤 W0 tensor hash를 확인하라.

저장 projector는 [5,14336,14336]이며
source layers=[4,5,6,7,8]의 slot0가 L4다.
Selected P4 fingerprint를 검증하고 저장 projector를 재사용하라.

Writer와 evaluator tokenizer의 BOS 설정을 구분하라.
원 evaluator의 microbatch16, 수동 left-padding, implicit position_ids를 보존하라. Tokenizer 설정 문자열만 보고 batching을 바꾸지 마라.

Migration manifest와 runtime의 job 표기가 다르다는 사실을 기록하되, model/source/sample/context/P/W/M hash와 entry/endpoint 결속으로 lineage를 판단하라.

5. CPU 분석을 먼저 완료하라

A. 저장 평가 종단 분석

Join key:
 (arm, metric_tag, identity)

같은 endpoint의 current/seen-full 중복은 일치 확인 후 한 번만 센다.

Signed safety margin:
- RS/PS: true_nll−new_nll
- NS: new_nll−true_nll
- 양수가 성공, tie는 실패

최초 anchor는 해당 요청이 도착한 batch의 current.json이다.
첫 sparse checkpoint를 최초 편집 시점으로 쓰지 마라.

필수 비교:
- At-write→각 checkpoint
- First100 cohort: 모든12경계
- First1000 cohort: B10 이후
- 인접 checkpoint 사이의 동일 요청
- Edit age별 변화

처음 실패, 이후 forgetting, 회복을 분리하고,
1→1/1→0/0→1/0→0의 분자·분모를 모두 보고하라.
Token-strict와 RS∧PS0∧PS1 joint도 포함하라.

Overwrite:
원래 all-case benchmark를 유지하면서 target-conflict-free와 active-version 진단을 별도로 보고하라. 동일 batch의 상충 target은 BATCH_INTERNAL_CONFLICT로 표시하라.

Bootstrap은 설계대로 request cluster2000회, seed20260920이다. 단일 stream의 결과를 다른 edit 순서에 대한 보장으로 해석하지 마라.

B. Actual W/M 분석

D_ab = float64(W_b)−float64(W_a)
E_t = float64(W_t)−float64(W0)

W norm/angle, M trace/diagonal, 구간 변화와 다음을 계산하라:
 J_ab = tr(D_ab M_a D_ab^T)

Trace 추정은 동일 random vectors256개를 모든11구간에 사용한다. 64/128은 진행 진단이며 adaptive stopping에 사용하지 마라.

J의 정규화, MC SE, per-batch squared-norm을 이용한 구간 교차항도 출력하라.
저장 M의 FP32 누적 오차를 명시하고, J를 factual forgetting으로 부르지 마라.

6. 작은 prefix 검사와 B1 재현을 통과하라

처음에는 B1/B2 요청과 geometry probe의 첫32개로 검증한다.

Native key:
context group[1,5] 안에서 평균한 뒤 group 사이 평균.
모든6개 context의 단순 평균으로 바꾸지 마라.

별도로 저장:
- Native mean key
- Bare-prompt subject key
- W0 bare block output
- Token/mask/position/context/source identity

L4 prefix를 capture한 뒤 suffix를 중단하는 경로는 physical full forward와 비교한 후 사용하라.

B1에서:
- K1K1^T와 저장 M1의 차이
- h0+(Wentry−W0)k_bare와 physical activation
- 원래 dense-RHS solve와 actual W1−W0
- Current-key response
- 원 evaluator row별 결과
를 검증하라.

이 archive의 B1은:
 RS100/100, PS190/200, NS867/1000

최근 EN의100/194/865를 기대값으로 쓰지 마라.

W0의 같은 first100 R/P/N도 평가해 별도 W0 anchor를 만든다.
Contract의 numerical gates를 통과한 범위만 확장하라.
결과를 맞추려고 tolerance를 넓히지 마라.

7. Key bank와 history operator 분석

검증 후 다음을 추출한다:
- Native700requests:
  B1,B2,B6,B11,B21,B51,B91
- Fixed geometry probe512:
  sealed10k case ID를 제외한 동일 dataset에서
  hash(seed,case_id) 순서로 선정

Geometry probe는 기존 train reference512와 별개다.
Case ID disjoint 및 subject/relation overlap을 기록하라.

동일 key를 모든 history에 재사용한다.
Pilot history는0,1,10,100.
통과하면 나머지9경계를 분석해 W0 포함13상태를 완성하라.

Raw operator:
 H = λI + P M
 Y = solve(H, P K)
 S = K^T Y
 B = Y (I+S)^−1

실제 구현은 inverse 대신 solve를 사용한다.
H는 비대칭일 수 있으므로 raw CG/Cholesky를 사용하지 마라.

같은 history의 RHS bank와 factorization을 재사용하되,
probe512와 native700을 하나의 editing batch로 합치지 마라.
각 native batch100개는 자기 S/B를 사용한다.

Raw score, projector/solve/skew 검사와 out-of-range 값을 보고하라.
Ideal 범위에 맞추기 위해 clipping하지 마라.

8. 저장 z로 demand와 write 부담을 분해하라

Pairing:
 B1  ← W0/M0
 B2  ← B1
 B6  ← B5
 B11 ← B10
 B21 ← B20
 B51 ← B50
 B91 ← B90

Residual:
 R_i = z_i − [h0_i + (Wentry−W0)k_bare_i]

Mean key를 bare residual 계산에 사용하지 마라.

B1 외에는 actual 다음 weight tensor가 없으므로
RECONSTRUCTED_NATIVE_WRITE로 구분하라.
Commit norm이나 current 평가가 일치해도 actual tensor를 확보했다고 하지 마라.

B1과 B91에서 factor 경로와 원 FP32 dense-RHS solve를 대조하라.
Late parity가 실패하면 해당 factor attribution의 과학적 주장을 보류하라.

Core burden decomposition:
 B = L Σ V^T
 Δfactor = R B^T
 ||Δfactor||F² = Σ_j σ_j(B)² ||R v_j||²

B의 직접 thin SVD를 사용하라.
Gain, target loading, energy를 각각 보고하고,
actual/native 재현 잔차와 근접 singular-value band도 기록하라.

대수적 대조:
- B1/B2의 K,R 고정 × history0,1,10,100
- 같은 K/operator에서 R 열 permutation20회

이 결과는 대수적 counterfactual이며 factual editing arm의 성능이 아니다.

9. 실제 activation과 neighborhood margin을 연결하라

구간:
- B1→B10
- B50→B100

두 구간 모두 first100 cohort의 NS1000에서
lost16 + matched retained16을 선정한다.
부족하면 실제 수를 보고하고 모집단을 임의로 넓히지 마라.
Matching·seed·conflict 제외는 설계를 따른다.

각 사례에서:
- D K
- E K와 D K의 상쇄·증폭 교차항
- W_a+sD, s=0,.5,1의 실제 margin/strict
- Entry margin gradient와 D K의 내적
- 실제 Δmargin과 선형 예측의 차이
를 계산한다.

True/new target의 TF 경로를 분리하고,
gradient와 K는 필요한 모든 valid input positions를 포함한다.

이 panel은 사후 기전 사례다.
대표 성능이나 방법 threshold 조정에 사용하지 마라.
연결된 parent requests의 R/P 변화도 함께 보고하라.

10. 계산량과 실패 처리를 구현하라

기본 예산:
단일48GB GPU, CPU8threads, RAM64GB 내 chunking.

실제 실행 시 자원을 확인하라.
과거 GPU idle snapshot을 예약으로 간주하지 마라.

모델 load, key capture, history별 LU와 검증 receipt를 재사용하라.
새 z optimization 호출 수는0이어야 한다.
Prefix/suffix forward, backward, solve, hash, I/O, peak RAM/VRAM을 따로 계측하라.

Pilot 비용을 측정해 확장 비용을 산출하라.
OOM이나 구현 오류는 기존 의미를 보존하는 chunking/실행 수정으로 해결하라. 성능 결과를 보고 과학적 조건을 바꾸지 마라.

각 cell은 PASS/FAILED/BLOCKED/SKIPPED를 기록한다.
FAILED/BLOCKED는 관련 후속 주장만 차단한다.
유효한 CPU 결과를 보존하고 독립 작업은 계속한다.
최종 보고서는 모든 cell의 성공을 기다리는 대신 terminal 상태를 수집해 반드시 생성하라.

11. 제출물과 완료 기준

Contract의 required_outputs 전체를 생성하라.
특히 다음을 포함하라:
- Input/source/state binding과 numerical parity
- Functional longitudinal table와 acquisition/forgetting 전이
- Actual W/M geometry와 trace 추정 오차
- Fixed-probe history 결과
- Native gain×target loading 분해와 reconstruction 잔차
- Counterfactual 대조
- Activation–margin 연결
- Cell별 상태·실제 실행 command·source/environment manifest
- 단계별 시간과 메모리
- 재생성 가능한 CSV 및 PDF/PNG
- 한국어 최종 report

최종 보고는 H1–H4를 각각
SUPPORTED / MIXED / NOT_SUPPORTED / UNRESOLVED로 판정하고 근거를 연결하라.
H5는 이번 결과에 따른 후속 연구 질문으로 남겨라.

분석 가능한 입력이 갖춰지고 gate가 통과하면 전체 설계 범위까지 이어서 수행하라. 최소 완료 기준만 충족했다는 이유로 가능한 나머지 분석을 임의 생략하지 마라.

상관을 인과 증명으로, 작은 singular value를 필연적인 write 증폭으로, weight norm을 locality 손상으로, 높은 RS를 single-layer의 무제한 capacity 증명으로 표현하지 마라.

현재 진행 상태, 최초 CPU 분석 결과, B1 parity, operator 확장 결과, 최종 기전 판정을 단계별로 보고하라.
# JLZ v10 T′ — SH3 구현·pilot·500-edit 실험

2026-10-03. 사용자 실행 지시: **“GH에게 이 method 설계와 실험 설계 전달해. SH4이 아닌 SH3에게 task 진행시키자.”** 정본 instruction ID는 `ODEEDIT-USER-GH-SH3-JLZ-V10-TPRIME-500-20261003-R1`이다. GH가 SH3/server3에 구현·qualification·pilot·본선을 배정한다. 기존 설계의 NOT_RUN은 검증 상태이며 이번 사용자 실행 권한을 취소하는 문구가 아니다.

정본은 [method](../method-ko.md), [method 계약](../contract-draft.json), [baseline 비교](../comparison-ko.md), [실험 JSON](experiment.json), [실행 명령](../execution-command.json)이다. V9/v7 문서 전체를 상속하지 않는다.

## 1. 목적과 고정 범위

공동 local-z와 actual writer 실현량을 연결한 T′가 편집·일반화·locality 및 층별 배분에서 보이는 결과를 native baseline과 비교한다. A는 층별 root energy 합, B는 root-sum-square다. 두 arm 모두 전 eligible layer를 최적화하며 한 층 집중을 허용한다. A/B 차이에는 비용 크기 차이도 포함되므로 분산 효과만 분리했다고 주장하지 않는다.

| 항목 | 범위 |
|---|---|
| 담당 | **GH→SH3/server3** |
| Model | Meta-Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| Benchmark | CounterFact fixed10k의 기존 순서 first500 |
| Main | **A/B 각각 독립 W0/H0, B100×5=500 edits** |
| Eligible layers | L4–L8 모두 |
| Candidate/update | batch당25/24, terminal 후보 채택 |
| Optimizer | q Adam LR.1, betas(.9,.999), eps1e−8, warm-up0 |
| Native norm | **λ_n=.5, actual v의 native 그룹 가중 비제곱 norm/a²** |
| KL / allocation | .0625 current∥own-entry / .1 |
| Dtype | Model FP32, geometry FP64, autocast/TF32 off, reference attention eager |
| Clamp | 없음; v/a의 .75 초과는 기록만 |
| 추가 B1 calibration / baseline fit / exact probe | **모두0** |
| 종료 | 각 arm W5 누적평가 후 종료, B6 없음 |
| Checkpoint | 저장하지 않음 |

B·layer 수·context 수·차원은 이 profile의 값이다. 구현은 runtime 실제 B/차원 및 부분 batch를 처리해야 한다. 서버 이동으로 model/benchmark/hyperparameter를 바꾸지 않는다.

## 2. 구현에서 지켜야 할 계산 그래프

각 후보마다 actual lower all-token write→상층 context key→whole-B native rewrite mean→ridge P/U를 구성한다. **Actual context key로 만든 v^a를 subject fit에 주입**한다. Fit key의 Uk^s 주입은 이전 T이며 이번 T′가 아니다.

V9 builder는 rewrite rows만 처리하므로 KL rows도 actual 하층 경로로 전달하도록 확장해야 한다. KL rows의 v^a는 KL subject 위치에 주입하되 ridge mean/history에는 넣지 않는다. Rewrite norm은 native key 그룹 평균 가중치(.5 및 prefix별.1), NLL은 native context 균등 평균(각1/6)을 유지한다.

G-only energy를 사용하고 E·detached teacher pulse·history replay·active clamp는 제거한다. V9의 R 직접 subject 주입, pulse 후보5/10/15/20, exact terminal 비교, clamp.75를 이어 쓰지 않는다. q/R/v=0의 norm subgradient0과 NaN 처리를 검증한다.

Microbatch별 독립 ridge solve 또는 K/P 영구 detach는 금지한다. 권고 구현은 v tensor 경계에서 masked NLL/KL/norm adjoint를 모아 원래 whole-B builder에 한 번 전달하고 allocation gradient를 합치는 방식이다. Temporary leaf에 분리한 gradient는 반드시 builder로 돌려보낸다. U/k 경계와 v 경계를 동시에 seed하여 중복 계산하지 않는다.

Terminal에서 actual 모델을 평가한 바로 그 materialized U를 commit하고 actual rewrite mean-key H를 한 번 누적한다. Candidate 평가 중 W_entry/H_entry를 누적 변경하지 않는다. Actual와 masked 기저 gap은 telemetry이며 이를0으로 강제하는 성능 gate를 만들지 않는다.

새 namespace는 `project/run_scripts/jlz_realized_subject/`다. SH3는 dedicated non-main worktree에서 이 namespace와 그 내부 tests를 소유한다. V9 frozen source `b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66`의 코드와 native adapter/evaluator는 읽어 재사용하되 실행 중인 다른 method source를 덮어쓰지 않는다. GH가 정확한 source scope와 구현/검증/본선 실행 권한을 전달한다.

## 3. 입력과 SH3 runtime 결속

[입력 reference](input-reference.json)는 v9에서 확인한 model/context/dataset/C0/input identities만 담는다. 과거 server4 경로는 위치 힌트이며 SH3는 server3의 실제 경로·SHA·native 함수·Python/torch/transformers/attention/dtype를 새 receipt에 기록한다. 공유 환경을 무조건 설치·변경하지 않는다.

포함한 [first500 schedule](case-schedule-first500.csv)의500 case ID compact JSON SHA는 `0be7d88c759e7f65a690514035f40a18c5c19d8591ddd33f97b2fa6187b64a95`다. 처음3개는16186/3743/1481, 마지막3개는3419/8549/5628이다. `active_at_W20`은 이전2000-edit 일정의 메타데이터이므로 W5 active 여부 판정에 사용하지 않는다. 입력·순서·분모를 바꾸거나 부족한 요청을 대체하지 않는다.

Raw 모델/데이터/C0를 이 전달 archive에 넣지 않는다. 기존 SH3 자산을 먼저 확인하고 부족한 프로젝트 자산은 GH가 승인한 전달 경로로 확보한다. Source/config/input/evaluator/seed를 실험 전에 결속한다.

## 4. 실행 단계와 pilot

| 단계 | 작업 | 종료 조건 |
|---|---|---|
| Q0 | 제공 CPU 대수·T′ adjoint 검산 재현, production source audit | Toy PASS와 actual model PASS를 구분 |
| Q1 | Main 밖 case541/6693/16935/17306, A/B 각각 독립BS2×2,25후보/24갱신 | 실제 native T′ 주입·KL·K/P gradient·B2 history 및 transaction 검증 |
| Q2 | **Main A의 cold B1 자체**로 whole-B100 동작·자원 확인 | 동일 terminal U commit/H 한 번; 추가 B100 fit 없음 |
| Q3 | A는 같은 B1에서 B2–B5, B는 독립 cold B1–B5 | 두 arm W5 누적 평가 및 collector 완료 |

Q1 state는 main으로 전달하지 않는다. Q1의 고정 후보에서 B1/B3 및 microbatch 분할·불균등 길이 parity를 확인할 수 있으나 추가 장기 fit arm으로 확대하지 않는다. Q2는 처음부터 main A B1으로 등록한다. 구현/source/config가 바뀌면 새 attempt로 cold restart하고 이전 결과와 합치지 않는다.

Qualification은 dense 기준과 분할/direct 경로의 forward·gradient, K/P full path, rewrite/KL row mapping, 동일-weight commit/history에 대한 기술 검증이다. 기본 FP32 forward 허용오차는1e−5+1e−4×reference scale, gradient RMS는1e−6+1e−3×reference RMS, FP64 solve residual은1e−8이다. Commit weight identity는 동일 materialized tensor로 검증한다. 실패 시 source-backed 원인을 기록하고 기술 수리 후 같은 계약을 다시 검사한다. 기준을 성능이 좋다는 이유로 완화하지 않는다.

낮은 PS/NS, 특정 층 집중, .75 초과, candidate budget 내 미수렴은 연구 결과다. 재fit·계수 조정·조기 중단의 사유가 아니다. Nonfinite, 잘못된 key/gradient, state corruption, source/input mismatch는 기술 실패다. Qualification 통과 후 단계마다 사용자 재승인을 기다리지 않고 승인된 범위를 진행한다.

## 5. 평가와 비교

W0 기준을 결속한다. W1–W4는 current100 전체 R100/P200/N1000, W5는 first500 전체 **R500/P1000/N5000**을 평가한다. W5 current100은 같은 raw에서 추출한다. W1–W4 current 점수를 누적 점수로 표시하지 않는다. Strict NLL preference rate를 primary로 사용하고 token-level success, true/new NLL margin, cohort at-write→W5 retention, W0→W5 neighborhood lost/gained를 저장한다.

주 비교는 기존 BLUE·MEMIT-H·AlphaEdit 완료 산출물이다. BLUE의 실제 L4+L8 및 writer 구성을 표시한다. 입력·평가기·runtime 차이를 명시하며 미결속 결과는 historical reference로 구분한다. CAKE 및 v9는 보조 참고로 사용할 수 있지만 새 baseline fit은 추가하지 않는다. Official P/N은 학습·teacher·계수 선택·후보/arm 선택·stopping에 사용하지 않는다.

## 6. Telemetry·비용·저장

매 후보에 R/q norm, actual v/a 분포와 .75 초과 수/분모, mean/context 실현 share, Q, native 손실, tokens/runtime/solve 정보를 남긴다. 매 batch 후보2/9/25에는 NLL/KL/norm/allocation 각각의 R·q gradient norm, 계수 전/후 값, cosine/radial 및 합 정합을 기록한다. Terminal에는 k^a 대 k^s, v^a 대 Uk^s, base gap, actual loss, commit/history identity를 남긴다. Whole-B numerator/denominator와 KL/rewrite 구분을 보존한다.

기본적으로 aggregate 및 요청/context scalar telemetry를 저장한다. Full W/U, R+P, H, optimizer/RNG resume bundle 또는 동등한 복원 checkpoint는 자동 저장하지 않는다. In-memory writer/gradient 계산과 저장 정책은 별개다. Per-case 평가 raw, log, source/config hashes 및 compact diagnostics는 `local/`에 보존한다. Exact resume가 불가능함을 기록한다.

Q1은100후보/96갱신, main은250후보/240갱신이며 Q2가 main에 포함된다. Calibration 후보0, exact probe0이다. 관측용 backward, suffix/geometry 재계산, 평가 시간을 분리 계측한다. SH3 GPU에서 실제 측정한 시간으로 ETA를 산출하고 이전 server4 시간이나 다른 GPU의 baseline 시간을 가속 배율로 사용하지 않는다.

## 7. 자원·권한·보고

이번 task 동시 GPU 상한은1이며 현재 더 엄격한 project/server cap을 준수한다. A/B는 필요하면 순차 실행한다. 기본 job은1GPU/8CPU/60416MiB host memory로 요청하되 SH3의 현행 scheduler 정책과 실제 memory qualification을 확인한다. Cap을 늘리거나 타 task를 취소하지 않는다. SH3 담당 지정은 SH4의 기존 job 취소·자료 삭제·다른 중단 task 재개 지시가 아니다.

GH는 현재 사용자 명령을 이 신규 task의 scoped 실행 권한으로 게시하고 SH3 직접 수락 ACK를 회수한다. SH3는 코드→검증→pilot→main→report를 맡는다. 새로운 반복 monitor/자동 retry를 만들지 않고 initial main admission 또는 정식 resource-pending까지 필요한 bounded 확인을 남긴다. Sealed runner/collector는 W5까지 자연 진행한다. GH/SH3의 이전 무관한 task 상태는 보존한다.

필수 산출물은 정본 SHA·source/config/input/runtime/evaluator receipt, Q0/Q1/whole-B qualification, Slurm job ID/자원, candidate/gradient/realization logs, per-case 평가 raw, W5 aggregate/retention, runtime/GPUh/memory, factual 한국어 보고서다. 접수·구현·GPU 검증·제출·완료 상태를 구분한다. SH3는 사실과 산술 비교를 보고하고 GH가 과학적 해석을 담당한다.

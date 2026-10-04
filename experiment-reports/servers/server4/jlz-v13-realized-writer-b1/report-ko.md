# JLZ V13 B1 완료·실현 측정 보고

최종 상태: **COMPLETED / B1_FIVE_WRITERS_COMPLETE**. 아래 부록의 PENDING은 제출 당시 snapshot이다.
2026-10-05 사용자 요청에 따라 완료 보고·compact 산출물·코드를 게시한다.

## 1. 실제 실행 범위와 재현 정보

- Task/nonce: `jlz-v13-realized-writer-b1` / `ODEEDIT-USER-GH-SH4-JLZ-V13-B1-REALIZATION-20261005-R1`.
- 실행 source: `082300955e21a2c29218d66d98c5d2c37bc53a20`; tree `3f7f8c28f1adddb798068cf8ea8caa31321ffa2a`.
- 실행 lock SHA256: `06b9fe80e31cd104f0d1045c8f4e2475f0aabc55bef9374e3ed7f6784ae977f5`.
- Server/session: `server4` / `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`.
- Model: Meta-Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.
- CounterFact fixed10k 첫100, seed `20261002`, cold W0/H0, BS100×1, L4–L8.
- V12 planner fit **1회**; 동일 terminal u/D/z/anchors로 RT/RD/MT/MD/CD를 순차 관측. Branch마다 cold W0/H0/RNG/context/cache/ledger로 복원한다.
- 100 unique requests의 5 endpoint이며, 500 unique edits나 5회 fit이 아니다. B2·추가 fit·별도 BS2 순차 pilot·새 baseline·sweep 없음.
- FP32 model/activation/plan, FP64 geometry, eager/autocast/TF32 off. NoCP, exact resume `NOT_AVAILABLE`.
- [설계 및 실행 override](../../../../plans/global/2026-10-05-jlz-v13-realized-writer/execution-b1-sh4.json), [method](../../../../plans/global/2026-10-05-jlz-v13-realized-writer/method.tex).

GPU job `58381`와 CPU collector `58382` 모두 `COMPLETED`, exit `0:0`이다.
원 collector의 cost snapshot에서 collector가 RUNNING인 것은 collector 생성 당시 관측이며,
최종 상태는 [scheduler-terminal.json](scheduler-terminal.json)에 별도로 보존했다.

## 2. W0 및 5 writer 평가

RS/PS는 `new_nll < true_nll`, NS는 `true_nll < new_nll`; ties는 실패다.
TF strict는 해당 target 전체 토큰 정답 여부이며 NLL preference와 다르다.
R/P desired=NEW, N desired=TRUE. Token micro와 prompt macro의 분모도 CSV/JSON에 별도 기록했다.

| Endpoint | RS R100 | PS P200 | NS N1000 | TF strict R | TF strict P | TF strict N |
|---|---:|---:|---:|---:|---:|---:|
| W0 | 5/100 (5%) | 20/200 (10%) | 886/1000 (88.6%) | 0/100 | 0/200 | 171/1000 |
| RT ridge tracking | 99/100 (99%) | 177/200 (88.5%) | 878/1000 (87.8%) | 99/100 | 113/200 | 160/1000 |
| RD ridge direct | 89/100 (89%) | 147/200 (73.5%) | 886/1000 (88.6%) | 79/100 | 81/200 | 169/1000 |
| MT mean equality tracking | 100/100 | 188/200 (94%) | 879/1000 (87.9%) | 100/100 | 129/200 | 165/1000 |
| MD mean equality direct | 100/100 | 188/200 (94%) | 879/1000 (87.9%) | 100/100 | 130/200 | 164/1000 |
| CD context equality direct | 100/100 | 190/200 (95%) | 875/1000 (87.5%) | 100/100 | 131/200 | 166/1000 |

산술 차이: RD→MD는 RS +11pp, PS +20.5pp, NS −0.7pp. MD→CD는 RS 0pp, PS +1pp, NS −0.4pp.
이 차이는 같은 B1 관측값의 산술 차이만 나타낸다.

| Branch | W0-correct N retained | N lost | N gained | W0 대비 NS 차이 |
|---|---:|---:|---:|---:|
| RT | 875/886 | 11 | 3 | −0.8pp |
| RD | 883/886 | 3 | 3 | 0pp |
| MT | 875/886 | 11 | 4 | −0.7pp |
| MD | 876/886 | 10 | 3 | −0.7pp |
| CD | 872/886 | 14 | 3 | −1.1pp |

RD의 NS 분자가 W0와 같아도 개별 이웃의 lost/gained가 0인 것은 아니다.
`margin_true_minus_new = true_nll − new_nll`; 반대 부호 필드는 별도 이름으로만 기록한다.
[metrics.json](metrics.json)에 정확 분자·분모·NLL·TF·paired 값을,
[comparison-B1.csv](comparison-B1.csv)에 endpoint별 수치를 보존했다.

## 3. 실제 작용 및 실현 telemetry

`actual`은 자기층 physical write의 직접 작용이며 `net_entry_change`(entry 대비 최종 hidden 변화)와 다르다.
공통 비교 목표는 planner D다. RT/MT solver의 tracking RHS `T=z_virtual−h_actual`와 D는 별도 대상이다.
Directional ratio=`dot(action,D)/||D||²`, norm ratio=`||action||/||D||`, cosine 및 relative error를 함께 기록한다.

| Branch | Native mean directional ratio 중앙값 | Rewrite-context relative error 중앙값 | Effective Q 층합 |
|---|---:|---:|---:|
| RT | 0.780081 | 0.624681 | 239.826439 |
| RD | 0.635557 | 0.532133 | 143.116857 |
| MT | 0.999803 | 0.242518 | 462.657262 |
| MD | ≈1.000000 | 0.138856 | 471.203667 |
| CD | ≈1.000000 | 0.00000528913 | 581.306832 |

각 scope와 분모:

- Native mean / canonical / KL: 각 500 요청×층 행, 비영 D 404 / ZERO_TARGET 96.
- Rewrite: 3,000 context×층 행, 비영 D 2,424 / ZERO_TARGET 576. Canonical은 rewrite의 부분집합이므로 더하지 않는다.
- 0 목표의 ratio/cosine/relative error는 null이며 0으로 대체하지 않았다. Error norm과 anchor-normalized zero leakage는 별도로 기록한다.
- KL은 관측 context의 종류다. 해당 ratio는 0 KL constraint에 대한 비율이 아니라 그 context의 D 대비 작용이다.
- 작은 D에서 비율이 커지는 실제 행이 있다. 예: RT canonical L5/case55의 D norm `4.881225e−5`, action norm `0.70667679`, directional ratio `5937.60777`. 따라서 평균/중앙값/최대/undefined를 같이 게시한다.

MD native mean의 평균 cosine `0.999999016`, 평균 error norm `1.27224e−6`.
CD rewrite의 평균 cosine `0.999996868`, 평균 error norm `2.29968e−6`.
Ideal / cast / effective FP32 / actual / net-entry 작용을 분리한 값은
[realization-summary.json](realization-summary.json), [CSV](realization-summary.csv)에 있다.
수치 기록은 B1 자기층 작용이며 누적 다배치 측정으로 확장하지 않는다.

## 4. 배분·writer·rank/range

계획 share의 owner 평균(L4→L8)은 `[0.492692963, 0.0295080171, 0.130037632, 0.272188883, 0.0755725052]`이다.
Native mean realized-vs-planned share L1 평균은 RT `0.641171`, RD `0.0798736`, MT `0.0826418`, MD `4.03788e−8`, CD `4.10015e−8`.
CD의 canonical/rewrite/KL share L1 평균도 각각 `4.15659e−8 / 4.10651e−8 / 4.15868e−8`이다.
Share의 `relative_total`은 entry anchor 대비 직접 작용 총량이지 계획 대비 실현율이 아니다.
정확 값은 [shares-summary.csv](shares-summary.csv)에 기록했다.

- RT/RD: native ridge. 최대 relative solve residual 각각 `3.538e−14 / 3.039e−14`, 봉인 기준 `1e−8`.
- MT/MD: 모든 층 rank100/100, `COMPATIBLE_EXACT_WITHIN_TOLERANCE`.
- CD: 모든 층 rank619/700, positive discarded81. Target incompatibility norm `1.05e−13…7.13e−13`; projection residual 최대 `7.014e−13`, 허용 최솟값 `2.547e−9`.
- Rank deficiency와 numerical projection failure는 구분했다. CD의 위 기록은 수치 검산 통과다.
- 25 solve / 25 materialization parity 영수증 확인. Jitter·threshold 재튜닝·ridge fallback으로 equality를 대체한 기록 없음.
- Q는 metric update cost이며 direction/norm ratio와 다른 수량이다. [writer-summary.csv](writer-summary.csv), [JSON](writer-summary.json)에 층별 ideal/effective Q와 rank/cutoff/range/parity/history 값을 기록했다.

## 5. Fit·검산·비용

100요청 모두 candidate0…24의 25회 평가/24회 update 후 `EVALUATION_BUDGET`으로 종료했다.
총 request evaluation 2,500 / request update 2,400; terminal extra forward/backward 모두 0.
Terminal J 평균 `0.07301568359631161`, 최소 `0.05633938176688354`, 최대 `0.20170507439830668`.
Fit 시간 `385.58507775887847초`; [fit-summary.json](fit-summary.json)에 집계했다.

| 항목 | 실제 기록 |
|---|---|
| GPU parent | 58381, COMPLETED 0:0, GPU1/CPU8/58GiB |
| CPU collector | 58382, COMPLETED 0:0, GPU0/CPU8/24GiB |
| Allocated GPU | 1,695초 = 0.470833 GPUh, parent 1회 계상 |
| Runner 측정 | 1,690.005초 |
| Peak VRAM | 40,046,434,304 bytes ≈37.296GiB |
| Peak RSS | 34,718,172 KiB ≈33.110GiB |

Branch writer capture/solve/apply/history 시간: RT155.583 / RD136.355 / MT123.506 / MD137.396 / CD124.511초.
Official endpoint evaluation은 이 writer 시간과 별도다.
`full_adapter_forward=3212`, `native_cached_forward=2508`, `native_full_forward=8`(full에 포함),
layer function first67716/recompute65016은 서로 다른 단위여서 합쳐 whole-model forward라고 표기하지 않는다.

실행 당시 검토는 owner audit, 독립 reviewer 미사용이었다. 완료 후 사용자 recall에서 별도 bounded source/receipt reviewer를 사용했다.
이 완료 검토는 source/scalar/hash 및 원시 metric 행의 대조이며, GPU 재실행이나 RAM-only solver/H Gram의 독립 재계산은 아니다.
정본 CPU16/생산 tiny CPU 회귀와 actual 8B 검산을 구분했다. Actual qualification은 B1 첫2요청의 고정후보 비교 및 각 branch의 cast/local-action/restore 검사다.
6 endpoint×1,300행의 분모·identity·token·margin 검산, 25 exact FP32 weight copy SHA와 final history/eval state 대조, 5 branch 복원 검산에서 확인된 불일치 없음.
각 branch H는 final rewrite-only native mean key 100열, CPUFP32, 각층 append1회(총5회), KL 제외, 동일 H0에서 시작한다.
검토 범위와 원자료 근거는 [postrun-review-ko.md](../../../../audits/servers/server4/jlz-v13-realized-writer-b1/postrun-review-ko.md)에 기록했다.

## 6. 코드·산출물·보존 경계

기존 [실행 코드](../../../../project/run_scripts/jlz_realized_writer/) 12파일은 동결 execution source와 byte 일치하며 변경하지 않았다.
새 `summarize_results.py`는 이미 저장된 결과의 stdlib CPU-only 출판 요약용으로, 원 GPU 실행 코드가 아니다.
Source/config/runtime/native/input lock identity는 [source-closure.json](source-closure.json)에 기록한다.

로컬 원자료: `/data/janghj/ODE-edit/local/jlz-v13-realized-writer-b1/20261005-v1/attempt-r1/`.
`B1/` 원시 평가·full telemetry·로그·receipt와 `cpu-report/`의 8개 원본 출력을 전량 보존했다.
큰 `realization.json`, full `writers.json`, fit-events 및 raw 평가 행은 Git에 복사하지 않고 compact 요약과 기존 원자료 SHA/size inventory를 게시한다.
원 collector report/cost/terminal/inventory는 수정하지 않고 exact-copy 이름으로 게시했다.
[collector-inventory.json](collector-inventory.json), [publication-manifest.json](publication-manifest.json)에 local path/SHA/size 및 게시 목록을 기록했다.
NO_BROADCAST_NOT_REQUIRED: 동일 서버 B1의 compact 산출물만 Git 공유. Raw/model/tensor/prompt/fullstdout/secret 미게시, checkpoint 미저장.
새 실험·재평가·job mutation·타 task 변경 없음. `monitoring_active=false`, `automatic_resume=false` 유지.

## 부록: 제출·초기 인계 원기록

다음 PENDING/NOT_OBSERVED는 등록 당시 snapshot이며 위 최종 상태를 대체하지 않는다.

상태: **SUBMITTED_RESOURCE_PENDING**. 실험 완료 보고가 아니다.

- 권한 nonce: `ODEEDIT-USER-GH-SH4-JLZ-V13-B1-REALIZATION-20261005-R1`
- Authority: `37019ce3ea3b4ae24f8fed29a236b31d102911a8`
- 실행 source: `08230095` (게시 main commit과 구분)
- 실행 lock SHA256: `06b9fe80e31cd104f0d1045c8f4e2475f0aabc55bef9374e3ed7f6784ae977f5`
- SH4/session: `server4` / `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`

## 범위 및 근거

정본 `plans/global/2026-10-05-jlz-v13-realized-writer/`와 B1 override 전체 정독/SHA 검산.
Llama3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, fixed10k first100, cold W0/H0.
fit 1회 후 같은 계획을 RT/RD/MT/MD/CD에서 관측한다. 각 branch W/H/RNG/context/cache/ledger를 RAM 복원한다.
5 branch는 100 unique requests의 5 endpoint이며 500 unique edits가 아니다.
B2·추가 fit·BS2 순차 pilot·새 baseline·계수 sweep·checkpoint 저장 없음.

## 검증 상태

| 항목 | 상태 |
|---|---|
| 정본/실행 JSON SHA 및 source 경계 | PASS |
| first100 native token/lookup | 700행, 원 native와 일치 |
| 평가 token identity | R100/P200/N1000, 1,300행 |
| 정본 CPU reference | 16 PASS; 원본 receipt 불변 |
| production CPU + planner 회귀 | 21 PASS; tiny random CPU Llama 포함 |
| 독립 reviewer | 미사용, owner audit/red 체크리스트 |
| actual 8B GPU 검산 | NOT_OBSERVED |
| B1 fit·5 writer 완료 | NOT_OBSERVED |
| 성능·실현 수치 | NOT_MEASURED |

CPU 결과를 actual Llama parity로 표기하지 않았다.
Qualification은 B1 입력 subset의 고정 후보/연산 비교이며 추가 fit은 없다.
확정 기술 불일치는 고정 tolerance로 차단하고 낮은 성능·range mismatch는 기록만 한다.

## 제출 및 초기 snapshot

| job | 역할 | 자원 | 초기 상태 |
|---|---|---|---|
| 58381 | 단일 B1 fit + 5 writer 평가 | GPU1/CPU8/59392MiB/24h | PENDING (Resources) |
| 58382 | 독립 CPU raw reducer/report | GPU0/CPU8/24576MiB/4h | PENDING (Dependency), afterany:58381 |

두 job 모두 owner/source/fullargv/node/partition/memory/wall/exportNONE/Requeue0/dependency를 held 상태에서 검사한 후 release.
등록 전 SH4 project active GPU0, tracked/local cap3, 본 task cap1. 기존 job 변경 없음.
24h는 요청 wall 상한이지 측정 ETA가 아니다. 초기 snapshot에서 GPU 할당 없음.

## 산출물·중단 경계

원자료 root: `/data/janghj/ODE-edit/local/jlz-v13-realized-writer-b1/20261005-v1/attempt-r1/`.
실행 중/종료 raw는 `B1/`, CPU 결과는 `cpu-report/`에 봉인 프로그램이 기록한다.
CPU collector는 metrics/paired/realization/writer Q/상태·history·비용 검산과 report/inventory 뒤 terminal을 쓴다.
부분 결과는 PARTIAL_OR_TECHNICAL_BLOCKED, 없는 값은 NOT_MEASURED.

정식 resource-pending 초기 인계 후 `monitoring_active=false`, `automatic_resume=false`.
새 polling/heartbeat/자동 retry 없음. 이미 등록된 프로그램만 B1까지 자연 진행한다.
Raw/model/tensor/prompt/fullstdout는 Git 미게시·원본 KEEP.
NO_BROADCAST_NOT_REQUIRED: 동일 서버 task로 compact source/report/manifest만 Git 공유.

구현/감사: `project/run_scripts/jlz_realized_writer/`, `audits/servers/server4/jlz-v13-realized-writer-b1/`.
등록 receipt: `runs/server4/jlz-v13-realized-writer-b1/submission-r1.json`.

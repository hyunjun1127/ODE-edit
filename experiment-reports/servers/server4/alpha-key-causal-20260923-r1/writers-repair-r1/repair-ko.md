# Alpha-key writers 실패 원인과 재실행 준비

Recall: `ODEEDIT-GH-SH4-ALPHA-KEY-WRITERS-REPAIR-20260923-R1`.
정본 `ac44d4d9`와 이후 직접 사용자 수치 비교 관찰 지시를 적용했다.

## 최초 원인과 실제 완료 경계

정확 실패 job은 `52565 / odeedit_alpha_key_writers_s4`, source `f9fbd56f31b0c520763ec9026e660a76cb3074ff`, tree `d7b824052de14288dec559f0cf235a59b8a81f12`다. Lock SHA는 `5c461288fc77ae7071e84b0264cc240cb845b8cb4862ac47ecf37e874fcbc38b`다.

Parent accounting은 FAILED/1:0, 2026-09-23 12:03:04–12:27:58 KST, **1494 allocated GPU-sec**다. 프로그램은 1489.2151초 후 `writer_runner.verify_sham`에서 `SHAM_NUMERICAL_CONTROL_NOT_ESTABLISHED`를 발생시켰다. OOM/시간초과/IO가 아니라 수치 동일성 조건에 의한 중단이며 후속 cleanup exception은 확인되지 않았다.

W50→B51의 NATIVE 및 SHAM은 각각 native solve 5회, history append 5회, 마지막 L8 residual, 8개 stage와 Current/N512/H512 observer 저장까지 완료됐다. SHAM은 이미 계산한 native100 z를 공유했다. H5 이후와 나머지 W70/W80/W90 writer는 미실행이었다.

원 FP32 `M − K_stamp K_stampᵀ + K_stamp K_stampᵀ`는 차감/재가산 roundoff를 낸다. 최초 L4 K/R은 exact이나 Δ 상대 Frobenius 차이는 `4.0618897e-7`; L8에서는 `4.2635394e-6`다. Current NS true-NLL 최대 절대차는 `8.2969666e-5`였다. 이 차이가 관측됐다는 사실과 실행 실패를 보존하며 수치 동등성 PASS로 승격하지 않는다.

## 최신 사용자 지시와 변경

> 이런 상대차는 크게 중요하지 않으니 실험에서 이런 gate들은 전부 관찰만 해

새 immutable attempt에서 SHAM K/R/Δ/NLL 비교와 후속 full-hook/physical 비교를 `OBSERVATION_ONLY_USER_DIRECTED`로 기록한다. 원 비교 verdict와 차이, 기존 threshold=0을 남기며 불일치 때문에 중단하지 않는다. `full_numerical_equivalence=NOT_ESTABLISHED`다. 원 FP32 산술/native solve/P/M/λ/관측집합은 **변경하지 않았다**. 검토했던 보상합산은 구현하지 않았다.

NaN/Inf, shape, 잘못된 input/source/order, 실제 상태 복원 실패, coverage 누락, 저장/자원 오류는 여전히 차단한다. 특히 scalar NLL finite 검사를 추가하여 NaN이 Python max reduction에서 숨지 않게 했다. 기존 제출 source와 실행 중 geometry는 hot-patch하지 않는다.

## 재사용과 재실행

| 대상 | 처리 | 신규 계산 |
|---|---|---|
| 52563 실제 gate/G1/timestamp | identity/READY 재사용 | 모델 gate 재실행 0 |
| 52564 geometry | 기존 실행 그대로 | 중복 job 0 |
| W050 NATIVE native100/5 writes/history/observer | 원 input CP+저장 factor 결속, RAM 재구성 | fitting·평가 0 |
| W050 SHAM 5 writes/history/observer | 동일 재사용, 차이는 관찰 전용 | fitting·평가 0 |
| W050 H5/H6/H56/MASS56 | 누락 실행 | z fitting 0, branch K/R/solve 새 계산 |
| W070/W080/W090 각 6 branch | 누락 실행 | native100씩 총 300 target, branch별 downstream 계산 |
| 승인 component/KR | 원 고정 범위 실행 | 추가 z 0 |
| 원 CPU reducer52566 | 그대로 보존 | 원 attempt 보고 경로 유지 |
| 새 통합 CPU reducer | geometry52564와 새 writer afterany | 혼합 provenance·부분실패까지 보고 |

완료 branch+gate 270개 파일을 새 SHA/size로 결속했고, 기존 native/diagnostic artifact SHA와 대조했다. CPU에서 승인 B050 CP와 저장 Δ/history key만으로 NATIVE·SHAM×L4–L8의 **post-W 10/10, post-M 10/10 SHA exact**를 확인했다. 저장 K stride는 `[1,14336]`로 원 transpose layout을 유지했다. 실제 새 GPU reconstruction은 아직 미검증이며 runtime에서도 각 pre/post hash와 selected seal을 다시 검사한다. 새 W/M checkpoint는 쓰지 않는다.

원 post-z RNG snapshot은 저장되지 않았다. pinned `compute_z`의 zeros/Adam/eval·비확률 source와 exact entry RNG를 결속해 재사용한다. 이를 과거 post-z RNG 직접 관측이나 완전한 checkpoint resume으로 주장하지 않는다. 각 branch의 entry 복원 및 원 source binding은 차단 검사로 유지한다.

## 검증·자원·비용 경계

CPU156 PASS(4.515초), actual 저장 SHAM 관찰정책 재검산, diff/compile 검사를 수행했다. 최초 test 호출은 GPU-hidden 환경 표기 누락 때문에 token-binding fixture setup 1건이 실패했고, 정확한 CPU-only 환경으로 재실행하여 통과했다. 모델 실행 실패와 구별한다. 별도 read-only red를 사용했으며 NLL finite·history stride 우려를 수정/검산했다.

재사용 native/observer의 원 비용은 52565의 1494 GPU초에 포함되며 새 forward로 세지 않는다. Gate52563의 196 GPU초, 더 이전 실패비용, 실행 중 geometry allocation과 새 attempt는 각각 구분한다. Allocation은 utilization이 아니다.

Project/task cap2: 기존 geometry 1GPU+새 writer 1GPU 이하. 새 writer는 1GPU/8CPU/60416MiB/exportNONE/Requeue0, 7일 운영 wall 예약이다. 7일은 과학 예산이나 실측 시간이 아니다. CPU reducer는 0GPU/4CPU/8192MiB다. 기존 자원 상한을 올리지 않는다.

남은 저장 계획은 실제 두 branch 파일 크기, 남은 22 branch, geometry 고정 tensor payload, component/KR, source/atomic 임시공간 및 별도 20GiB 공유-volume 안전분을 포함한다. 이전 전체작업 50GB margin은 역사로 보존하며 새 scoped 잔여계획으로 대체한다. 준비 시 free 165,991,739,392B, 미래 필요 164,178,341,976B였다. 진행 중 geometry가 이미 쓴 bytes는 현재 free와 미래 필요에 이중 계산하지 않도록 제출 직전에 footprint만 갱신한다. 독점 reserve나 사용자 storage waiver가 아니다. 원자료 삭제·output 축소 0.

## 등록 및 종료

최종 job/source/lock/상태는 후속 `handoff.json`에 기록한다. GPU 부족이면 정상 held-inspection/release 후 pending 근거를 인계하고 STOP한다. 실행 가능하면 W50 reused branch 복원→새 H5 완결→entry restore의 수리 경로 초기 정상성만 확인하고 STOP한다. 전체 완료를 기다리지 않으며 이후 자동 polling/retry/agent 재개는 없다. 선등록 프로그램과 CPU reducer만 자연 진행한다.

원 실패 source/raw/12 input CP KEEP. 기본 새 full-state checkpoint 미저장, 승인 진단factor만 저장. SEQ/ORDER/FUTURE는 `FOLLOWUP_NOT_SUBMITTED`. `NO_BROADCAST_NOT_REQUIRED`. 전용 branch 소형 source/사실 보고만 게시하며 main 통합은 GH가 수행한다.

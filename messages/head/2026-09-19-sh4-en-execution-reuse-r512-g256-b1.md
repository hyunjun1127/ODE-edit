# GH → SH4: EN 실행 중복 제거 / R512-G256 / cold SINGLE BATCH

Instruction ID: `ODEEDIT-S06-EN-EXECUTION-REUSE-R512-G256-B1-SH4-V1`
Nonce: `ODEEDIT-GH-SH4-EN-REUSE-R512-G256-B1-20260919-R1`

## 1. 사용자 권한과 이번 종료 지점

사용자 원문: “/mnt/raid5/janghj/.codex/worktrees/odeedit-en-execution-reuse-20260919/plans/global/2026-09-19-en-execution-reuse-r512-g256-design-v1.md 이 설계문 자세히 읽고 SH4에게 TASK 전달하자. 먼저 SINGLE BATCH로 실험해보는 것 까지 지시하라.”

GH는 해당 worktree의 설계·contract·설계검사 JSON 원 bytes를 이 publication에 보존했다. 아래를 전체 읽고 exact SHA로 결속한다.
- `plans/global/2026-09-19-en-execution-reuse-r512-g256-design-v1.md`
- `plans/global/2026-09-19-en-execution-reuse-r512-g256-contract-v1.json`
- `audits/global/2026-09-19-en-execution-reuse-design-checks.json`
- 설계에 연결된 EN-F 완료 보고·compute.csv와 실제 source/alltoken/binding/runtime/runner/sequential_runner/geometry/optimizer, 상속 EN native·geometry·finite 설정, PROTOCOL.
- 근거 source `3a399b93b1a4ff14909c23f029655f5c0ef5bba5`; 실제 새 execution source와 분리한다. Design check는 source/arithmetic 확인이지 CPU runtime 또는 actual Llama 검증이 아니다.

**승인: 구현 → CPU 및 실제 모델 검증 → 필요한 R512/Dev128 generated256 준비 → 동일 cold B100 한 batch의 legacy/optimized 비교 → 상세 factual 보고/main 게시까지.**
기존 O0 first100, W0/zeroM4, 한 동일 native WN·K_E·Q_E를 사용한다. 명시적 신규 정책/sweep/다른 batch 없음. `max_batches=1`, `sequential_authorized=false`, `auto_continue=false`를 코드·실행 lock·CPU 회귀검사에 넣는다.

**어떤 결과여도 B2 이후/sequential을 제출하거나 실행하지 않는다.** Held/pending/afterok 예약, 자동 callback/재개도 금지한다. B1 보고 후 `WAITING_USER_APPROVAL_FOR_SEQUENTIAL`로 종료한다. 향후 sequential은 새 사용자 허가가 필요하다. BPCW B1 실패 gate는 이 새 EN 효율 구현의 선행 PASS 조건도, 자동 확대 권한도 아니다.

이번에는 초기 gate만 보고 중지하지 않고 **single-batch 비교·최종 보고까지** 진행한다. 기존 EN T-skip/FD-skip/noCP/storage waiver/취소·삭제 권한은 상속하지 않는다. 현재 설계에 필요한 실제 검증은 생략하지 않는다. 별도 대규모 T/M campaign도 만들지 않는다.

## 2. 비교를 섞지 말 것

이 task는 EN-F의 실행 중복 제거와 R512/generated256 data adapter를 구현한다. BPCW choice-QP 또는 GSS 방법이 아니다.

- Primary matched pair: **LEGACY_SCHEDULE_R512_G256**(새 데이터 adapter+기존 중복 실행 순서) 대 **REUSE_SCHEDULE_R512_G256**(같은 adapter/loss/native+endpoint/hidden/score reuse).
- 두 경로 모두 **같은 R512 모든 문서·모든 실제 생성 위치·전체 vocabulary**를 사용하며 각각 gradient sweep과 trial 계산을 독립 실행한다. Timed arm 사이 method gradient/후보/결정 결과를 공유하지 않는다.
- Native fit/teacher/input/불변 geometry는 정확 identity를 결속해 공유하고 실제 연구비용에는 한 번만 계상한다. 새 cold native가 필요하면 **B100 한 번만**; 이전 정확 native가 유효하면 읽기 전용 재사용 여부/원 실행·시간과 이번 비용을 분리하며 새 native 실행으로 오기하지 않는다. 다른 native/entry를 조용히 대체하지 않는다.
- 기존 S64/corpus128은 historical regression fixture이지 primary timed arm이 아니다. S64 과거시간 대 R512 현재시간을 dedup speedup으로 쓰지 않는다. 자료가 삭제돼 없는 과거 EN checkpoint를 복원 실험으로 재생성하지 않는다.
- N4는 공유 native endpoint의 기준 행으로 평가할 수 있으나 별도 native chain/추가 cold batch는 만들지 않는다. 동일 endpoint 관측만 identity 검증 후 중복 없이 재사용한다.
- Pure dedup 비교는 **같은 candidate FP32 bytes/판정/selected endpoint**를 우선 요구한다. 원 보호 tolerance 통과를 구현 동등성 tolerance로 쓰지 않는다. 불일치는 원인/차이/영향을 별도 기록하고 speedup PASS로 포장하지 않는다.
- Head chunking, GPU FP64 gradient 누적, generation KV는 필요시 각각 분리된 수치 검증/identity/비용을 요구한다. Optional 최적화를 무조건 동시에 넣지 않는다. 필요한 chunked adapter가 있으면 두 primary schedule에 같은 adapter를 적용하여 dedup 효과만 비교한다.

## 3. 고정 방법과 데이터

EN-F 수학/optimizer 수치 정책을 그대로 둔다: 실제 immutable FP32 WN, all-valid-token K_E, allowed-space와 edit-null 교집합 Q_E, G=grad L(WN), H=GQ_E, chi, eta0=L/chi, 최대8trial·반감. **전체 gradient sweep 최대1회/arm**이고 이는512문서 각각 backward를 포함한다. 첫 Armijo+Current/Past+invariant 통과 수용 또는 기존 fallback. max trial을2로 줄이지 않는다. 새 z/추가 layer/history objective/recency weight/alpha sweep/GSS reference 축소 금지.

정확 R512 input SHA `507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb` 재사용. 추가 문서선정/미래 요청·공식 P/N로 filtering0. Dev128은 별도 observer, Report256 미개봉.

새 data ID `EN-R512-G256-v1`: prompt BOS+128 자연token=129 IDs. W0 raw argmax/lowest-token-ID tie/no processor/max_new_tokens256/실제 configured EOS 조기 종료. fake EOS·최소길이 강제·짧은 문서 제외0. Old16/128 capsule·natural-prefix full teacher를 새 generated256 teacher로 이름만 바꿔 쓰지 않는다.

각 문서 실제 T_i, TF input [x,y0[:-1]] 길이129+T_i−1, 점수 위치128..128+T_i−1; T_i256이면 TF384/fullgeneration385를 확인한다. W0 TF teacher [T_i,V] FP32 logp와 generation/input/tokenizer/model/mask/position/hash 결속. 모든 완성 capsule에서 canonical TF argmax=y0 확인. mismatch 문서 삭제/label 교체로 통과하지 않는다. 완료/partial store를 구분한다.

목적은 full-vocab forward KL의 **각 문서 실제 T_i 평균 →512문서 평균**, signed reduction/gradient 누적 순서 유지. Last short chunk에 동일 chunk 가중치를 주지 않는다. 새 GeneratedTeacherStore를 쓰고 old TeacherStore strict schema를 wildcard로 완화하지 않는다. W0 behavior anchor를 factual truth라고 부르지 않는다.

## 4. 실행 재사용 구현의 핵심

- EndpointSession은 CPU immutable owner/SHA/shape/dtype/finite와 GPU handle을 결속. WN+현재candidate 2slot, endpoint당 한번 H2D. Gradient leaf/graph·physical 검사 전송은 별도. Pointer/version만으로 외부 alias mutation 검증을 대체하지 않는다.
- Native/Candidate ObservationBundle은 정확 endpoint/input/label/position/model epoch/teacher/trial/coverage를 포함. Current final-normalized hidden은 detach CPU FP32, rows를 재사용. 이전 weight/batch/candidate/partial observation을 잘못 hit하지 않게 하고 예외 시 해제한다.
- 첫 dedup 단계에서 guard target-position head shape와 invariant valid-token16-position chunk를 유지. Hidden/score는 공유하되 head GEMM shape를 조용히 합치지 않는다. 독립 physical 검사 경로는 optimized cache를 우회한다.
- 순서: CPU materialization/cheap checks → 전체 R512 objective → Current/Past guard → actual geometry/full-token invariant →첫 통과 수용. Armijo 탈락 전후 검사순서를 변경하지 않는다.
- Geometry 고정 native/key 통계는 **동일 dtype/backend/block/reduction route**에서만 캐시. Torch proposal과 NumPy/SciPy invariant scalar를 수식만 같다는 이유로 대체하지 않는다. 실제 candidate DK/leakage/logits/NLL은 후보별 검사.
- Reference key/residual은 입력·model·teacher identity가 맞는 고정 upstream만 공유. 다른 endpoint의 downstream hidden/gradient/Q는 공유하지 않는다. Full bank 참여와 resident chunk 수를 구별한다.
- 기존 Past64 own-WN/version/overwrite/개별 NLL·ID 조건 유지. B1은 actual Past 없음/N/A이며 nonempty history correctness는 CPU fixtures/구조검사로 표시하고 실제 B2 검증을 주장하지 않는다.
- Commit당 history1, 후보/observer append0; W/M/RNG/context/order/registry atomic checkpoint와 restore. 선택과 observer 분리.

## 5. 검증·계측·실패 경계

CPU tests: endpoint transfer count, CPU/GPU mutation·alias·epoch·pack/teacher invalidation, partial/stale graph 차단, legacy/cached NLL·strict/pair·max/RMS, EOS1/중간/256·shift·chunk 경계, R512 누락/중복/Dev혼입, no-update/fallback/거절/emptyPast 수명, B2차단, corruption/OOM/evidence error의 technical 처리.

Actual 같은 cold batch에서 loss rows/분모/G/H/eta0/각 trial FP32 SHA/Armijo·quality·invariant/첫선택·fallback·history/restore/coverage 비교. 전체512와 실제 위치 gradient 참여 counter를 남긴다. Physical/cached 독립 parity와 실제 FP32 보호반응을 확인한다. CPU math나 과거 Tskip을 실제 Llama PASS로 재사용하지 않는다.

시간: setup W0generation/teacher/cache, native z/key/write, endpoint validation/H2D bytes/calls, reference suffix/head/loss/backward/teacher I/O, current/past suffix/head/guard/invariant/hit·miss, geometry, hash/event/CP, observer/audit를 분리한다. _weight 외부 upload와 wall/CUDA 경계를 기록한다. 중첩 timer를 합산하지 않는다. 4C→C는 해당 suffix 호출 구간이고 총4배속 주장이 아니다. B1 1회로 p50/p90/안정적 배수 주장을 하지 않는다.

기술적 오류는 실제 exception/source-backed RCA·원 source/raw/비용을 먼저 보고·보존하고 동일 B1 범위 최소수리/새 immutable attempt로 검증한다. 유효한 no-gain/fallback/느린 결과는 기술오류로 rerun하지 않는다. 보호기준·수식·bank·trial·precision 정책 변경이 필요하면 명확히 보고하고 임의 변경하지 않는다.

## 6. Server4 경계·자원

Target server4/session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, expected CWD `/data/janghj/ODE-edit`, repo `hyunjun1127/ODE-edit`.
GH `01a04939-8873-7673-8dca-4c7fc5e31af0`.
새 clean `codex/server4-en-execution-reuse-r512-g256-b1-v1` worktree, shared root/dirty와 기존 EN/BPCW 실행 source/raw·체크포인트 보존.

Server4 **project GPU cap2**, 이 matched 비교는 동일 GPU/runtime에서 **task 동시1GPU/1job**로 두 schedule 순차 실행을 기본으로 한다. Preparation도 task1을 넘기지 않는다. 1GPU/8CPU/host≤60416MiB/exportNONE/Requeue0, walltime은 새 setup·parity·두 schedule·observer 추정으로 실제 제출 전 lock. 사용자 GPUh hardcap 없음. 다른 task의 resource-only 점유를 pre-admission에 확인하되 scientific log/결과 monitoring 재개·cancel/hold/삭제 권한은 없음.

R512+Dev128 full teacher 상한78.28125GiB, key/residual16.875GiB, 합95.15625GiB는 **payload일 뿐**이다. CP/native/geometry/temp/atomic write/host/graph 여유까지 포함한 실제 저장·메모리 preflight를 준비 전에 수행하고 EOS actual T로 갱신한다. mmap/document streaming을 쓰되 fullvocab/문서/위치 축소 금지. 기존16-token reserve/EN삭제/storage waiver 상속0. 실제 공간이 부족하면 정확 필요량/여유/가능한 합법 대안을 보고하고 보존 데이터를 임의 삭제·이동하지 않는다. 신규 원격 대량 전송은 별도 exact 승인 없이 수행하지 않는다.

Slurm 본 B1 구현 검증/teacher 준비/두 schedule 실행은 이 envelope로 승인. Held owner/source/args/node/resources 검사→release, 실제상태와 CPU/준비상태 분리. 조건이 가능한 승인 단계마다 반복 승인 요청하지 않는다.

## 7. 쓰기 권한 및 원본 보존

새 코드/adapter/runner/launcher/tests/reducer는 `project/run_scripts/en_execution_reuse/**`를 우선 사용한다.
설계상 필요한 hook만 다음 기존 파일의 **dedicated branch 새 버전**에 구현할 수 있다:
`project/run_scripts/single_layer_edit_preserving_correction/{alltoken.py,binding.py,runtime.py,runner.py,sequential_runner.py,geometry.py}`.
같은 namespace 신규 `endpoint_session.py`, `observation.py`, `generated_teacher.py`, `generated_reference.py`와 해당 task tests 허용.
`optimizer.py`는 optional observation context 전달만 필요한 경우 legacy callback 기본경로를 보존해 narrow 변경 가능; 수치/판정/trial 정책 변경 금지. 기존 native AlphaEdit/BLUE/model/evaluator/TeacherStore strict schema와 global 설계 수정 금지. Old frozen executable/archive/local code는 덮어쓰지 않는다.

본 task 전용:
- `plans/updates/server4/en-execution-reuse-r512-g256-20260919-v1/**`
- `audits/servers/server4/en-execution-reuse-r512-g256-20260919-v1/**`
- `experiment-reports/servers/server4/en-execution-reuse-r512-g256-20260919-v1/**`
- `messages/acks/server4/2026-09-19-en-execution-reuse-r512-g256-b1.md`
- `messages/server-heads/server4/2026-09-19-en-execution-reuse-r512-g256-b1.md`
- `tasks/status/en-execution-reuse-r512-g256-b1-20260919/server4.json`
- `runs/odeedit_en_reuse_r512_g256_b1_s4_20260919/**` compact-only explicit allowed
- ignored `local/en-execution-reuse/20260919-v1/**`.

Preflight/postrun data/eval/firewall/parity/resource/source/legacy-regression/history/readonly/checksum及raw-free 감사. 필요한 복잡한 독립 구현/검산만 bounded subtask로 분담하며 단순 상태조회 agent 남발0. red block은 근거를 보고하고 임의 waiver0. Nonforce source/compact report branch push 및 최신 main own-scope integration을 승인한다. Conflict면 보존·보고, 공용 Git role/session/model 설정 변경0. raw/tensor/teacher/prompt/fullstdout Git0. Same-host `NO_BROADCAST_NOT_REQUIRED` 이유 기록; 강제 broad copy0.

## 8. 중간보고·최종 산출물

FULL_READ/M0에 확보한 source, 미구현 모듈, 기존 input/신규 teacher 구분, primary 두 schedule, single B100/명시 B2차단, 실제자원 및 시간 추정을 남긴다. 이후 구현 readiness/source freeze, 제출/job, actual parity, B1 첫 표와 최종보고를 전달한다. GH 중복 raw/GPU audit를 선행조건으로 요구하지 않는다.

최종 `experiment-reports/servers/server4/en-execution-reuse-r512-g256-20260919-v1/b1/diagnostic-report-ko.md`:
- 동일data legacy↔reuse의 loss/gradient/trial/선택 exactness, 불일치·미적용 최적화
- 전체512/각 T_i/전체vocab coverage, EOS/censor·W0 capsule/teacher provenance
- Shared N4와 각 sealed endpoint Current R100/P200/N1000·strict/NLL·paired ID, Dev observer·학습 KL 구분
- 전송/forward hit/miss·actualtiming/memory/storage, setup/standalone/research total/실패비용·중첩 미계측
- source/config/runtime/sample/checkpoint/largeartifact inventory 및 재현방법
- B1 Past N/A, 구현동등성·기술검증 범위와 NOT_TESTED/NOT_RECORDED.

표·링크·실제 렌더 가능 여부/코드그림·manifest/독립 reducer를 검산하고 own-scope main 게시한다. SH는 사실·수치와 계약의 기계적 PASS/FAIL만, 과학효능·우월성·후속선택 해석은 GH가 별도로 한다. **single batch 완료 후 종료, sequential 신규등록/예약/자동재개0, 사용자 별도 승인 대기.**

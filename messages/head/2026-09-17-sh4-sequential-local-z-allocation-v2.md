# GH → SH4: Sequential Local-z Allocation v2 — 여섯 cold arm

instruction_id: ODEEDIT-S06-SEQUENTIAL-LOCAL-Z-ALLOCATION-V2-SH4-V1
nonce: ODEEDIT-GH-SH4-SEQUENTIAL-LOCAL-Z-ALLOCATION-V2-20260917-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 1. 최신 사용자 원문·전환 경계

사용자 원문:

> /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-17-sequential-local-z-allocation-design-v2.md
> 이 실험 SH4에게 진행시켜라. 이전의 repair 실험은 motivation을 충분히 반영하지 않는다고 판단했다. 자세히 파악해보고, SH4에게 중간보고도 받으며 task 진행시켜라. 모니터링 자체는 초기 gate가 통과하면 모니터링 멈추는 것으로 명령해

가장 최근 자원 지시 “cap 2로 늘리자”를 유지한다. Server4 project 합계 GPU cap은 **2**다.

이번 신규 v2가 앞으로의 실행 우선순위다. 이전 ODEEDIT-S06-L4-PRESERVING-REPAIR-TWOARM-SH4-V1의 R-GD/R-QP 후속 구현·본실험 제출·추가 pilot·분석은 재개하지 않는다. 그 source/raw/사용자 no-checkpoint override는 해당 task 기록으로 보존한다. GH의 direct 상태 확인에서 이전 기술 pilot49238은 마지막 PENDING, 본실험 두 arm 미등록이었다. 이것은 현재 scheduler 상태 확인이 아니다.
새 admission을 위해 정확49238의 상태/자원 점유만 한정 확인하고, active 또는 실행 가능한 pending이면 cap2에 포함한다. **취소·hold·재시작을 이번 사용자 문구에서 추론하지 않는다.** 이미 제출된 pilot은 그대로 두고 새 main이 기존 작업과 합계2를 넘지 않게 한다. 기존 job 변경이 필요하면 정확한 상황만 보고한다. 기존 repair 종료를 기다리는 모니터링을 만들지 않는다. 새 v2에는 repair 방향/QP/P8 없는 repair runtime을 가져오지 않는다.

별도 clean codex/server4-sequential-local-z-allocation-v2-v1 worktree/branch와 새 local namespace에서 진행한다. 최신 main 및 타SH/GH 변경을 보존한다. 사용자 shared dirty/root/이전 main과 원 native helper를 수정하지 않는다.

## 2. 정본 FULL_READ 및 실제 구현 경계

다음 원본을 전체 읽고 exact bytes를 local authoritative에 봉인한다. GH는 이 문서들 및 CPU source를 읽고 전달하지만 실제 Llama 검증이나 새 GPU 실행은 하지 않았다.

- plans/global/2026-09-17-sequential-local-z-allocation-design-v2.md — SHA fac114d403502c96cbcb33a58b0ff1ee2b0b65b017c44b22a2a22adcda749b73.
- companion contract-v2.json — SHA a7aaac67ffe1f6fb47d372b1eaea62d33d4cf12fcb9e0874fbe92d2750239f8c.
- companion cells-v2.csv — SHA 92f51ceffa031a5373ef782e0d08587c5c4634590932ed646c1099b7630ac9ad.
- audits/global/2026-09-17-sequential-local-z-allocation/의 controller_reference.py, test_controller_reference.py, validate_design.py, cpu-validation.json, design-checks.json, convergence-evidence-ko.md, convergence-evidence.json.
- companion dispatch/source-input-manifest.json, 최신 PROTOCOL/등록 session/cap 정책, 본 envelope.
- cold7 execution32a92ad6f3fff2f258d8778f3936d152e975ac1b의 실제 fitter/model/engine/selector/transaction/observer, 원 BLUE AlphaEdit_main/compute_z/compute_ks/native write/history.
- GH cold7 독립리뷰 prose는 이미 main의 plans/global/2026-09-17-l4-preserving-repair-sh4-dispatch/references/cold7-independent-review-ko.md에 exact mirror로 있다. 본 v2가 그 문서의 과거 제안 및 이전 repair 설계보다 우선한다.

원문에 있는 GH 로컬 경로는 source-input-manifest의 원본→repo relative mapping으로 읽는다. 별도 GH worktree가 SH4에도 존재한다고 가정하지 않는다. 이전 input/raw inventory와 small receipt를 우선 재사용하며, 기존 cold7 전체 raw/GPU를 동기 점검 명목으로 재계산하지 않는다. 삭제된 cold7 CP21개를 존재한다고 쓰거나 재생성하지 않는다. W0 cold capsule와 model/P/teacher/input은 별도로 결속한다.

현재 제공 코드는 CPU controller 참조이며 actual model runner가 아니다. 새 versioned adapter에서 L4–L8을 지원한다. 기존 NativeSingletonFitter의 4/8 제한을 기존 파일에 직접 패치하지 않는다. Source/config/import/dependency/hparams/model/teacher/sample identity를 실행 전에 잠근다. Python/SciPy 관측값을 현재 값으로 추측하지 않는다.
Optimizer는 SciPy1.15.3 legacy COBYLA이며 shared environment를 업그레이드하지 않는다. 필요한 경우 task-local 환경만 사용한다. 공식 1.15.3 문서의 maxiter는 function-evaluation 상한이고 tol은 trust-region 하한이다: https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-cobyla.html . 이 둘을 loss 허용치/학습 수렴 보장으로 바꾸지 않는다.

## 3. 실행 범위·같은 cold capsule

신규 scientific allowlist는 **N4, F48, G48, C4, C48, C45678**뿐이다. 각 pretrained W0/zero M4..M8에서 fixed first1000 B100×10. 여섯 chain/60 batches/6000 arm-request observations, unique1000. 이전 cold7 N4/LD/REFIT4는 motivation/historical reference이며 이번 paired 주표의 새 arm을 대신하지 않는다.

| Arm | 정책 | 최종 commit |
|---|---|---|
| N4 | local L4 full | 원 native endpoint |
| F48 | local4 .75 → 실제 prefix local8 .5 | raw fixed, 품질 guard는 관측만; N4 fallback 금지 |
| G48 | a4={.75,1} × a8={0,.5,1} | v2 guard/selector로 새 실행 |
| C4 | 연속 a4 | v2 feasible pool |
| C48 | 연속 a4/a8 + exact-zero pruning | v2 feasible pool |
| C45678 | 연속 a4..a8 + exact-zero pruning | v2 feasible pool |

같은 model revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eager, matmul/cudnn TF32off, seed20260916, native context text·token IDs·tokenizer/padding/microbatch를 공통 봉인한다.
S4 전용 fixed10k 경로 /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1, 모델 load 전 load_prefix/verify.
dataset SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
whole order5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729,
first1000root40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd.
Ordinal[0,1000), shuffle/filter/성공기반 제외/다른 batch순서0. Warm checkpoint 반입0.

S64 W0 full-vocab teacher와 Dev128 observer를 분리한다. Teacher manifest f81b798f44ce626ac1e2e402ca7438363b1dd60f5681ec0ac17b9e92d096761a는 후보이며 실제 cold7 effective model/token/position/scoring 연결로 재사용 판단한다. 파일명 일치만으로 PASS하지 않는다. 불가하면 같은 locked IDs/tokens의 teacher를 공통 W0에서 한 번만 준비하고 비용을 분리한다.
Past64는 받은 사실 중 latest-active, current overwrite 제외, 고정 LZ-ALLOC-v1|20260916|past|stable_event_id 우선순위다. Future 접근/성공기반 selection0. 새로운 paraphrase 생성·훈련·online guard0; 공식 P/N은 선택 후 observer만이다.

## 4. 변경하면 안 되는 방법 계약

매 batch own We에서 L4 native fit 한 번으로 D4와 full WN을 확보한다. 앞 layer gate를 실제로 materialize한 상태에서 후속 layer의 local target/key/solve를 새로 계산한다. Native lr.1/decay.5/clamp.75/KL.0625/L2=1, 최대25 loss/24 Adam 및 기존 early stop/source 연산은 불변이다. 25→50/100, warm-start target, entry z8 bank, terminal-z 배분, joint target/gate learning0.
Gate는 각각[0,1], 합1제약0, a4=0 연속 경계 허용. 정확0은 해당층 fit/solve skip, 정확1은 native endpoint copy, 그 사이는 FP32 U+a(V−U). 작은 gate를 threshold로0에 snap하지 않는다. 중간 prefix의 낮은 품질로 완성 후보 계산을 자르지 않는다.

Cache는 model 실제 상태/layer/M/P/request token/context/source/hparams/RNG를 포함한다. 앞 gate가 바뀌면 필요한 suffix target/key를 무효화한다. 같은 prefix의 fit만 해당 layer의 여러 gate에 공유한다. P4..8 physical→asset mapping을 tensor로 확인한다. 이전 L4observer index metadata 오류를 복사하지 않는다.

N4 자체를 incumbent로 두고, 완성 후보의 E≤E(WN)+1e-4, H≤H(WN)+1e-4 및 Current/Past TF-strict·new<old pair 성공 ID subset, finite를 확인한다. E는 native rewriting token→context→request 평균, H는 Past canonical 평균이다. B1 Past guard는 비활성, old target 없을 때 pair만 비활성/분모를 공개한다. **.05 plateau 없음**. Canonical Current mean/개별 NLL/PS를 새 online guard로 추가하지 않는다.
Global feasible Bmin+1e-6 tie set을 매번 계산하여 exact own N4→적은 실제 nonzero layer→작은 concatΔW norm→gate lexicographic→endpoint SHA 순으로 선택한다. Epsilon 열화 누적0, solver 반환점 자동 commit0. Own N4는 과거 다른 layer update를 포함한 We의 full L4이며 독립 N4 chain으로 되돌리는 것이 아니다.

COBYLA u=1−a, u0=0(all full), rhobeg.25/tol.01/catol1e-8/maxiter64/dispFalse. Objective B/max(B_N4,1e-6), E/H slack 및 pair margin scale.05, strict margin scale1을 유지한다. .05는 수치 단위이지 품질 여유가 아니다.
Raw u bounds를 변환 전에 검사하여 out-of-box는 finite dummy1/위반 constraints만 반환, GPU0/clip0/품질후보0/incumbent0. 동일 x의 objective/constraints는 같은 관측을 공유한다. Strict/pair 최종 실제 ID 검사를 유지한다.

연속 탐색 extra suffix fit search32+prune reserve8=40, scored endpoint24+4=28, extra Adam9600, 호출64; native N4 fit100target/2400Adam/1solve/1score는 별도. Suffixfit 전에2400 worst-case reserve를 확인하고 실제 Adam만 차감, 잔여 반환. 중간 request를 budget 때문에 자르지 않는다.
불완전 suffix는 이미 쓴 비용/cache만 보존하고 We로 복원하며 score/feasible 후보에서 제외한다. Search가 pruning reserve를 빌리지 않는다. Adam pruning reserve는 별도로 없으며 미완료를 기록한다.
C48/C45678은 incumbent에서8→7→6→5 추가층을 한 번씩 exact0로 검사하고 필요한 suffix를 새로 계산한다. L4 pruning0(연속경계0과 구분). 모든 실제 후보는 같은 global selector를 사용한다.
Budget stop/solver 미수렴은 incumbent 선택 가능한 정상 상태이고, NaN/Inf/OOM/shape/cache/hash 오류는 기술 실패다. 후자를 N4 fallback으로 감추지 않는다.

모든 여섯 arm은 M4..M8를 유지한다. 최종 selected endpoint의 전체 Current key로 layer별 정확히1회 append, 후보 중0. Gate0/zero-step도 제외0, gate가중0, 과거 전체 key refresh0. 총300 appends는 계획상 수치이며 실측 검산한다. N4 미사용층 history 비용도 분리한다.

## 5. 기술 준비·중간보고·승인된 제출

CPU 참조 및 새 adapter/실제 solver에서 설계의 cache/budget/rollback/IDs/global tie/pruning/history 계약을 확인한다. 원 CPU20+SciPy3과 문서66checks는 재사용된 synthetic/설계 근거이지 실제 Llama PASS가 아니다. 기존 validate_design.py의 GH 절대경로·출력 덮어쓰기 동작을 그대로 실행해 정본을 변경하지 않는다. 필요하면 새 task-local 검증 harness에서 경로만 대응하고 기록한다.

필수 actual W0 기술 셀:
- local4(1)↔native N4; local4/8 all1↔same-entry native BLUE; local4..8 all1↔동일 layers native BLUE 분기.
- gate0/cache 및 같은 fixed 후보 replay, teacher/repeated score, generalized P/M mapping, 실제 nonselected guard/rollback와 selected whole-batch commit/history5/다음 entry 연결.
- E/H 반복차≤5e-5, B≤5e-7, strict/pair ID exact. 실제 효과를 보고 허용치 완화0.
- C45678 completed search gate vectors(N4제외)≥d+2=7 및 distinct a4≥2. 동일 endpoint 캐시 gate와 actual unique endpoints를 분리. 운영 coverage proxy이지 solver simplex/최적성 인증 아님.

W0 기술 coverage 부족은 INSUFFICIENT_SEARCH로 보고하고, 설계 §7의 계산가능성 범위에서 새 budget/search 구현 version을 성능 튜닝 없이 먼저 봉인한다. 임의 quality/native/arms 변경은 하지 않는다. 원 상한을 넘는 실질 자원 확대가 필요하면 정확한 모순/추정을 보고하여 지시를 받되 독립 구현 작업은 계속한다. Lifelong 도중 coverage 부족은 결과로 남기고 같은 run의 예산을 늘리지 않는다.

**중간보고를 GH에게 direct로 보낸다.** 중복 GH raw 점검이나 단계별 “진행해도 되나”는 선행조건이 아니다.
1. FULL_READ/M0: 새 v2와 repair 차이, 실제 old49238 admission, source·미구현·기존 자산 재사용,6arm/cap2/저장·시간 추정/모니터링 경계.
2. CPU/기술 준비: source/import/수치 계약, 실제 Llama 검증과 C45678 coverage, 첫 실측 native fit·candidate/teacher/history/observer 시간·peak, 아직 미검증.
3. 제출: 공통 source/input/dependency lock, 여섯 job/array mapping, requested resources 및 cap 증명, PENDING/RUNNING과 actual gate를 구분.
4. INITIAL_VALID 또는 PENDING/TECHNICAL_HOLD 인계: actual 단계·저장 근거·현재 미실행 scope와 resume 경로, 이후 모니터링0.

Slurm은 이 envelope에서 **ALLOWED**: 필요한 bounded 공통 기술/teacher(if needed), 위6 science chain만이다. 기술 READY와 source/config/dependency/capsule seal 후 6arm을 모두 upfront 등록한다. 기술 완료 뒤별도 GH 재승인 대기0. 성능이 나쁜 첫 batch로 arm을 선별하지 않는다.
공통 기술 job과 main을 afterok+READY처럼 분리하고, 동시 admitted capacity가 총2 이하임을 확인한다. 다른 project 점유가 없으면 array%2로 두 slot을 사용한다. 49238이나 다른 project 점유가 남으면 합계2인 lane/dependency로 구성하며 기존 job 변경0. Technical gate 실패/미실행인 job을 science 시작으로 표기하지 않는다.
각1GPU/8CPU/mem60416M/exportNONE/Requeue0를 기본으로 실제 cache/5M·P/후보 snapshot memory와 walltime/disk를 preflight한다. Hour cap은 지정되지 않은 null; 수치 event 상한은 실제 GPUh 예산이 아니다. 메모리 상한을 무단 상향하거나 OOM을 no-update로 치환하지 않는다. 공유 모델/환경/타 사용자 자원 변경0.
Six scientific upper bounds는 원 contract대로 target89000/Adam408000/loss497000/solve890/score960/history300이며 기술·observer·teacher·I/O는 별도다. 최대치이지 실측/예약/효율성 주장이 아니다. 실측 후 예상과 차이를 중간보고한다.

## 6. INITIAL_GATE_ONLY — 최신 사용자 종료 규칙

이번 task는 완료까지 자동 관찰하는 예외가 아니다. **초기 실제 gate 통과 뒤 agent 모니터링을 중지한다.**
기술 READY→가능한6arm upfront 제출→최초 실제 scientific batch의 candidate selection/commit-history5/다음 entry 연결까지 한정 확인하여 INITIAL_VALID를 인계한다. 대표 초기 확인은 가능하면 가장 복잡한 C45678을 우선 배치해 새 방법의 상태 경계를 확인한다. N4만 확인하고 전체6arm actual PASS로 쓰지 않는다.
기술 pilot PASS를 science 전체 완료로 쓰지 않는다. 설계의 필수 준비가 끝나지 않았으면 등록/미등록 범위를 공개한다. 기술 또는 science가 PENDING이면 시작을 기다리는 polling 없이 PENDING_HANDOFF하고 actual NOT_RUN/NOT_OBSERVED를 명시한다. 장기 자원대기로 자동 heartbeat를 만들지 않는다.

Initial gate 또는 PENDING/HOLD 인계 시 source+compact submission/initial factual report/receipt를 own-scope nonforce 게시하고 즉시 MONITORING_PAUSED_AWAITING_USER. 이미 제출된 프로그램은 정해진 B1..B10/선택 후 관측/저장을 계속한다. 중지는 agent 작업이지 Slurm hold/cancel이 아니다.
이후 scheduler/result/log polling, sleep loop, heartbeat, callback/자동재개 예약, terminal 대기, 추가제출, 분석·상세보고 확장/main 통합0. USER가 명시 recall할 때만 후속 검산/분석/최종보고를 재개한다. GH도 종료 상태를 추적하는 별도 자동 모니터를 만들지 않는다.
동료 agent를 쓰더라도 복잡한 독립구현만 소유파일 분리하고 pause 시 모두 정지한다. 이전repair/ORBODE/다른pausedtask 자동재개0.

## 7. 관측·저장·후속 상세 보고 준비

설계 최소 산출물 전부를 프로그램에 넣는다: capsule/source manifests, endpoint gate/prefix state/target/fit hash, full precision E/H/B·success IDs, native loss components/target counters/Adam·clamp 및 종료 이유, solver raw-call/cache ledger, phase별 budget/미완료 사유, selected transaction/history receipts, canonical 문항별점수, 비용 원자료.
Original native loss print의 반올림값을 exact NLL로 쓰지 않는다. 계측으로 native 수치를 변경하지 않고 계측 유무 차이를 기술 검사한다. 최종 추가 gradient 관측은 controller에 되먹이지 않고 비용을 분리한다.

매batch entry/selected Current R/P/N, TF strict/token, true/new NLL, W0-success N 참조; W5/W10 fullseen 및 같은 first500의 유지변화, atwrite→later, ALL/ACTIVE/SUPERSEDED/overwrite/요청별tail·loss/gain. 동일 rows는 재사용하고 분모/forward 중복계수0.
B1/5/10의 모든 completed scored 후보 공식 P/N은 선택·다음 상태 결정 봉인 뒤 immutable branch에서만 관측, W/M/RNG 복원. Budget 미완료 상태를 새로 완성하거나 controller가 observer P/N을 읽지 않게 한다. Dev128도 observer다.
선택 gate/support≥3/실제 nonzero층 수, global feasibility/ownN4 사유, pruning의 시도·완료·실패·예산부족, actual endpoint 다양성, layer delta/path/net/energy와 history를 저장한다. 필요 시 GH가 비교할 수 있도록 원문항 identity를 보존한다.

저장정책은 v2 최소 산출물과 실제 disk/cache preflight에 맞춰 잠근다. CPU snapshot이라는 인터페이스를 모든 후보의 full model 디스크 저장 의무로 확대하지 않는다. Full pretrained/Jacobian 복제0, 중복 weight/cache 대량 보존0. 기존 repair의 no-W/M checkpoint 지시를 수정하거나 이미 삭제된 CP를 재생성하지 않는다. v2에서 별도 disk checkpoint를 만들지 않는 경우 exact crash-resume 및 사후 tensor 재구성 한계를 명시하고, runtime entry rollback/5-history exactly-once·receipt와 필수 평가/후보 ledger를 생략하지 않는다. 기존 source/raw/teacher/checkpoint 삭제·덮어쓰기 권한0.

최종 비교는 N4/F48/G48/C4/C48/C45678을 같은 1k분모로 family별 명시한다. C48−G48은 범위/탐색예산도 달라 연속성 단독효과가 아니다. Fixed 대비 calibration-best 우월성, global support 최적성/층 필요성, capacity 고갈을 주장하지 않는다. RS/PS 0pp 과학목표는 실행 gate가 아니며 낮은 PS나 큰 cost도 남긴다.
SH는 factual-only, 해석은 GH 소유. 최종 상세 review/main은 사용자 recall 후만이다. PNG는 직접 코드 생성, GFM 표·분모·링크를 검산한다. NO_BROADCAST_NOT_REQUIRED: 기존S4local자산 재사용/신규rawS4local. 새로운 원격전송 필요 시 exact allowlist·owner·size를 먼저 보고하고 무단 broadcast0.

## 8. 허용 write·수리·보고 경로

- project/run_scripts/sequential_local_z_allocation/ — 새versioned runtime/adapter/CPU tests/observer/analysis/plot; 과거 shared runner 수정0.
- /data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/ — authoritative/capsule/technical/arms/receipts/inputs/raw/cache/필요한 최소tensor.
- audits/servers/server4/2026-09-17-sequential-local-z-allocation-v2/
- experiment-reports/servers/server4/sequential-local-z-allocation-seq1000-2026-09-17-v2/
- messages/acks/server4/2026-09-17-sequential-local-z-allocation-v2.md
- messages/server-heads/server4/2026-09-17-sequential-local-z-allocation-v2.md
- runs/odeedit_sequential_local_z_v2_s4_20260917/
- tasks/status/odeedit_sequential_local_z_v2_s4_20260917/server4.json

Envelope 밖 source/공유PROTOCOL/global설계/타서버파일 수정0. Implementation·preflight·numeric negative cases·data-leakage·budget/source/import·state mutation·resource·report evidence를 red 관점으로 점검한다. 단순 검사는 직접하고 실제 독립 red 수행 여부를 사실대로 기록한다. Warn은 근거/한계로 구분하고 과학조건을 위반하는 block은 멈춰 보고한다. GH에 중복 전체raw/GPU 검증을 요청하지 않는다.
본scope 구현/source/tests 및 compact 제출·초기인계만 latest main과 conflict-free nonforce 통합 허용. 충돌은 무단 overwrite 없이 정확경로 보고. Raw/tensor/teacher/prompt/fullstdout Git0. 접근 helper의 기존 stale session 또는 좁은 path rule은 실제 registry/명시 envelope와 구분하여 제한을 기록하고 허위PASS/sharedhelper수정0.
최초 M0와 각 중간보고를 direct peer로 보낸 뒤 위 초기 종료 경계를 지킨다.

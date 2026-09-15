# GH → SH4: EP-TW-1 alpha cap sweep / W0→SEQ1000 완주

Instruction ID: ODEEDIT-S06-EP-TW1-ALPHA-CAP-SWEEP-SH4-V1
Nonce: ODEEDIT-GH-SH4-EP-TW1-ALPHA-CAP-SWEEP-20260915-R1
발신 GH session 01a04939-8873-7673-8dca-4c7fc5e31af0.
수신 server4 / session 01a04939-b5c7-7a03-ba2d-ef3343d62cfd / CWD /data/janghj/ODE-edit / repository hyunjun1127/ODE-edit.
사용자 최신 요청: “project/proposals/2026-09-15-ep-tw1-alpha-cap-sweep-gh-instruction.md 이거 server4에게 task 진행시켜라. alpha cap 부분의 sweep이다.”

## 1. 최신 권한과 읽기 범위

사용자의 위 요청으로 원문 GH-EP-TW1-ALPHA-CAP-SWEEP-20260915-V1을 활성화한다. 원문 6개 정본을 전체 읽고 source에 결속하라. GH는 원문·설계·contract·cells·독립 리뷰·기하 summary를 정독했다. 새 dispatch는 plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-dispatch.json, arm별 GH science lock은 같은 이름의 디렉터리 arms/{CAP1,CAP10,CAP100,NORM_ONLY}.json이다.
원 설계의 execution_authorized_by_this_file=false 및 CPU geometry의 model_forwards=0은 작성 당시 기록이므로 바꾸지 않는다. 본 사용자 승인과 새 dispatch가 실행 권한이다.

정본(같은 repo-relative 내용이며 GH 절대경로 링크는 relative path로 찾아라):
- project/proposals/2026-09-15-ep-tw1-alpha-cap-sweep-gh-instruction.md
- plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-design.md
- plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-contract.json
- plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-cells.csv
- audits/global/2026-09-15-ep-tw1-completed-artifact-review-ko.md
- audits/global/2026-09-15-ep-tw1-alpha-cap-geometry/summary.json
보조 by-batch.csv/candidate-geometry.csv/checks.json도 같은 geometry 폴더에 원 bytes로 발행했다. 원문·정본의 SHA/bytes는 dispatch-inputs.json에 있다. 링크된 과거 local raw를 새로 전부 원격 수집하거나 GH 기하검산을 중복하는 task가 아니다. 기존 SH source/input/검증 기록을 활용한다.

이 task만 **초기 gate/PENDING 뒤 중지 규칙을 대체**한다. 구현·제출·승인 범위의 모든 arm1000 완주/기술수리·CPU 상세 사실분석·보고서/코드 own-scope main 통합까지 계속한다. 단일 B1 pilot으로 줄이거나 매 단계 허가를 반복 요청하지 않는다. CAKE48101 및 다른 paused task에는 이 override를 적용하지 않는다.

## 2. 과학적 범위와 cap 구현

기본 신규 CAP10 bounded10 / CAP100 bounded100 / NORM_ONLY disabled-null 각각 fresh pretrained W0/coldM0 → B001..B010, B100×10. CAP1 job47962는 reuse가 기본이다. 공통 과학 동작 변경으로 비교가 부적절할 때만 이유를 선기록하고 CAP1 같은1000의 조건부 재실행을 허용한다. 최대 신규4chains/40batches, 기본3chains/30batches/3000 arm-request observations, unique1000이다. 앞 arm 성능으로 뒤 arm을 제거/튜닝하지 않는다.

정확한 baseline source6d317bdb2660d7e9919bc3a9fb878564e9729e37와 gate-skip-r1/execution.lock SHA5a19c2be919362d08b2ea80e69a406d7b6de569aade8de639036f69db5d5d8f9를 재사용 기준으로 한다. 현재 main의 관련 source와 실행 frozen source의 차이를 구분하라.
NumericalPolicy에 alpha_cap_mode="bounded"/"disabled"와 null 검사를 추가하되 실제 수치의 finite 검사는 유지한다. inf/NaN/0/임의 큰 값으로 disabled를 대용하지 않는다. alpha_norm=.25*||actual Vp-Wentry||F/(||dA||F+1e-12), bounded는 기존 순서 그대로 min(cap,alpha_norm), disabled는 alpha_norm이다. 저장 scalar 계산/receipt 추가가 RNG/state/gradient/rounding에 영향을 주지 않게 한다.
원 exact-zero/epsilon, halfspace, target-ball→C-only25%write-trust, actual Vp anchor, E/D 각1 sweep, RAW/C1/C05/C025 전체 finite menu/min-D/byte-dedup/RAW-priority tie/양의 E 허용0/strict exact-ID subset/RAW native commit/history1/ledger를 고정한다. Native fitter·writer는 수정하지 않는다.
후보 조기 종료, target refresh, quality restoration, old replay, N4 calibration/KL budget, 새threshold/seed/order/reference, 추가cap/beta/layer/10k는 금지한다.

GH arm science lock은 각 하나의 chain용 dispatch fields를 별도로 발행했다. SH4는 구현 후 exact execution HEAD/tree/archive/import/source/assets/resource/output를 결속한 **새 arm별 execution.lock.json**을 local create-once로 완성할 책임과 권한을 가진다. 이 입력 잠금은 아직 모델 실행/제출/검증 PASS가 아니며, aggregate3-chain contract를 기존 single-chain validator에 넣거나 validator 전체를 끄지 않는다. Source가 아직 없으므로 GH 문서의 execution_binding=PENDING_SH_SOURCE_FREEZE를 final execution lock으로 오기하지 않는다.

## 3. 공통 fixture와 저장·평가

기존 W0 revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, seed20260915, Llama L4 config/P physical4→asset0→local0, FP32/eager/원 kernel·TF32·MB16·tokenizer/position/contexts를 exact lock에서 결속한다.
fixed10k 자산 /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1의 first1000만 사용한다. whole orderedroot5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729, prefix40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd. 기존 sample.lock SHA871f6c04df4697f236353938e09290462eed67871f07f02d8f0b112e9f9ee7d3를 결속한다. 모든arm B1[0:100]..B10[900:1000], 재추출/shuffle/filter0.
Teacher47592 완료192/24shards, manifestSHAf81b798f44ce626ac1e2e402ca7438363b1dd60f5681ec0ac17b9e92d096761a 및 C4 reference identityf5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0/S64/Dev128을 재사용한다. 누락/손상된 정확 범위만 복구하고 오래된PENDING 때문에 teacher 재생성하지 않는다.

모든arm 자기 W/M/RNG/ledger로 nativeZ/K/A/gE/gD를 새로 계산한다. 불변 W0/P/reference/teacher/input만 공유하며 과거cap1의 미래방향 scale replay·warmW50·타arm endpoint에서 출발0.
CAP1 checkpoint10개 현재부재와 selected L4만 남은route/native에서 복원확인된 경계를 명시하고, cap1 결과 재사용에 완전 resume CP를 부당하게 요구하지 않는다. GH는 이번에 추가 raw 감사를 하지 않는다.
**CAKE의 “weight 저장하지 말라” 지시는 CAKE 한정이며 이 새 sweep에 상속하지 않는다.** 이번에는 원문대로 각arm W10 L4 weight/model/token/context identity를 실제 보존한다. 기존 EP 평가/계측·checkpoint schedule을 유지하며 native/route/evaluation/M/RNG/ledger 생성·현재 보존 inventory를 구분한다. 동일arm 완전자체state에서만 technical resume; weight-only를 fullresume이라고 하지 않는다. 저장량은 제출 전 실제 확인, 기존파일삭제/보존점임의축소0.

원문 표의 필수 action/candidate/Current+accepted-old/whole1000/first500 W5→W10/atwrite→W10/active-superseded/S64·Dev/cost·I/O 기록을 모두 남긴다. WholeW10 분모RS1000/PS2000/NS10000 및canonical true/new NLL 정의/tie failure 유지, current재사용분모중복0.
최대C1, RAW=0 포함 selected, nonzero selected 크기/선택빈도는 별도 주표. 성능/엄격ID 실패/많은RAW는 기술실패gate가 아니다.
기존 N4/AlphaEdit/MEMIT/BLUE/L4-only first1000 reference 재사용; 새baseline editing0, REFIT4 W50suffix와혼합0. 원본 과학값/한계 유지, source/CPU arithmetic consistency를 미분 검증 PASS로 대체하지 않는다.

## 4. 실행 envelope / 자원 / 기술 수리

새 branch codex/server4-ep-tw1-alpha-cap-sweep-v1 및 local root /data/janghj/ODE-edit/local/ep-tw1-alpha-cap-sweep/20260915-v1/를 사용한다. CAP10/attempt-v1, CAP100/attempt-v1, NORM_ONLY/attempt-v1 및 조건부CAP1/attempt-v1로 완전 분리; 수리는 attempt-rN으로 oldbytes/cost 보존.
허용 write:
- project/run_scripts/bg_tw_reference/ep_tw/ 의 cap config·validation·receipt·launcher·필수계측/tests/분석; 필요한 parent bg_tw_reference/ 실행결속만 추가 허용.
- 위 local root의 input/source/archive/lock/raw/checkpoint/log/control.
- audits/servers/server4/2026-09-15-ep-tw1-alpha-cap-sweep/
- experiment-reports/servers/server4/ep-tw1-alpha-cap-sweep-2026-09-15-v1/
- messages/acks/server4/2026-09-15-ep-tw1-alpha-cap-sweep.md
- messages/server-heads/server4/2026-09-15-ep-tw1-alpha-cap-sweep.md
- runs/odeedit_ep_tw1_alpha_cap_sweep_s4_v1/ 및 tasks/status/odeedit_ep_tw1_alpha_cap_sweep_s4_v1/server4.json.
원native/BLUE/fitter/reference recipe/sharedruntime/다른task/global설계 파일 변경0. 필요하면 task-local exactpatch와diff만, 원본자료보존.

Slurm ALLOWED: 각chain1GPU/8CPU/mem60416M/exportNONE/Requeue0, job명 odeedit_ep_tw1_cap10_s4 / cap100 / normonly; 조건부cap1동등. wall12h 기본, 실제구현비용에 맞춘 resource lock; GPU-hour hardcap=null. 예상기본6.4117GPUh/조건부8.5489GPUh는 47962 선형추정이지budget/실측/성능gate 아님. Teacher98s재사용비용과 이전실패473+69s는 별도, native내부component중복합산0.
**Server4 모든 project active+admitted pending 포함 GPUcap2**. CAKE48101은 마지막PENDING 인계였으며 지금 상태를 추정하지 말고 제출 직전 한정 scheduler/resource 확인으로 점유 capacity를 세라. CAKE예약1이 남아 있으면 sweep은 안전한1lane로 upfront pending queue(CAP10→CAP100→NORM_ONLY)를 구성하여 합산≤2. CAKE예약이 없으면2slot까지 사용 가능하나 과학순서/설정은 미리 고정. Cap1조건부도같은cap. 기존CAKE/타job 취소·requeue·throttle·source/config/output/monitoring변경0.
가능한 기본3arm을 upfront등록하고 안전한 dependency/array throttle로 자연진행시켜라. 이전arm품질로후속submit을선별하지 않는다. 한armtechnical문제는독립arm실행을불필요차단하지 않는다. 실제OOM/NaN/잘못된state/source/저장오류는범위내최소수리 후 affectedarm만재개/재실행, 실패비용/원자료보존.

좁은 CPU config-mode/null/finite/legacycap1산술/disabled/trust/receipt fixture 및 source/sample/resource 검사를 직접 수행한다. **과거 waiver의 FD/ULP/jitter/fullgradient/VJP/selfKL 등 GPU진단gate 복원0**, numerical_validation=NOT_ESTABLISHED / SKIPPED_USER_DIRECTED 유지. 최소red는 scientificscope/source/state/분모/자원/생략표시를 확인하며 성능 gate를 새로 만들지 않는다. 복잡한독립구현/감사에만 bounded subagent를 사용하고 단순파일검사·통신은 직접 처리한다. 원문으로해결가능한선택은판단기록하고재승인대기하지않는다.

## 5. 완료 보고와 GH 해석 분리

SH4는 이 task의 구현→기술수리→모든chain완료확인→CPU사실보고/main통합까지 계속한다. 초기gate/PENDING만으로최종STOP하지않으며 적절한간격의한정관측을사용한다. 별도daemon/heartbeat/다른task자동재개0. 종료/핵심typedfailure/원분모firsttable/최종보고만간결통신하고GH중복remote검산을선행조건으로요청하지않는다.
Factual report는 diagnostic-report-ko.md, arm별receipt/reuse/artifact/input/analysismanifest/rootedreceipt, 후보·batch·cohort·paired·baseline/costCSVJSON과codePNG를담는다. SH는실제사실/산술/미실행/비용만기록; 기하예측과새model실측분리. 해석·우열·다음방법/추가run판정은GH별도 global review이다.
SH ownscope 코드/사실보고 완료후 최신main보존 cleanintegration/nonforce push ALLOWED. raw/tensor/prompt/teacher/fullstdoutGit0. NO_BROADCAST_NOT_REQUIRED; 기존S4로컬재사용, 신규cross-serverdata transfer/삭제는없다. 다른GH/SH변경/표렌더수정보존. 원문의scope밖후속chain제출0.
첫ACK: FULL_READ/정본SHA, CAP1reuse판단, 기본3/조건부4, cap-disabled변경, source/output/실제자원계획. 최종에성공/실패/재개범위와정확job/source/hash/경로를반환한뒤STOP하라.

# GH → SH4: Local-z adaptive allocation — W0 B100×10, 7개 신규 정책

instruction_id: ODEEDIT-S06-LOCAL-Z-ADAPTIVE-ALLOCATION-SEQ1000-SH4-V1
nonce: ODEEDIT-GH-SH4-LOCAL-Z-ADAPTIVE-ALLOCATION-20260916-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
target_server: server4
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit

## 1. 사용자 요청·정본·권한

사용자 원문:
“/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-local-z-adaptive-allocation-design-v1.md
이 task SH4에게 진행시켜라.”

전달 준비 중 도착한 최신 사용자 override: **“server4 gpu cap은 1이다.”**
이는 본 task뿐 아니라 Server4 신규 admission의 현재 한도다. 과거 cap2/4보다 우선하며 다른 서버 cap은 바꾸지 않는다. 현재 tracked control/gpu-concurrency-policy.tsv의 server4도1로 갱신한다. 기술 준비·teacher·본실험을 합쳐 최대1GPU이며 7arm은 순차 queue로 올린다. 이미 제출한 다른 job을 임의 cancel/restart하지 말고 admission에 포함한다. 초과 기존 점유가 있으면 새 실행을 막고 상태를 보고한다.

위 설계를 실제 Llama 구현·필수 기술 준비·지정 7-arm 실행에 연결하라. GH는 설계169행/21218B, contract405행, cells70행, CPU 참조154행/기존35checks 및 재사용 fitter/sequential runner를 읽었다. 원본5파일을 main에 byte-exact로 게시하며 SHA/size는 companion source-input-manifest.json에 있다.
- design SHA0aa17998b55850c7b361545c6388930bf75f7ed9d11dc5e4968acf868c370b69.
- contract SHA07579b9976090266310b3f771312a7dd486c2afacc233cfd1332b0c9a69279df.
- cells SHA6c4fa65a61307400189f214e0af6da8b0697ce13ec7b098ddc9e07989a35e65d.
- audits/global/2026-09-16-local-z-adaptive-allocation/validate_design.py 와 checks.json은 설계·selector CPU 참조다. 35CPU 검산은 실제 Llama PASS가 아니다.

과학 정본은 최신 local-z 설계/contract/cells다. 이전 donor 설계는 source/evidence 참조만 하고 warm W50/M50·재구성M8·고정Historical128/Wiki/MMLU·옛 conditional arm은 상속하지 않는다. C4 계약은 데이터/token/teacher 의미만 재사용하며 옛 EP route/alpha/quality screen/one-chain 범위를 상속하지 않는다.
직전 CAKE/baseline·cap 보고서 분리는 main030444c52b64621672ae740d1a80bb9f69874fd7에서 완료/STOP한 direct 답변을 확인했다. 이번 별도 task/worktree에서 시작하며 그 v2와 SH2 SL-ZFlow를 포함한 최신 main을 보존한다. 단계별 재승인·GH 중복 raw/GPU 감사를 선행조건으로 요청하지 않는다.

## 2. 범위: 일곱 정책 모두 같은 cold W0에서 새로 실행

Llama-3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, 모든 parameter FP32/eager, matmul/cudnn TF32 모두off, seed20260916. 각 arm은 동일 pre-edit W0·M4/M8 exactzero에서 B1–B10/B100×10. 원 지식 P4/P8은 기존 자산을 유지하며 zero화하지 않는다.

| arm | 정책 | 최종 history 대상 |
| --- | --- | --- |
| N4 | local4 (1,0) | L4 |
| REFIT4 | L4 .75 뒤 같은 L4 fresh fit, 두 subwrite | L4 |
| L75 | LOCAL (.75,1), 고정 raw endpoint | L4/L8 |
| T75 | TERMINAL (.75,1), 고정 raw endpoint | L4/L8 |
| L4D | local4 a4=.75 또는1, a8=0 | L4 |
| LD | LOCAL a4=.75 또는1 × a8=0,.5,1 | L4/L8 |
| TD | TERMINAL 같은6gate + own-entry common N4, 총7후보 | L4/L8 |

신규7chains/70batches/unique1000/arm-request7000. 기존 N4/REFIT4/RES8/FULL8/BLUE/AlphaEdit/CAKE는 historical 참고일 뿐 이번 paired N4 등을 대체하지 않는다. 다른 seed/cold 결과나 warm prepared/history를 가져오지 않는다. TD는 pure-terminal이 아니라 terminal-proposal+N4 controller다.
FULL8/BLUE 추가chain, terminal(.5,1) 과학arm, request별 gate, L4+L5, full10k, 최적fixed 탐색, 추가order/ratio/barrier/ODE/Adam은 미승인이다.

S4 /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1 을 official load_prefix로 모델 전에 검증한다. datasetSHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1, wholeorder5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729, prefix1000 orderedroot40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd. ordinal[0,1000), shuffle/replacement/filter/성능기반 제외0.
공통 cold capsule은 model/tokenizer/context text+actual token IDs/P/hparams/RNG를 한 번 봉인해 전 arm에 사용한다. Context가 필요하면 원 native 절차로 W0에서 한 번만 생성한다. 모델 seed20260916과 기존 C4 sampling seed20260915를 혼동하지 않는다. 보존 텍스트를 재선정하지 않는다.

## 3. 실제 target/writer 연결

원 BLUE311b076a AlphaEdit_main/compute_z/compute_ks와 project/run_scripts/low_cost_write_donor_pilot/{fitting.py,sequential_runtime.py}를 읽고 read-only 재사용하라. 기존 fitting은 singleton local residual, sequential_runtime은 W50/B51–60로 고정돼 있으므로 그대로 launch하지 않는다. 별도 cold runner/TERMINAL adapter/.5 materializer를 구현한다. 원 native/공유 module 변경0.
전 arm ridge/L2=1, 기존 BLUE/L4 target hparams(decay.5/clamp.75/lr.1/max25loss·24Adam 등)를 exact config로 봉인한다. 원 direct solve 순서·context/key 평균 유지, inverse 선계산/대칭화/새 solver 치환0.
- LOCAL: own entry z4/h4/K4→full N4, actualFP32 a4 partial W, 각 a4 상태에서 fresh z8/h8/K8. 같은 a4의 a8 후보끼리만 fit 공유.
- TERMINAL: own entry Z8 한 번 고정. L4 R=Z8−h8(entry), L8 R=같은Z8−h8(partial). Key는 해당 writer layer/currentstate. Z8−h4 또는 singleton fitter에 z8만 주입하는 경로 금지.
- gate0/1은 U/V exactcopy, 나머지 FP32(U+g*(V−U)). 새 materializer에 .5를 명시하되 기존 materialize_alpha는 변경하지 않는다. RHS scaling과 bitexact를 가정하지 않는다.
- 후보 평가는 모든 token에 actual weight가 적용된 model forward다. subject activation hook 효과로 대체하지 않는다. Gate 합1 제약0, normshare를 기능기여율이라 하지 않는다.
- P physical4→asset0, physical8→asset4, 각각 singleton local0의 tensor hash를 결속한다.
- REFIT4는 같은 L4 두 번째 fit, innerhistory0/final M4 append1이다. I2/I4 write-refresh 등 다른 정책을 넣지 않는다.

## 4. selector·Past·state

E=native rewrite training target-new NLL의 token→context→request 평균. S_cur=canonical rewrite target-new TF-strict ID집합. H/S_past=과거 도착한 Past64 canonical rewrite만. D=C4 S64 fixedW0 full-vocab KL(p0||candidate), vocabulary합→128position평균→64document평균.
Feasible: E(c)<=max(E(N4),.05)+1e-4, S_cur(N4) subset; Past가 있으면 H(c)<=H(N4)+1e-4 및 strict subset. 기준N4는 각 dynamic arm의 own entry에서 실제 계산하는 action이며 독립 N4 chain에서 빌리지 않는다. 고정 N4/REFIT4/L75/T75에 dynamic quality screen을 추가하지 않는다.
Feasible Dmin+1e-6 이내 tie는 commonN4→actual L8delta0→concat(ΔW4,ΔW8)의 작은 Frobeniusnorm→candidateID 사전순. 동일 materialized weight만 dedup하고 선언 후보/equivalence/score reuse를 모두 기록한다. no-plateau/strict-only shadow는 이미 얻은 점수로만 진단하며 commit/후속 state에 영향0.

Past64는 received-event ledger에서 raw(subject,relation)의 최신 target, fact별 최신event 하나, 현 batch overwrite fact 제외, 규정 SHA256 UTF-8 priority로 최대64개다. stable_event_id 인코딩을 사전봉인, 같은 ledger면 같은 ID, B1empty. 성능·미래목록·official P/N·옛 Historical128로 선택0. EP accepted-only ledger를 그대로 복사하지 말고 본 설계의 received-event 의미를 구현한다.

후보마다 같은 W/M/P/context/RNG에서 격리하며 생성 순서가 target/endpoint/선택을 바꾸지 않아야 한다. Cache key에 state/layer/context/lookup/source/hparams/RNG를 포함하고 case/layer명만으로 branch를 공유하지 않는다. Candidate RNG 복원·선택 endpoint 이후 RNG 처리도 봉인해 평가/cachehit가 다음 batch를 바꾸지 않도록 한다.
Inner history0, 최종 eligible layer마다 whole B100을1회 append. LD/TD가 a8=0 또는commonN4를 선택해도 M8 append1. History gate 가중·request제거·과거M8전수refresh0. 실패batch는 W/M/context/RNG/hooks를 entry로 복원하고 미commit 처리한다.

CPU참조의 choose에는 non-reference nonfinite 후보를 skip하는 toy 경로가 있다. Production은 설계/contract의 technical_failure를 우선해 NaN/손상state/OOM을 단순 infeasible로 버리고 N4 fallback하지 않는다. 새 namespace에서 필요한 negative test를 추가하고 원 CPU evidence는 보존한다. 정상 quality 부적격→N4 선택은 사전 정의된 정책이다.

## 5. reference와 필요한 기술 준비

Reference768 identity f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0. Teacher manifest=/data/janghj/ODE-edit/local/bg1-c4-ours-first/20260915-v1/attempt-v1/teacher-output-v1/teacher-manifest.json, SHA f81b798f44ce626ac1e2e402ca7438363b1dd60f5681ec0ac17b9e92d096761a. Teacher47592/192documents는 이미 완료된 자산이다. 옛 C4 계약의 PENDING을 현 상태로 쓰지 않는다.
기존 token256+BOS1, logit[128,256)128positions/fullvocab/문서ID 유지. 이번 공통 W0/TF32off 설정과 기존 teacher kernel/tokenizer/score position을 실제 forward로 연결한다. 규정 재현성을 만족하지 않으면 같은 IDs/tokens/W0에서 필요한 S64+Dev128 teacher를 공통 설정으로 한 번 재생성하고 ID·준비비용을 공유한다. 결과를 보고 teacher를 선택하거나 Report256 준비를 선행조건으로 추가하지 않는다. Dev128은 B5/B10 observer만.

기술 준비: local(1,0)↔N4, local(1,1)↔동일entryBLUE, terminal(.5,1)↔원 nonblue[4,8]/L2=1의 source·actualupdate 연결, 전체 token, gate0/1/.5/.75, history once, state복원, 후보순서/cache freshness, online/observer 분리.
FP32 endpoint scaling과 native RHS divisor의 차이는 기록하되 설계가 요구하지 않는 역사적 bitexact를 새 hardgate로 만들지 않는다. 동일 endpoint E/H 오차<=5e-5, D<=5e-7, strict 동일은 설계 고정 기준이다. 실패 시 evaluator/구현을 수리하고 성능을 보고 epsilon을 키우지 않는다.
옛 EP gate-skip을 이번 필수 검증에 자동 상속하지 않는다. 동시에 이번에 필요 없는 gradient/FD/ULP/KKT heavy gate·motivation 재증명을 추가하지 않는다. 원35CPU는 real-model PASS가 아니다.
기술 실패는 첫 exception/episode/cost 보존 후 최소 구현수리. 기존 승인 범위 repair는 단계별 재승인 불필요하나 방법·허용치·arm을 바꿔야 하는 모순이나 실행 불가 자원은 명확히 HOLD한다. 유효한 낮은 PS/높은 손상·비용/N4선택률은 실험 중단 gate가 아니다.

## 6. cap1 등록·초기 gate 이후 pause

최신 Server4 cap=1. Active+admitted pending의 실제 동시 실행 가능 용량을 합산한다. 기술/prep/teacher와 본실험을 합해 한 slot이며 새 standalone과 array를 각각1로 따로 계산해 동시에2가 되면 안 된다. 각1GPU/8CPU/mem60416M/exportNONE/Requeue0, GPU-hour hardcap=null. 다른 서버 cap은 불변이다.
필요 resource-only admission 조회 외 다른 task 결과/모니터링 재개0. 기존submitted job은 임의 cancel/restart하지 않으며 그 slot이 비워지기 전 새 실행을 막는다. 새 7arm은 %1 array 또는 동등 단일 afterany lane으로 upfront queue 등록한다. 필요한 common technical/prep는 afterok/READY 등으로 science와 직렬 연결한다. 이전 cap2 정책·동시2배치를 상속하지 않는다.
설계 총12000target request-calls/최대288000Adam·300000loss/150solve/190후보는 계획 산술이지 실측 시간이다. Source/config/sample/seed/teacher/cache/arm mapping/자원·storage lock을 등록 전에 봉인한다. 최초 실제 load/fit/candidate forward/history/eval/I-O/peak로 wall·disk reserve를 산정하고 기존 데이터 삭제0. 기술 준비와 과학비용을 분리한다.

기존 사용자 INITIAL_GATE_ONLY/PENDING_HANDOFF 방침을 유지한다. 이번 요청에 end-to-end 자동모니터링 예외를 새로 부여하지 않는다.
1. 공통 기술·source 준비가 가능하면 완료하고 7개 프로그램을 모두 B1–B10의 평가·저장까지 자동 진행하도록 등록한다.
2. 시작 가능하면 처음 실제 dynamic(LD/TD 등)의 candidate score→selector→actualcommit/history→nextentry 연결을 최소 초기gate로 확인하고 MONITORING_PAUSED_AWAITING_USER.
3. PENDING/외부자원 대기면 실제미실행으로 기록하고 등록된job/미등록scope를 명확히 인계한다. 시작을 기다리는 polling/sleep0. 기술 미완료를 모든7science 제출완료/PASS로 표기0.
4. Pause 뒤 scheduler/result/log polling, heartbeat/callback/terminalwait, 자동 추가submit·분석·report확장·main통합0. 이미 등록된 프로그램은 변경하지 않고 자연 진행한다. 전체완료 review/report는 사용자 recall 때 재개한다.
5. 초기 source+compact submit/gate/pause 보고의 own-scope main publication은 pause 확정까지 허용한다. 기술PASS만을 scientific성능완료로 삼지 않는다.
성능으로 후속 arm을 삭제·gate를 조정하지 말라. SH2 SL-ZFlow/CAKE·cap/ORBODE/다른pausedtask는 재개하지 않는다.

## 7. 저장·평가·후속 review 준비

매batch entry/selected CurrentR/P/N, canonical true/new NLL/tie=failure/TFstrict/two-P, allrequested 분모를 유지한다. Official P/N은 observer만 접근한다. B5actualfirst500, B10samefirst500/후반500/all1000와 Dev128, W0→entry/entry→atwrite/atwrite→final, ALL/ACTIVE/SUPERSEDED, margin/악화tail/p95p99/lostgained를 exact ID로 연결한다. 필요한 W0관측은 공통 W0에서 한 번 얻고 재사용한다.
모든 후보 E/H/D/strict집합/feasibility/선택이유·gate, TD의commonN4 선택과terminal제안 품질, target/loss/Adam/clamp/solve/copy/history/eval/teacher streaming/I-O/peak를 저장한다. Dedup·연구용 공유계산과 online필수 비용을 구분하고 nested timer 중복합산0. Purewriter가 미분리면 그대로 표시한다.
Selected W4 및 필요한W8/M4/M8/context/RNG/nextbatch/source/model/P/order/receivedledger를 포함한 snapshot을 최소B1/B5/B10에서 보존한다. 각batch commit/link/실제delta·후보재구성 근거를 남긴다. 저장량과 전체candidateweight 저장 여부는 source freeze 전에 구체화하고 큰fullmodel/denseJacobian을 추가하지 않는다. CAKE만의 no-checkpoint 지시를 상속하지 않는다. CPU reload와 GPU continuation 수준은 분리한다.

최종 사용자 recall용 독립 package는 diagnostic-report-ko.md, final7arm표, candidate/selection/Current/retention/paired/NLLtail/compute/source-config-data-state-history inventory, manifest/rootedreceipt/재현명령을 갖춘다. 주 비교 LD−L4D/LD−L75/LD−REFIT4/LD−TD/T75−L75. SH는 사실·수치·미측정을 정리하고 과학해석·claim 판정은 GH global review로 분리한다. Own-entryN4와 독립N4, 개발pilot과 blindtest, 강도와capacity를 혼동하지 않는다. PNG는 코드 생성만. 과거baseline/CAKE/warm은 별도historical 참고이며 새7arm 주표에 섞지 않는다.

## 8. write envelope와 종료

새 clean branch codex/server4-local-z-adaptive-allocation-seq1000-v1/전용worktree, 최신main 기준. 구현이 이미 있다고 가정하지 말고 실제 import/source를 SHA로 봉인한다.
허용 write:
- project/run_scripts/local_z_adaptive_allocation/ — 새 runner/terminal adapter/selector/observer/transaction/config/test/analysis/plot.
- /data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/ — authoritative/common/technical/arm-attempts/receipt/lock/raw/checkpoint.
- audits/servers/server4/2026-09-16-local-z-adaptive-allocation/
- experiment-reports/servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/
- messages/acks/server4/2026-09-16-local-z-adaptive-allocation.md
- messages/server-heads/server4/2026-09-16-local-z-adaptive-allocation.md
- runs/odeedit_local_z_adaptive_allocation_s4_20260916_v1/
- tasks/status/odeedit_local_z_adaptive_allocation_s4_20260916_v1/server4.json
- local/state/server4-gpu-cap1-20260916-v1/ — 최신 cap1 수신/적용기록.
기존 donor/native/EP module/다른report/설계원문·sharedsessionenv는 수정하지 않는다. 명시 허용경로를 generichelper가 거부하면 근거와 한계를 기록하고 거짓PASS/공유helper변경0.

Slurm allowed는 이7chain과 필요한 한정technical/cold/teacher 준비만이다. Pre/post검산은 source/state/data/family/history/비교유효성·자원·단위 및 비용을 포함하며 임의 성능AND gate를 추가하지 않는다. 자체검산/독립reducer/별도red 수행 여부를 사실대로 기록한다. 복잡한 독립 구현을 분담한다면 파일소유권·타인변경보존·pause시worker정지를 지킨다.
NO_BROADCAST_NOT_REQUIRED: S4 기존local 자산과 신규local결과이며 대형복제 불필요. 새 외부/타서버 전송 필요시 exact대상/size/owner를 보고하고 무단broad rsync0. Raw/tensor/prompt/fullstdout Git0.
Own branch와 main은 검증된source/compactreceipt/report만 최신타SH변경을 보존해 nonforce 게시한다. Pause확정 이후 자동확장0, force/reset/임의ours-theirs0. 다른scope 충돌은 보존 후 보고한다.
첫 FULL_READ/M0에 source/미구현/cold·teacher 준비/7armmapping/cap1 단일lane/산술범위/실측전 자원예상과 monitor경계를 보고하고 진행한다. 초기 actualgate 또는PENDING/HOLD 뒤 job/source/lock/경로/resume receipt를 GH에 인계하고 사용자호출을 기다린다.

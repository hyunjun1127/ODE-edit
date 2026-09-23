# GH → SH4: 기존 BASE delayed-write E0(endpoint) → E1 → E3 실행

Instruction ID / ACK nonce: GH-SH4-NATIVE-DELAYED-WRITE-E3-20260924-V1
발신 GH: 01a04939-8873-7673-8dca-4c7fc5e31af0.
수신 SH4: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd, server4,
/data/janghj/ODE-edit, repository hyunjun1127/ODE-edit.
사용자 원문: “이 task GH에게 SH4에게 TASK 진행하도록 명령하는 지시문을
전달해라. E3 실험까지 우선 각 GATE에 DEPENDENCY를 걸어서 이어서 진행하도록 시키자.”

## 1. 실제 실행 승인과 현재 경계

GH는 정본 지시문/DAG/리뷰/최소안/원안 대비 범위를 검토했고,
본 envelope로 필요한 구현·자산 준비·최소 기술검증 및 아래 전체 dependency
chain의 조건부 제출을 승인한다. 단순 계획 작성이 아니다. G00에서 구현/source/
resource 사전 검토 증거를 봉인하고 기술 gate에 따라 진행한다. 단계별 새 사용자
승인이나 별도 GH 재승인을 기다리지 않는다. 같은 task의 구현·등록 중간보고는
보내되 보고 ACK를 다음 stage 선행조건으로 만들지 않는다.
기술 오류는 정의/고정된 수치기준을 유지한 최소수리·새 attempt·영향 downstream
재결속을 허용한다. 과학 결과가 나쁘다는 이유로 반복/수리하지 않는다.

2026-09-24 GH actual thread/resume: 정확 SH4/CWD 일치, active turn 없음,
직전 전체 job 리뷰 turn 01a0cef5-e2a1-7bb1-812d-e4a4d2a9db8d completed.
main892b804f에 해당 리뷰 종료 게시됨. 이는 현재 GPU 가용성 관측은 아니다.
제출 직전 SH4가 own project allocation/admitted pending·disk를 실제 확인하라.
다른 task의 job/source/dirty/보고는 보존한다. 그 task의 pause 해제 권한은 없다.

## 2. 정본 실제 수신

GH 원자료 root:
/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-24-delayed-write-lifelong-review/
- gh-sh4-e3-instruction-ko.md: f05d63ec61c4eb3c9c2ff0f9a4c05729e112d2910f548afe1ca4869db65093f0
- e3-dependency-plan.json: 736a91f40765759d049c4645e95e1e3df0d1e6c08df48e16d2cca680d95a7c58
- dispatch-input-manifest.json: 145253fa813aaeef031070ef14e69a1ab253255ab386df8ed9f34531ad2dcef2
- review-ko.md, revised-minimal-protocol-ko.md, paper-by-paper-ko.md
원 첨부9개는 원 instruction의 exact attachment root에 있다.

transfers/approvals/2026-09-24-native-delayed-write-e3-sh4.json의
15파일/245092B exact source→destination allowlist를 승인한다.
SH4 sole pull, SOURCE_KEEP/create-once, receiver size/fullSHA15/15 기록.
destination=/data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1/inputs/design/
review/6파일+original/9파일을 먼저 확보/정독한다. 미커밋 source이므로 git fetch만으로
전달되었다고 쓰지 않는다. 논문 PDF18개나 불필요한 큰 CP 전체 복제는 이번 승인에 없다.
현재 server4의 exact BASE checkpoint/원 raw·자산을 우선 재사용한다.
실제 필요 자산이 없으면 exact missing 목록/typed BLOCKED_ASSET과 필요한 준비를
보고한다. 없는 자산/패널을 다른 family나 문항으로 몰래 대체하지 않는다.
승인된 repository-local ordinary artifact의 정확한 누락 입력 수신은 PROTOCOL의
비파괴 allowlist 절차에 따르며 대규모 외부경로 전송은 별도 근거 없이 확대하지 않는다.

우선순위: 최신 사용자 범위 + 본 envelope/gh-sh4-e3-instruction 및 DAG >
revised-minimal-protocol > 원첨부. 원243cell은 참고/역사이며 제출범위가 아니다.

## 3. Source·입력·과학 범위

전용 codex/native-delayed-write-e3-20260924-v1 branch/worktree.
project/run_scripts/native_delayed_write_e3/만 새 구현; 기존 native source read-only.
기존 BASE_ALPHAEDIT42657 / BASE_MEMIT42658, FP32 Llama3-8B-Instruct,
zero-based L4–L8 down_proj, 동일 fixed10k B100×100 blue=False lineage.
최신 upstream/BLUE/local-z/ours로 교체하거나 새 baseline chain을 실행하지 않는다.
AlphaEdit P(KKᵀ+M)+10I, MEMIT 15000C0+KKᵀ/FP64 solve→FP32 등 원 source
identity를 해석의 기준으로 보존한다. MEMIT에 M을 도입하지 않는다.
이번은 endpoint forward/hybrid/hook이며 새 native z fitting/write/history append0.
E0 새 W50→B51/W10→B11 continuation, E2, E4–E6는 NOT_AUTHORIZED.
이전 alpha-key task의 gate/arms를 이번 delayed-write E3에 혼합하지 않는다.

E1: common W0 + family별 W1/5/10/20/30/40/50/60/70/80/90/100 =25logical.
N_diag1000(100case×10), H_diag B1 R100/P200, BaseEval256, GeneralEval128을
outcome 전에 고정 hash/seed/token/reference/exclusion으로 봉인한다.
Active/superseded 및 W0 behavior vs ground truth를 별도로 기록한다.
m=NLL_true−NLL_new, g=−m; true/new completion별 고정 토큰 정렬.
H_sδK와 base/future-write module 분해는 NLL 가산 기여율이 아니다.
Streaming stats/contraction, patch에는 실제 전체 해당 token K를 사용한다.

E3 core4: Alpha(s1,t50/t90), MEMIT(s1,t10/t20).
확장 horizon4: Alpha(s1,t10/t100), MEMIT(s1,t5/t100).
누적 prefix4: Alpha(10,50)/(50,90), MEMIT(5,10)/(10,20).
총12조합은 core 결과 부호/효과/유의성과 무관하게 고정 실행한다.
A=W8(Ws)−W8(W0); s1이면 B1의 L8 성분만, s>1이면 누적 write.
B=W4:7(Wt)−W4:7(Ws), θref=θt−A−B, 나머지 weight 고정.
00/10/01/11; 11 actual endpoint 결과 재사용은 identity exact일 때만.
K11=K01/K10=K00 및 module교차출력=AδK를 실제 측정 envelope로 확인.
actual11 L8 output patch v11−λAδK, λ0/.5/1/−1과 token-RMS matched
rotation/sign-permutation3seeds. seed/위치/수치기준은 scientific outcome 전 lock.
48logicalfactorial +72newmodified(dose3+rotation3)/λ0reuse, job수와 구분.
전체 panel signed paired 효과 주결과, lost-only 보조, cluster uncertainty/sideeffect.
Query-specific 진단 hook은 배포가능한 repair나 새로운 학습 trajectory가 아니다.
Common/L6/단일하위층/activation-transplant는 이번 필수항목 선행조건으로 추가0.

## 4. 자동 DAG와 자원

G00→G10→G20→G21→G30→G31→G40→G50→G51→G60→G70.
정본 e3-dependency-plan.json을 그대로 실행 분해한다.
G00 actual asset/panel/source/runtime/implementation/resource closure.
G10 최소 endpoint fidelity/원 row/hook/L4/repeat-envelope/rollback.
G20 E1, G21 completeness, G30 corefactorial/G31 identity, G40 corepatch,
G50 extensionfactorial/G51 identity, G60 extensionpatch, G70 report.
별도 job은 afterok:<actualID>+instruction/attempt/source/data/panel SHA가
일치하는 atomic PASS를 모두 요구. 같은 job의 연속 stage도 동일계약.
file existence만/실패 exit0 masking/쉘 세미콜론만으로 연결0.
계산결과가 0/역효과/가설반증이어도 technical valid이면 정상 완료이다.

범위 내 필요한 모든 job을 upfront dependency 등록하거나 scheduler가 이어가는
durable continuation을 구현하라. chat agent polling/새 사용자 호출이 등록조건인
구조는 금지. CPU terminal collector를 afterany/equivalent로 구성하여 FAIL/BLOCKED
및 dependent cancellation을 기록한다. invalid dependency 영구대기를 막되,
collector가 과학 gate를 bypass하지 않는다. 정확 해당 task의 failed-dependency job
정리만 허용하며 타 task cancel/hold0.
첫 PENDING/초기 gate 후 중지하는 다른 task 규칙은 이번 자동 DAG를 막지 않는다.
기술 실패 시 구체적 단계/원인/비용/재사용범위·현재 dependency를 즉시 보고한다.

Server4 project cap2, 이번 task GPU 동시1job/1GPU (array이면 %1).
각 GPU job8CPU/host60416MiB 이하/exportNONE/Requeue0, server4 node,
wall/GPUh/peak/output reserve는 구현·기술실측에 따라 구체적으로 lock.
기존 active+admitted pending capacity와 합산하여 cap 준수. 순차 dependency를
중복동시 capacity로 세지 말되 타 project 점유는 제외하지 않는다.
자원 부족 시 pending을 허용하고 전량연결 제출을 유지한다. 다른job 선점/변경0.
프로젝트 cap2를 이 task의 2GPU 동시실행 승인으로 읽지 않는다.
save_checkpoints=false; 기존 W/CP 삭제·이동0, exact new resume bundle 생성0.
진단 row/필요 선택key/통계/receipt는 보존하며 fullvocab/rawkey 무제한저장0.

## 5. 기술 검토·허용 경로·완료

Preflight source/input/model/token/hook/lineage/leakage/held owner/fullargv/
source/resource/dependency/atomicPASS와 fail path를 bounded 검토하라.
Postrun 독립 reducer/schema/row/identity/paired분모/selector없음/비용·재사용분리/
manifest/memberSHA/Markdown 표·링크/코드그림 재현을 검사한다.
첨부CPU12는 기존 toy evidence, 대규모 의례적 CPU/중복GPU gate를 추가0.
Technical block은 중지/수리 대상, 과학적 음성은 warn조차 아닌 valid outcome이다.
실제 자기검산과 독립 reviewer 사용 여부를 구분한다. 공용환경/정책/native 변경0.

허용 tracked paths:
- project/run_scripts/native_delayed_write_e3/
- experiment-reports/servers/server4/native-delayed-write-e3-20260924-v1/
- audits/servers/server4/native-delayed-write-e3-20260924-v1/
- plans/updates/server4/native-delayed-write-e3-20260924-v1/
- messages/acks/server4/2026-09-24-native-delayed-write-e3.md
- messages/server-heads/server4/2026-09-24-native-delayed-write-e3.md
- tasks/status/native-delayed-write-e3-20260924-v1/server4.json
- runs/odeedit_native_delayed_write_e3_s4_20260924/
- transfers/verifications/2026-09-24-native-delayed-write-e3-sh4/
전용 child worktree ignored session-boundary.env는 실제 session/CWD 값으로 허용.
Raw/local root local/native-delayed-write-e3/20260924-v1/, 기존 source/raw 읽기 전용.
정확 허용 runs/status가 generic helper의 과거 pattern에 없으면 해당 narrow limitation을
explicit envelope와 별도 기록하며 helper PASS 위장/공유helper수정0.

산출물: execution-manifest/panel-manifest/job-dependencies, stage gate-result,
fixed-panel/layer-key-drift/factorial/module-interaction/path-patch/active-overwrite
rows, compute.csv, artifact-index, terminal.json, report-ko.md.
Raw/tensor/log/prompt Git0. 불필요 전체broadcast 대신 NO_BROADCAST_NOT_REQUIRED
근거와 receipt 기록; source/report는 검산 후 ownscope nonforce main 통합 허용.
실제 E3완료/기술실패·missing을 구분한 compact terminal을 GH에 direct 보고한다.
SH4는 factual-only, 별도 기전적 해석과 최종 claim은 GH 소유다.

## 6. 초기 회신

수신 nonce/실제 session/CWD/accepted turn과 15개 receiver SHA검증 receipt를 먼저
짧게 회신하라. 등록 전이면 IMPLEMENTING_NOT_SUBMITTED/job_ids=[]를 명시한다.
이 ACK는 실험 완료가 아니며 승인scope 작업은 계속한다.
chain 등록 뒤 actual jobID/dependency/resource/source/lock 표를 별도로 전달한다.
장기진행은 scheduler에 맡기고 E3완료/기술 실패를 별도 보고한다.
새 사용자 허가 없이 E2/E4–E6나 후속 method/10k 확장0.

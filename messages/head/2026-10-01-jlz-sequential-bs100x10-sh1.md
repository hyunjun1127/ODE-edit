# GH에서 SH1로 JLZ sequential 실행 배정

Instruction / ACK nonce: `ODEEDIT-GH-SH1-JLZ-BS100X10-20261001-R1`.
Source handoff: `ODEEDIT-JLZ-BS100X10-TO-GH-20261001-R1`, research session01a0f6b9-74e5-7683-a798-029e477c29b1의 사용자 명시 실행·전달 요청.
Target: registered SH1 `01a04939-f93a-7b50-bca0-65438eab2062`, repository hyunjun1127/ODE-edit.
App-server direct resume에서 실제 CWD `/mnt/raid5/janghj/ODE-edit`, idle을 확인했다. registry의 과거29e4 worktree 경로를 현재 boundary로 가정하지 않는다.

## 즉시 수행할 승인 범위

이것은 새 server1 task의 구현·CPU/실제 BS100 최소 preflight·Slurm 제출·B1→B2 초기확인·자동10회 실행/수집 승인이다. 사용자 재승인을 선행조건으로 두지 않는다.
정본 `plans/global/2026-10-01-jlz-sequential-bs100x10-v1/design-ko.md`, `contract.json`, `cells.csv`를 전체 읽고 hash를 확인한다. 권한 순서는 사용자 fixed-budget override → 이번 계약/envelope → pilot v2 → 첨부 v1이다.

Pilot은BS2/120oracle의 후보fit만 확인했고 commit0/history0, job56668 FAILED2:0은 당시 convergence-only 정책의 정상budget stop이다. 원문/코드/원raw는 보존한다. 이번에는 finite/clamp/새반환평가/정합을 만족한 미수렴 반환후보도 반영한다. NONFINITE/feasibility/commit parity/identity/IO 오류는 entry 복구 후 중단한다. 미수렴을 수렴으로 재명명하지 않는다. 단일JLZ만 first1000 BS100×10, freshW0/H0 시작이며 중간W0reset/요청제외/nativefallback/newbaseline/heavysweep/Qwen0.

## 입력과 source 소유권

GH는 root의 untracked pilot 소스/설계/소형결과를 독립 clean branch로 exact 채택했다. 원 root의 unrelated dirty/deleted를 stage/reset/stash/revert하지 않았다.
`audits/global/2026-10-01-jlz-sh1-dispatch/input-manifest.json`의 SHA/bytes와 canonical copied sources를 확인한다. 원 `project/run_scripts/jlz_pilot/`, `jlz_ref/`, `jlz_smoke/`는 read-only reference다. 새 구현은 `project/run_scripts/jlz_sequential/`에 둔다. 원 첨부의 반대KL 및 오래된cap/수렴-only commit이 새코드로 새어 들어가지 않도록 한다.

실제 local read-only 자산은 사용자 handoff 및 contract의 절대경로다. 새 worktree의 local/에 자동으로 있다고 가정하지 않는다. Python EasyEdit/.venv(2.9.1cu128/4.57.1), pinned Llama snapshot, fixed10k, contexts, C0, 원 native compute_z와 기존 canonical evaluator의 필요한 source/import closure만 사용한다. Offline load, download/C0 재계산/공유환경수정/대형전송0. Server3 job·source·로그·결과·모니터링은 재조회조차 하지 않는다.

GH clone root boundary는GH세션용이므로 SH1 ID를 맞추기 위해 root설정을 덮지 않는다. 새 전용 clean branch `codex/server1-jlz-sequential-bs100x10-20261001-v1`와 worktree를 만들고, 그곳 ignored `servers/local/session-boundary.env`에 실제 SH1 session/CWD/repo/server1만 결속할 권한을 준다. Shared git identity/agent.*은 변경하지 말고 command-scoped/worktree-scoped identity를 사용한다. 기존 stopped historical 이관 task는 재개하지 않는다.

## 구현과 사전 사후 red 검토

전체10batch runner/반환후보 admission/transaction/history/observer/reducer/launcher를 구현한다. 작은 CPU fixture를 production 함수에 연결해 old21 회귀와 새 fixed-budget policy, BS100 shape/가중치, 비활성 요청, 서로 다른 길이·multi-token, NONFINITE trial 뒤 final finite, line-search last accepted vs rejected, commit 또는 observer 예외 rollback, append중간실패/중복, RNG/context/ledger9연결, metadata 직렬화를 검토한다. 수리 전후 diff와 실패 fixture를 보존한다.

독립 bounded red reviewer에게 새source·계약만 읽게 하여 pre-submit과 postrun CPU gate를 받는다. Pre: 목적/KL/C0/rounding/native정합, budget120의 최종call예약, original convergence-only branch제거, source/import/전체분모, cap/memory/noCP/허용경로를 점검한다. Post: 원per-case→NLL/TF 독립재집계,10commit/50layerappend/9link/비용·실패·상태/원자료보존/보고주장의범위를 점검한다.
Reviewer는 science나과거source를 수정·threshold완화·추가GPU실행·직접Gitpush하지 않는다. SH1이 수리·통합한다. red가 실제 미수행이면 PASS라고 쓰지 않는다. 이러한 review는 기능 품질 gate가 아니며 설계 외 반복GPUaudit를 만들지 않는다.

실제 BS100 preflight는 새 계약대로 최대6 whole-batch 비교oracle만 허용한다. PilotCPU21 또는 BS2actual을 BS100 PASS로 확대하지 않는다. 효율화는 같은 FP32 materialization과 loss/gradient 의미 확인 후만 적용하며 nonselected모델을 훈련하거나 BF16로 바꾸지 않는다. 최소 실제비용/peak를 남기고 120whole-batch×10 예상은 관측치와 분리한다.

## 제출 자원과 보존

Slurm write는 이JLZ GPU job(통합 preflight+singleRAMchain 선호) 및 CPU afterany collector만 승인한다. 부득이 분리preflight도 task GPU 동시1, 중복 과학fit0이다. Project cap2(현재local파일), taskcap1, GPU1/CPU8/131072MiB≤183296MiB/devbox, exportNONE/Requeue0, wall48h기본. 실제 partition/maxwall을 확인한다. SH1은 BS100실측의 운영상필요로 요청wall을제출전에조정·봉인할 수 있으나 cap/과학120회/표본/수치임계값은 바꾸지 않는다. wall/ETA/actualallocation/GPUhbudget을 구분한다. 다른 job취소0, admission resource-only 외 타task상태조회0.

순수입력/소스/최대prefixcache와FP64층별streaming·RAMrollback·scalar출력·temp여유를 storage/memory preflight로 확인한다. 저장공간예산16GiB는 출발계획이며 실제계획에맞게봉인한다. 임의자료삭제/공유storagewaiver상속0. 새editedW/H/optimizer/R/전체Δcheckpoint0, 정확resume불가를명시한다.

기술구현오류는 선보고 후 승인된 동일과학식의 narrow수리/newimmutableattempt 재제출 허용, 원source/raw/cost KEEP. 같은오류 반복-to-PASS나 hyperparameter 조정은 금지한다. noCP 실패는 완료prefix를가상재개하지 말고 freshW0재실행이필요한지 exact상태로보고한다. 이번권한은대형무제한재실험 캠페인이 아니다.

## 모니터링과 보고

실제 source/config/data/import를봉인하여 held inspection후release한다. 정상등록된10batch전체프로그램과collector를 제출한다. 대표B1 commit→H다섯append→RPN/TFobserver→B2entry exact연결까지 bounded 관측하고 이후MONITORING_PAUSED_AWAITING_USER로중단한다. Initial미관측/resourcepending·technicalfailure·terminal을분리한다. 이미등록된runner/collector는자연진행. GH에게initial과jobIDs/ETA근거를compact보고하고 longpoll/heartbeat/daemon/자동recall은만들지않는다. 장기완료 상세리뷰는 다음userrecall; 자동collector는원raw검산·결과package를남긴다.

Local root `/mnt/raid5/janghj/ODE-edit/local/jlz-sequential/20261001-v1/`는 새create-once namespace다.
추가 own-scope 쓰기/branch와main nonforce 게시를 승인한다:
- `project/run_scripts/jlz_sequential/`
- `experiment-reports/servers/server1/jlz-sequential-bs100x10-20261001-v1/`
- `audits/servers/server1/jlz-sequential-bs100x10-20261001-v1/`
- `plans/updates/server1/jlz-sequential-bs100x10-20261001-v1/`
- `tasks/status/jlz-sequential-bs100x10-20261001-v1/server1.json`
- `messages/acks/server1/2026-10-01-jlz-sequential.md`
- `messages/server-heads/server1/2026-10-01-jlz-sequential.md`
- `runs/odeedit_jlz_sequential_s1_20261001/` exact namespace.

Raw/prompt/model/teacher/tensor/fullstdoutGit0; NO_BROADCAST_NOT_REQUIRED. 원source/raw/실패/다른task보존. Generichelper가 exact허용경로를거부하면 그사실과envelope권한을분리기록하고공유policy수정0.

최초 회신은 이nonce수신·실제boundary/현재job등록여부·담당ACK를짧게남긴다. 이어정본확보/구현부족/예산·metrics·noCP·resource계획을M0로보고하고 승인범위구현을계속한다. GH의추가ACK나연구sourcechat의응답을작업선행조건으로두지않는다.

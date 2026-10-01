# GH에서 SH1로 JLZ 효율화 구현과 benchmark 배정

Instruction 및 ACK nonce: `ODEEDIT-GH-SH1-JLZ-EFFICIENCY-20261001-R1`.
사용자 승인 handoff: `ODEEDIT-JLZ-EFFICIENCY-TO-GH-20261001-R1`.
Target SH1: `01a04939-f93a-7b50-bca0-65438eab2062`.
Repository: `hyunjun1127/ODE-edit`; 실제 app CWD는 `/mnt/raid5/janghj/ODE-edit`로 direct 확인했다.
관련 active turn `01a0f6f6-6f38-7aa0-98a4-5baf0dc40892`에 전달하되 send 직전 다시 확인한다. 다른 turn으로 바뀌면 unrelated steer를 하지 않는다.

## 실행 권한과 금지 범위

사용자는 구현 효율화 및 실제모델 최소 동등성·성능 실험을 승인했다. 계획이나 ACK만 남기지 말고 source 구현, CPU 검토, actual4요청 및 kernel/BS100 paired benchmark, runner/collector 등록, 초기확인과 사실보고를 수행한다. 추가 GH ACK나 research source chat 응답을 선행조건으로 두지 않는다.

기존56684의 source/config/contract/job/출력은 read-only로 보존한다. cancel/hold/requeue/중간이식/재시작0. 기존 B1→B2 초기확인 의무를 유지한다. 이번 효율화 통신을 그 초기확인 PASS로 간주하지 않는다. SERVER3의 job/source/result/모니터링은 조회·변경하지 않는다. 새1000/10000 과학chain, Qwen, 추가baseline arm, method/hyperparameter sweep는 승인하지 않는다.

정본은 `plans/global/2026-10-01-jlz-efficiency-execution-v1/{design-ko.md,contract.json}`이다. 원 연구7파일은 `plans/global/2026-10-01-jlz-efficiency-v1/`에 exact 채택했다. `audits/global/2026-10-01-jlz-efficiency-sh1-dispatch/input-manifest.json`의7SHA와 reference10source SHA를 확인하고 모두 읽는다. 원 proposal/review는 CPU설계근거이지 실제Llama PASS가 아니다.

## 구현과 비교

과학 의미를 보존하고 새 `project/run_scripts/jlz_efficiency/`에서 구현한다. 기존 `jlz_sequential/`, `jlz_pilot/`, `memit_hj/`, shared native/evaluator를 수정하지 않는다. 필요한 계측 사본/import closure를 새 namespace에 봉인한다. 생산 reference7b4de31d와 benchmark execution SHA를 분리한다.

E0→E1/E2→E3/E4→E5와 독립 cache/sync, 조건부E6를 수행한다. Already implemented prefix/shared W/ours head/관측재사용을 새 절감으로 이중계산하지 않는다. Fixed input/normalization/materialization/solver/teacher/history와 precision 유지가 우선이다. 원 순서 MB2부터 확인하며 stable-length는 이번 최소범위에서 제외한다.

Direct VJP는 원 dX, FP64 adj, overflow-domain과 same-candidate reference 판정이 필수다. CPU 반례를 숨기지 않는다. 성능/수치 gate가 실패하면 해당 route 제외 및 partial 비용 보고이며 원 science 변경은 아니다. fixed numerical tolerance를 완화하거나 동일 실패를 repeat-to-PASS하지 않는다.

소형은 첫4요청, 최대160 whole-batch oracle의 상세 분할과 short12-call fixture를 계약대로 따른다. Native도 같은4요청만 원 singleton→cached singleton→독립 batching을 비교한다. 최초 singleton teacher/anchor/loss0/첫Adam과 각 요청25candidate/24update 규칙을 보존한다. 원 native/hparams는 실행import와 SHA로 확인한다. HJ SPG나 JLZ solver를 native로 바꾸지 않는다.

Short12에는 initial·모든 내부reference/history 재확인·final예약1을 포함한다. 별도32 reserve는 standalone qualification/finite 비교만 허용하며 short12를 늘리는 데 쓰지 않는다.

B100은 고정 같은R의 참조/최종후보 각 warmup1+측정3, 총8 oracle다. 추가 MB/route B100측정은 없다. Full R100/P200/N1000 observer는 같은endpoint에서 별도 장부로 측정하며 과학성능 feedback0. 모든 microbatch F/B와 fallback/initial/final을 별도 기록한다. 예산 공유가 안 되는 dynamic route는 배포하지 않는다.

## Boundary와 쓰기 범위

기존 SH1 작업을 clean하다고 가정해 변경하지 않는다. 새 branch `codex/server1-jlz-efficiency-20261001-v1` 및 전용 clean worktree를 만든다. 원 root와 연구 chat worktree의 dirty/untracked/deleted는 reset/stash/revert/stage하지 않는다. SH1 실제 session/CWD/repo/server1만 새 ignored session-boundary.env에 쓰고 GH root boundary나 공유 user/agent 설정을 덮지 않는다. Command/worktree-scoped identity만 사용한다.

다음 own-scope source/report 및 nonforce branch/main 게시를 승인한다.

- `project/run_scripts/jlz_efficiency/`
- `experiment-reports/servers/server1/jlz-efficiency-20261001-v1/`
- `audits/servers/server1/jlz-efficiency-20261001-v1/`
- `plans/updates/server1/jlz-efficiency-20261001-v1/`
- `tasks/status/jlz-efficiency-20261001-v1/server1.json`
- `messages/acks/server1/2026-10-01-jlz-efficiency.md`
- `messages/server-heads/server1/2026-10-01-jlz-efficiency.md`
- `runs/odeedit_jlz_efficiency_s1_20261001/` exact namespace
- local create-once `/mnt/raid5/janghj/ODE-edit/local/jlz-efficiency/20261001-v1/` 및 전용 worktree ignored local scratch.

Read-only 자산은 기존 SH1 model/data/context/C0/runtime와 exact7b4de31d closure다. Offline 모델이며 download/C0재계산/shared env수정/대형전송0. `inspect_costs.py`는 옛mutable56684로그를 읽으므로 전체 재실행하지 말고 저장evidence를 검산하거나 task-local입력으로 CPUlayout부분만 분리한다. 기존 log polling을 효율화조사 명목으로 늘리지 않는다.

## 자원과 Slurm 권한

Slurm submission은 allowed다. 새 GPU task 동시1, server1 projectcap2이며 기존56684와 admitted pending을 포함해 제출 직전 계산한다. 각 GPU1/CPU8/131072MiB(절대 ceiling183296MiB)/devbox/실제gpu partition/exportNONE/Requeue0/wall12h로 봉인한다. wall은 ETA나 hard GPU-hour budget이 아니다. 작은 benchmark를 큰 science chain으로 늘려 wall을 채우지 않는다.

자리가 없으면 기존job변경 없이 정상 PENDING 또는 exact owner resource-only 확인된 cap 점유job에 afterany를 걸어 제출한다. Resource-only admission 외 다른task분석0. 같은GPU 공유0. 신규benchmark는 원56684와 같은node/GPU종류를 확인하되 paired reference/candidate는 반드시 동일 물리GPU다.

8GiB scalar/source/temp 저장계획과 model+effectiveW+gradient/FP64/cache/RAMrollback peak를 제출 전에 산정한다. 실제free/reserve가 부족하면 보고하며 삭제나 oldwaiver로 우회하지 않는다. CPU collector는 GPU0/CPU4/16384MiB/2h/afterany:newbenchmark만 허용한다. Held 상태에서 owner/name/source/fullargv/GPU/CPU/mem/node/export/requeue/wall/dependency를 검사 후 release한다.

새 edited weight/H/R/optimizer/복원용delta checkpoint 저장0, exact_resume=NOT_AVAILABLE. 기존자료 삭제0. RAM 기술 commit/history/restore는 허용하며 과학 chain으로 세지 않는다. Raw/prompt/model/teacher/tensor/fullstdout Git0. Same-host 기술benchmark이므로 `NO_BROADCAST_NOT_REQUIRED`를 기록하고 타server raw전송0.

## 사전 사후 검토와 수리

SH1 owner가 구현을 통합한다. 독립 bounded red reviewer는 source/contract/CPU만 검토하고 preflight에 normalization/teacher/FP32cast/dX/overflow/leftpad/branch/budget/transaction/noCP/source 및 자원격리를 확인한다. Post는 독립 compact reducer로 parity·오차·branch·호출수·분모·측정통계·실패비용·보고 범위를 검산한다. Reviewer는 production수정/GPU추가/threshold완화/직접push0. 실제 독립 reviewer가 없으면 owner검토와 구분해 보고한다.

재현된 구현 오류의 좁은 수리와 newimmutableattempt는 같은 예산/범위 안에서 허용한다. 실패원source/raw/cost는 보존한다. 수치 qualification실패를 기술예외로 바꿔 반복하지 않는다. 원reference도 identity/finite/IO/restore에서 실패하면 benchmark중단·RCA보고, 기존56684는 별도 유지한다.

## 초기 관찰과 결과 인계

기존56684 B1→B2 초기확인을 우선 유지하면서 CPU구현을 진행한다. 이번 task의 모든 최소benchmark와 collector를 사전 등록한다. 실제 소형qualification 및 첫유효benchmark결과까지 bounded 관측하고 남아있으면 MONITORING_PAUSED_AWAITING_USER로 중단한다. 자원부족 PENDING은 정상등록·release와 실제근거를 확인한 후 pause할 수 있다. 초기경계 전에 전체짧은benchmark가 끝나면 CPU검산·최종보고/main게시 후STOP한다.

Pause뒤에는 scheduler/log/result polling·heartbeat·daemon·자동recall0. 등록runner/collector는 자연진행하며 결과·manifest·사실표를 만든다. 미회수 terminal 상세리뷰는 사용자recall 때 한다. 결과는 수치/시간/선정·제외 사유만 보고하고 아직하지않은1000chain 품질/속도확인을 주장하지 않는다. GH는 ACK수집 후 장시간 실험완료를 기다리지 않는다.

위 pause는 새 효율화 task에만 적용한다. 원56684의 B1→B2 초기확인 의무는 그 task의 원 권한으로 계속하며, 효율화 pause가 원task 의무를 취소하지 않는다.

먼저 이번 nonce를 포함한 실제 담당 ACK와 `RECEIVED/IMPLEMENTING/NOT_SUBMITTED` 상태를 같은 turn commentary로 보내라. 이어 FULL_READ/M0에서 source/미구현/정확budget/자원/원task초기확인보존을 보고한다. 추가 승인 요청이 아니라 사용자 승인 후속의 실제 착수 지시다.

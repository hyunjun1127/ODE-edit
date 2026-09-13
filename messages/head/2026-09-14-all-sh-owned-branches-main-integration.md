# 모든 SH — 본인 소유 미통합 branch main 게시
Policy ID: ODEEDIT-ALL-SH-OWNED-BRANCH-MAIN-INTEGRATION-20260914-V1
사용자 2026-09-14: "다른 branch들도 main에 push하도록 명령하자."

## 실행 권한과 범위
SH1/SH2/SH4는 본인이 소유한 local/remote codex/ branch의 미통합 변경을 확인하고, 검증된 source·tests·raw-free report·manifest·기록을 clean integration에서 origin/main으로 non-force 통합한다.
이것은 Git 검토/필요 CPU 검사/main 통합의 명시적 recall이다. 중지된 실험 자체의 재개/새GPU/Slurm/model/evaluator/remote raw 접근 권한은 아니다.
현재 SH4 seq10 상세보고 작업은 그대로 계속하고 이 통합 의무를 함께 처리한다. 보고서가 아직 작성 중이면 완성된 범위 통합 후 해당 보고는 기존 task 종료 시 통합한다.
GH의 동일 diff/raw 중복 감사나 branch마다 재승인 대기를 선행조건으로 두지 않는다. 간단한 inventory/검사/통합은 직접 수행한다.
기존 dirty source/worktree/user변경은 보존; 깨끗한 별도 codex/<server>-owned-branches-main-integration-20260914-v1 worktree 사용.
허용 source는 그 SH에 이미 명시 위임된 project/run_scripts/ 및 자기 서버 소유 report/audit/status/message 경로뿐이다.
새 global plan/프로토콜/다른 SH 소유 코드로 권한을 확대하지 않는다. Branch 소유가 불명확하면 OWNER_UNRESOLVED로 보고.
main 포함 여부를 실제 최신origin/main ancestry와 patch/file 내용으로 판단한다. 단순 ahead 수를 미통합 변경량으로 오인하지 않는다.

## Branch inventory
자기 서버 관련 branch마다 name/HEAD/tree/기존taskID/소유자/변경파일/원래base/main포함근거/통합결과를 남긴다.
분류: ALREADY_IN_MAIN, INTEGRATED, PARTIALLY_INTEGRATED, NOT_READY, CONFLICT_REQUIRES_DECISION, OWNER_UNRESOLVED.
동일 패치가 merge/cherry-pick으로 이미 포함된 branch는 ALREADY_IN_MAIN 근거를 남기고 중복merge0.
필요 scientific source가 실행완료인데 아직 별도branch만 있다면 runtime source를 exact pin하고 검사 후 통합한다.
실험 전체 미완료라는 이유만으로 독립적으로 검증·완료된 source/report까지 누락하지 않는다. PARTIAL/technical-failure 결과도 사실대로 표시한 완성 report는 게시할 수 있다.
반대로 작성중파일/미검증adapter/깨진tests/진행중runtime변경을 억지로완성시키거나 완료·valid로승격하지 않는다.
미통합 잔여마다 구체적파일/사유/필요한사용자판단만 보고; 이번Git작업을완료하려고새GPU실험/튜닝/평가를시작하지 않는다.

## 안전 통합
1. 실제server/session/CWD/repo경계 및 최신origin/main을 확인. 실행 source/analysis source/publication SHA는 서로 구분한다.
2. 각candidate 실제diff와raw-free/source provenance를 읽는다. 기존성공checks는재사용하고 충돌가능부분만 focusedCPU/compile/shell/diff/access검사.
3. Raw tensor/checkpoint/model weights/dataset/prompts/cache/fullstdout/credentials/SSH/configsecret는Git금지. 단순경로·SHA·집계만.
4. 실행 중·봉인된 원본worktree/branch/source를수정하지말고통합worktree에서처리한다. destructive reset/checkout/branch삭제0.
5. 필요한본scope만nonforce merge/cherry-pick; 원본bytes와동작의의미있는변경은source lineage에기록한다.
6. 충돌은base/ours/theirs의실제의도와소유권이명확한범위에서만해결한다. blanket ours/theirs,다른SH변경되돌림0.
7. Push 직전 fetch. 다른SH가main을앞서갱신하면그commit을보존하며재통합/검사후일반push한다. forcepush0,주기적인remote pollingloop0.
8. 원문Markdown hardbreak/CRLF/manifest mode같은형식은필요한예외를정확히기록하고sealed bytes를무단정규화하지 않는다.
9. CPU검사/코드PNG가필요하면직접코드실행. GH/SH scientific claim이나metric값을통합편의를위해바꾸지 않는다.
10. Branch/archive/worktree 정리·삭제는이번권한아님. 저장공간정리도하지 않는다.

## 제출물과 종료
각 SH 허용신규경로:
audits/servers/<server>/2026-09-14-owned-branches-main-integration/
messages/acks/<server>/2026-09-14-owned-branches-main-integration.md
messages/server-heads/<server>/2026-09-14-owned-branches-main-integration.md
tasks/status/<server>/2026-09-14-owned-branches-main-integration.json
그외이전에위임된source·자기report경로만.
branch-inventory.csv와한글integration-report.md: 포함/이미포함/미완료/충돌branch,검사범위,최종mainHEAD/tree,reportSHA,잔여목록.
Red pre/post: 소유권/scope,검사재사용근거,raw-free,main기존내용보존,미통합잔여정직성. 단순Git작업으로불필요subagent추가0.
Slurm permission NOT_ALLOWED, GPUcap2 불변, 신규GPU/model/evaluator/monitoring0.
Artifact broadcast NO_BROADCAST_NOT_REQUIRED:Git control/source/report only, 새rsync/remote raw0.
완료scope nonforce mainpush후remoteHEAD/tree/cleanintegration/aheadbehind를확인하고GH로보고한다.
모든branch가통합됐다는주장은실제inventory결과로만. 미완료/충돌이있으면전체통합완료라고하지않는다.
종료 TASK_COMPLETE_STOP. 기존모니터링pause유지; SH4의명시승인된현재seq10상세report는그완료까지계속한다.

## 대상 경계
SH1: server1/devbox, session01a04939-f93a-7b50-bca0-65438eab2062, /mnt/raid5/janghj/ODE-edit.
SH2: server2, session01a0493a-074c-7f91-9a13-769116326fef, /mnt/raid5/janghj/ODE-edit.
SH4: server4, session01a04939-b5c7-7a03-ba2d-ef3343d62cfd, /data/janghj/ODE-edit.
共通 repository hyunjun1127/ODE-edit. 원격서버작업은해당SH소유; GH가직접main실험code를중복통합하지않는다.

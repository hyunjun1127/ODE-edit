# SH1 소유 branch main 통합 기록 — 2026-09-14

정책: `ODEEDIT-ALL-SH-OWNED-BRANCH-MAIN-INTEGRATION-20260914-V1`.
이번 작업은 Git 게시와 CPU 검사이며, 실험 재개·성능 재평가가 아니다.
**검증된 SH1 범위를 통합했으며 모든 과거 branch가 통합된 것은 아니다.**

## 경계와 기준

- server1/devbox, session `01a04939-f93a-7b50-bca0-65438eab2062`.
- repository `hyunjun1127/ODE-edit`; 원 root `/mnt/raid5/janghj/ODE-edit`의 dirty 파일과 기존 worktree를 보존했다.
- clean integration: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-owned-branches-main-integration-20260914-v1`.
- branch: `codex/server1-owned-branches-main-integration-20260914-v1`.
- fetch 기준 main `7d5bae2e3be8dba87c92b86a727d5de9ed549af3`, tree `32f695a3cae93a0bef391093ba74990bba6bc1e4`.
- source 통합 checkpoint `b3b40db422bdb4d4e19ed04fca535b9d903ce8ec`.
- 최종 report-bearing main HEAD/tree 및 이 문서 SHA는 동반 `main-publication.json`과 최종 direct ACK에 결속한다. 문서가 자신의 commit/hash를 재귀적으로 포함하지 않는다.
- 최신 정책 전체와 PROTOCOL을 읽었으며 SHA는 `inventory-before.json`에 기록했다. session-boundary PASS, worktree-local identity=head-server1/server-head/server1이다.

## Branch inventory

`branch-inventory.csv`는 이 clone의 SH1 소유 근거가 있는 local/remote 81 refs를 기록한다. 동일 local/remote 및 통합 branch가 있어 고유 HEAD는 66개다. detached execution worktree는 branch 수에 넣지 않았고 변경하지 않았다.

| 분류 | ref 수 | 의미 |
|---|---:|---|
| ALREADY_IN_MAIN | 63 | ancestor, patch-equivalent cherry-pick 또는 전체 변경 blob/mode 동일 |
| INTEGRATED | 16 | 이번에 잔여 SH1 범위를 반영, 후속 main 수정은 보존; A의 SH2 부분도 owner가 게시한 main을 병합해 해소 |
| PARTIALLY_INTEGRATED | 0 | push 전 SH2 게시를 반영하여 종전 2 refs의 잔여 해소 |
| NOT_READY | 1 | BGODE-FBP F0-R1의 미해결 필수 수정 |
| CONFLICT_REQUIRES_DECISION | 1 | 과거 PIR-U H diagnostic과 최신 runtime 충돌·원 worktree dirty |

단순 ahead 수는 미통합 근거로 쓰지 않았다. 초기 tree/patch 비교와 원래 base의 보존된 reflog를 함께 기록했으며, 존재하지 않는 과거 task ID/base는 추정하지 않고 미추출/미기록으로 표시했다. `publish-p1r52-piru-postenergy-warn-r1*` 두 ref는 author와 server1 source/report 변경을 확인해 소유 미확정 상태를 해소했고 이미 main 포함이었다.

### 이번 게시 범위

| 원 source/branch | 이번 조치 |
|---|---|
| E01 `2d0c7909 → f13ce1ed → b51dcf5a → 58f50a25` | 기존 source와 중복되는 부분은 유지; cold/warm/case/mismatch/performance 사실 패키지, 실행·복구 launcher와 observation-resume 잔여 게시 |
| Multilayer A `ba91f274` | SH1 common/fixture/history/bank/A0/A-OS source/tests 및 완료 A0 부분표·수정 attribution·peer 비교 게시 |
| BGODE S1 `63c95449` | 독립 CPU factual analyzer 게시, syntax/helper 검사. 과거 실험 재분석·방법 승격 없음 |
| P1R54 `c0dc8e3f`, 분석 `3aaf6712` | 봉인된 T2/T3/T5/DIRECT source·test·원 보고서 게시. 후속 runtime 확장을 유지하면서 원 callback/role dispatch를 결속 |
| P1R55 `5f74fde1` | 원 15-file source와 기술수리 계보를 non-force merge. 실험 완료 여부를 이번에 새로 판정하지 않음 |

source/report 변경은 160 files, 23,564,574 bytes이며 전수 SHA/blob/mode는 `publication-checks.json`에 있다. 기존 숫자표·PNG·manifest bytes를 다시 생성하거나 정규화하지 않았다. 숫자/해시 기반 per-case·microstep 표는 포함하되 raw prompt·target 문자열·tensor·모델·checkpoint·full log·credential은 포함하지 않았다.

## 충돌 해결과 실행 source 구분

1. E01 README/continuation/warm_plan은 main에 이미 후속 source가 존재했다. 원 중간 commit을 순차 cherry-pick하며 나온 충돌은 최종 `58f50a25`와 기존 main blob 동일성을 확인한 해당 파일만 보존했다. 이전 functionality로 되돌리지 않았다.
2. `terminal_performance.py`는 SH4가 main에 통합한 schema-normalize와 일반 terminal-batch 검사가 있었다. 이를 보존했다. 과거 실패한 raw `request_order` 접근을 복원하지 않았으며, 이 main 파일을 원 E01 실행 bytes라고 주장하지 않는다.
3. P1R54의 옛 전체 runtime snapshot은 최신 역할과 충돌했다. 원 `c0dc8e3f`의 선택적 callback(기본 None), 허용 horizon5, 고유 역할 dispatch만 적용하고 이후 FZ/realization/P1R55 경로를 유지했다. 새로운 objective/threshold/실험 조건을 만들지 않았다. 원 execution source는 계속 `c0dc8e3f`이고 이번 main은 composition/publication source다.
4. SH2 functional/linear_solve/elastic_qp, track_b, 관련 tests 및 server2 audit는 SH1이 대신 cherry-pick하지 않았다. SH2에 exact peer-direct 소유 분리를 전달했다. 최초 A CPU 검사는 `ba91f274`의 SHA 확인된 SH2 dependency를 읽기전용으로 연결했다. push 직전 SH2가 게시한 main `a2ecac5458d533f841e3f9e696865b07bb9f7956`을 정상 병합했다. 종전 잔여 15 files는 원 `ba91f274`와 모두 byte-exact였고, 외부 worktree dependency 없이 통합 경로의 실제 shared closure로 230 tests를 다시 통과했다 (`postmerge-focused-tests.json`).
5. 옛 launcher의 당시 session/branch/resource lock(cap3 등)은 역사적 source bytes다. 이번 게시가 그 launcher의 신규 제출 승인이 아니며 현재 prospective cap2 정책을 무효화하지 않는다. 신규 제출·현 job resource 변경은 0이다.

## CPU·publication 검사

최종 공통 실행은 **230/230 PASS, skip0, CUDA initialized=false**였다.

| 검사 | count/결과 |
|---|---|
| E01 source/analysis unit tests | 105 PASS |
| Multilayer A/common tests + pinned shared dependency | 74 PASS |
| P1R54/P1R55 및 관련 기존 writer/target-time regression | 45 PASS |
| 통합 특화 source-byte/API/role/default/helper 검사 | 6 PASS |
| 변경 Python AST + py_compile | 54 PASS |
| 변경 shell/bash -n | 8 PASS |
| memory policy audit | 최초 163 files, SH2 main 병합 후 167 files, failure0 |
| 재사용 report manifest의 가용 report-member SHA | 49 references 일치, mismatch0 |
| 전수 publication path/type/JSON/CSV 폭/비밀·raw field 검사 | PASS |
| diff whitespace | PASS, 기존 CRLF/Markdown hardbreak 예외만 허용 |

처음 P1R54 T5 fixture에서 기존 main의 horizon 상한3과 누락된 원 상한5 patch가 드러나, 원 patch 그대로 통합했다. 기존 server4 capacity unit fixture는 devbox hostname에서 실패하므로 CPU fixture에서만 hostname을 mock했다. 실제 host/cap을 바꾸지 않았다. audit 전용 호출의 PYTHONPATH·상수 import·TestLoader root 문제도 해당 검사 코드/호출만 고쳤다. 실험 source의 수치 tolerance를 변경하지 않았다.

재현 명령:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /mnt/raid5/janghj/EasyEdit/.venv/bin/python audits/servers/server1/2026-09-14-owned-branches-main-integration/run_cpu_checks.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /mnt/raid5/janghj/EasyEdit/.venv/bin/python audits/servers/server1/2026-09-14-owned-branches-main-integration/run_cpu_checks.py --local-shared
python3 audits/servers/server1/2026-09-14-owned-branches-main-integration/publication_checks.py
python3 scripts/slurm_memory_policy.py audit
```

`publication_checks.py`는 본 task base→봉인 source checkpoint `b3b40db4`의 Git bytes를 검사한다. 이후 main 갱신이나 audit 자기참조 때문에 검사 대상을 바꾸지 않는다. 원 실행 asset 재다운로드/remote raw 접근/모델 평가를 수행하지 않는다. audit 자체 추가 파일은 별도 staged access/compile/format 검사로 확인한다.

기존 access helper는 `tasks/status/<task>/server1.json` 패턴만 허용하여 이번 사용자 명시 경로 `tasks/status/server1/2026-09-14-owned-branches-main-integration.json` 한 파일에서 exit7이었다. 이번 정책의 exact 허용 경로를 우선 적용하는 단일 scope 예외이며, 공통 access helper·역할·다른 경로 권한은 변경하지 않았다. 나머지 staged 경로는 helper 범위 안이다.

## 보존·잔여와 필요한 판단

- **BGODE-FBP F0-R1 4891906a: NOT_READY.** 기존 GH `F0_R1_HOLD_F0_R2_REQUIRED`의 전체-W0 restore, observer accounting, device-order, receipt typing/mode 등의 미해결 결함을 재확인했다. 21 files는 원 branch에 그대로 남겼다. Git 통합을 끝내려고 F0를 새로 수리하거나 과학 task를 재개하지 않았다.
- **PIR-U H diagnostic 1d083ede: CONFLICT_REQUIRES_DECISION.** `p1_runtime.py`, `p1r52_sequential_contract.py`, `p1r52_sequential_runtime.py`가 최신 main과 충돌한다. 원 worktree에는 추가 미커밋 3 files도 있다. 10 committed files를 남겼다. 이후 명시적 정합화 범위 결정 또는 archive 유지가 필요하며 blanket ours/theirs로 해결하지 않았다.
- **Multilayer A의 SH2 부분 15 files: 해소.** SH2 own-scope main 게시를 병합했고 원 source bytes와 모두 일치했다. 새 서버2 runtime 변경을 SH1 작업으로 주장하거나 덮어쓰지 않았다.
- `p1r52-llama-seq-10xb100-fourarm-v1`의 committed HEAD는 이미 main에 있지만 6 tracked dirty files는 별개로 보존했다. PIR-U의 3 tracked dirty files도 미게시다. 이를 branch HEAD의 ALREADY_IN_MAIN과 혼동하지 않는다.
- 각 잔여 파일의 정확 경로·HEAD·소유·사유는 `remaining-files.csv`; 원 worktree dirty 목록은 `inventory-before.json`/`branch-inventory.csv`에 있다. untracked 사용자 자산은 stage하지 않았다.

## 과거 보고서와 현재 상태

이번에 게시한 A0 보고서의 A-OS 미완료 표기는 당시 snapshot이다. 이후 완료분 A-OS CPU 리뷰는 이미 main의 `experiment-reports/servers/server1/multilayer-joint-edit-a-2026-09-11-v1/middle-aos-review-recall-v1/`에 별도로 있다. 이번 작업은 그 과학 검사를 다시 하지 않았다.

E01의 최신 Middle/Late full-seen 완료 리뷰는 이미 main의 `experiment-reports/servers/server1/baseline-mechanism-first-e01-2026-09-12-v1/completed-middle-late-review-v1/diagnostic-report-ko.md`이다. 앞선 cold/warm/performance 부분 패키지는 역사적 관측 범위 그대로 추가했으며 전체 E01 완료로 승격하지 않았다.

새 GPU/model/evaluator/Slurm/remote raw/rsync/monitoring/새 PNG 생성 모두 0. 기존 실험 pause 유지. `NO_BROADCAST_NOT_REQUIRED`, `scientific_promotion=false`. Git 통합 요청의 검증 가능한 범위를 게시하고 잔여를 보고한 뒤 `TASK_COMPLETE_STOP`.

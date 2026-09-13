# SH4 소유 branch main 통합 보고

Policy ODEEDIT-ALL-SH-OWNED-BRANCH-MAIN-INTEGRATION-20260914-V1. 사용자 지시와 seq10 recall을 정독하고 별도 clean child에서 Git/CPU 작업만 수행했다. Shared checkout, 실행 소스/아카이브, 모든 raw, 타 SH 변경은 보존했다. 신규 GPU/model/evaluator/Slurm/원격 raw/삭제0.

## Inventory와 통합 결과

83 local/remote refs(동일 branch의 local/remote 중복 포함)를 검사했다. 초기 main7d5bae2 기준62 ALREADY_IN_MAIN, 이번 후보16 INTEGRATED,5 PARTIALLY_INTEGRATED. 숫자는 실험 개수나 고유 branch 계열 수가 아니다. 실제 ancestry, changed-file blob, git cherry patch equality로 판단했고 ahead 숫자만 쓰지 않았다. 각 HEAD/tree/base/task/잔여 파일은 branch-inventory.csv 및 branch-details.json에 완전 수록했다.

- BLUE helper1075540: 원본 BLUE sequential helper8files 게시. 사용자 local-only L4/L8 hook은 이동하지 않았다.
- Official lifelong85a05d0:16files(13신규/3기존 additive 변경) 게시. 현재 최신 후속 source만 사용했다.
- FzCB completion9b57693: source/tests/봉인 K0 failure report19files를 Git merge로 보존했다. 과거 KILL 상태 그대로이며 새 실행이 아니다.
- FzCB f5f7a78 및 조상:65files source/초기 HOLD/TECH-R1 numerical-failure reports 게시. 생산 prototype 코드가 있다는 사실을 실제 성공 실행으로 승격하지 않는다. 과거 보고서의 main_push=0은 당시 사실이므로 수정하지 않았다.
- Seq10 execution5e96dcb와 completed review94ce2b0/6ae22bb: 실제60batch/18CP/54links/9000z 상세표·7PNG·보고서 게시. 뒤의93c3e4f admission/release helper2files와 사전 기록3files도 exact bytes 게시. Numerical execution5e96과 control93의 SHA를 별도 기록했으며 job source/archive 변경0.

## 남은 5 refs

1. BLUE 초기 v1/tech-r1의 evaluation/runtime/test 차이는1075540의 검증된 후속 수정 전 버전이다. 과거 코드를 최신 파일 위에 되돌리지 않았다.
2. FzCB joint-b1-analysis-v1의7파일은 후속 TECH-R1/f5f7a78과 다른 구형 경로다. 최신 controller-validity와 failure evidence를 유지하며 오래된 내용을 자동 혼합하지 않았다.
3. Official lifelong initial local/remote2refs의 probe/test 차이는85a05d0의 stock track='out' 반환형 수리 전 코드다. 최신 source만 통합했다.

모두 원본 branch/worktree에 보존했다. 잔여 exact 경로는 branch-details.json에 있다. 과거 버전의 모든 bytes가 최신 main에 동일하게 있다는 주장은 하지 않는다. 구형 동작을 복원/혼합하는 새 결정은 이번 통합에서 하지 않았다. 신규 실행 승인을 요구하거나 새 실험으로 잔여를 채우지 않는다.

## 검증 및 한계

- Seq1055 focused CPU,7PNG byte-identical reproduction,40-member package full SHA,462output raw SHA와18CP CPU audit 재사용. Control repair 후 submission/cost6tests 추가 PASS. 원래 runtime numerical math 수정0.
- BLUE6tests, FzCB completion7tests, FzCB initial/TECH/production31tests PASS.
- Lifelong10tests: 처음 기본 환경에서는 easyeditor.util.device 부재로2ERROR. 원래 pinned EasyEdit-stock-14cea824를 task-local PYTHONPATH에 연결하자10PASS. Shared환경 수정/의존성 설치0. device.py SHA e67f0c34c52dda457468e0e51cfb7792156eea1af756d73132a2694593e4117b.
- 확장 unittest discovery는40개 통과 뒤 pytest 부재 module-import1ERROR였으며 전체 PASS라고 하지 않는다. 기존 uv cache의 독립 pytest 환경(-9qh7DnerVmBLAyH/lib/python3.12/site-packages)을 task-local로 연결한 observer9tests는 별도로 PASS. 중간 단독 pytest path는 pluggy 부재로 실패했으며 수치/소스 변경으로 우회하지 않았다.
- FzCB 기존3package member SHA/size와 rooted manifest를 검증했다. 옛 raw/GPU 상태를 새로 검산한 것은 아니다.
- Source compile/shell syntax와 자체 명시 allowlist/raw-free 검사. Generic access helper의 과거 runs 거부는 EXPLICIT_TASK_ENVELOPE_ALLOWED 예외로 기록했으며 helper를 수정하지 않았다.
- CSV CRLF는 csv.writer의 봉인 bytes이므로 cr-at-eol로 검사했다. FzCB 원본13파일의 EOF 빈 줄 경고는 원본 byte preservation 예외다. 결과/hash/원문을 formatting 목적으로 수정하지 않았다.
- Main a2ecac5의 다른 SH 통합은 그대로 merge해 보존했다. Own-scope diff는 이 최신 remote 기준으로 검사한다. Force/reset/branch삭제/일괄 ours-theirs0.

## 결과와 종료

Seq10 보고서: experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/diagnostic-report-ko.md SHA97d0fe587ac864ecd8db21debbfee246dddb8819d2b066d2bf79a0742c3fb492. Source/report science promotion=false, claim_decision=PENDING_GH_REVIEW.

Inventory의 integration_candidate는 이 기록을 넣기 직전의 payload commit이다. 자기 commit SHA를 자기 파일에 넣는 순환 정의를 피하며 실제 push 뒤 최종 main HEAD/tree/ahead-behind와 본 보고서 SHA는 GH peer-direct 및 local completion receipt에 기록한다. 별도 실험 monitoring은 재개하지 않는다. 검증된 범위 main 게시 후 TASK_COMPLETE_STOP.

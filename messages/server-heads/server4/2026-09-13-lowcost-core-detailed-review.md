# SH4 low-cost core 상세 리뷰 완료

Instruction: ODEEDIT-S06-LOW-COST-WRITE-DONOR-CORE-DETAILED-REVIEW-SH4-V1.
Job46451 COMPLETED0, 여섯 core endpoint 및 저장 gate/terminal/state/eval 검산 완료.
과거 PENDING/NOT_YET_RUN 관측은 그대로 보존한다.

정본: experiment-reports/servers/server4/low-cost-write-donor-pilot-2026-09-13-v1/core-completed-review-v1/diagnostic-report-ko.md.
첫 표 first-core-table.csv 및 최종 analysis-manifest.json/rooted-receipt.json을 함께 제공한다.
Current R/P/N 분모100/200/1000. N4=100/197/711, S875=100/196/714,
S75=100/195/718, FULL8=100/199/709, RES8=100/197/712, REFIT4=100/197/716.
Historical NS는 같은 순서883/883/883/888/885/883 (분모1280).
RES8/N4 historical NS는11lost/13gained, historicalPS는1lost/0gained이다.
MMLU32 정답19/19/18/19/18/19. 원표의 모든 손실·NLL tail·참고선 초과를 보존했다.

12신규 CPU tests와 raw/표/PNG5재현/full package rehash PASS. 실행source7ece056과 분석source를 구분한다.
원본runtime 변경/새GPU/evaluator/후속audit/suffix0. Claim decision은 GH 소유이며 PENDING_GH_REVIEW다.
본 완료 scope만 non-force main 통합하며 exact HEAD/tree/SHA는 direct 최종 handoff에 기록한다.
TASK_COMPLETE_STOP은 core 리뷰 완료이며 전체 pilot 완료나 후보 승격이 아니다.

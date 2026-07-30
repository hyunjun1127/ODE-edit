# ODE-Edit

`ODE-Edit`는 sequential knowledge editing에서 layer-synchronous,
capacity-aware edit flow의 motivation이 성립하는지를 Session 01 — Motivation
Validation으로 먼저 검증하는 독립 연구 repository다. 이 저장소는 기존 프로젝트의 patch 공간이
아니며, `project/proposals/00.proposal`을 출발점으로 삼는다.

현재 claim 상태는 **hypothesis only**다. proposal의 방법·성능·문헌 해석은
실험으로 검증되기 전의 연구 입력이며, Session 01 — Motivation Validation의
canonical 판정 기준은
[`plans/global/2026-07-30-session-01-motivation-validation.md`](plans/global/2026-07-30-session-01-motivation-validation.md)에
있다.

## Canonical 경로

- 운영 규칙: [`PROTOCOL.md`](PROTOCOL.md)
- 원 proposal: [`project/proposals/00.proposal`](project/proposals/00.proposal)
- Session 01 연구 근거: [`project/proposals/sections/01-motivation-validation.md`](project/proposals/sections/01-motivation-validation.md)
- Session 01 실행 계획: [`plans/global/2026-07-30-session-01-motivation-validation.md`](plans/global/2026-07-30-session-01-motivation-validation.md)
- Session 01 global evidence index: [`experiment-reports/global/2026-07-30-session-01-motivation-validation.md`](experiment-reports/global/2026-07-30-session-01-motivation-validation.md)
- redacted 서버 인벤토리: [`servers/connection-inventory.md`](servers/connection-inventory.md)
- server1 onboarding record: [`servers/active/server1.md`](servers/active/server1.md)
- 실행 스크립트: `project/run_scripts/` (현재 등록된 실행 스크립트 없음)
- raw artifact·dataset·checkpoint·full log·credential: ignored `local/`
- session boundary: ignored `servers/local/session-boundary.env`

## 운영 요약

Git은 plan, instruction, audit, compact metadata, report를 위한 control
plane이다. SSH/Slurm/rsync와 ignored `local/`은 execution plane이다. 실제
server-head가 등록되고 red-team onboarding gate를 통과하기 전에는 실험을
제출하지 않는다. `messages/inbox/<server>.md`의 actionable instruction은
해당 server-head가 sync 후 읽어 실행하며, Git message 자체가 실행기가
아니다.

GPU cap과 host-memory request cap은 ignored `servers/local/gpu-caps.tsv`에
있다. Slurm job은 `scripts/check-slurm-resource-cap.sh <server> <gpus>
<mem_mb>`를 먼저 통과해야 한다. Codex session은 server record에 기록된
session ID, repository CWD, Git identity가 모두 일치할 때만 이 repo를 조작한다.
Actionable session은 먼저 `scripts/check-session-boundary.sh <session_id>`를
통과해야 한다.

원격 저장소와 실제 서버 접속 정보는 이 문서에 기록하지 않는다. raw IP,
username, port, key, token, password와 private dataset secret은
`servers/local/` 또는 `local/`의 ignored private path에만 둔다.

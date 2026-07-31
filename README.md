# ODE-Edit

`ODE-Edit`는 sequential knowledge editing에서 layer-synchronous,
capacity-aware edit flow의 motivation을 독립적으로 검증한 연구
repository다. 이 저장소는 기존 프로젝트의 patch 공간이 아니며,
`project/proposals/00.proposal`을 출발점으로 삼는다.

Session 01 — Motivation Validation은 2026-07-31에
**`motivation_not_supported_cross_model`; `NO MV3`; no retune**으로
종료됐다. MV-0 implementation fidelity와 MV-1 two-model allocation
signal은 통과했지만, MV-2 refreshed-direction effect가 Llama와 Qwen에서
반대 부호여서 cross-model ODE/relinearization claim을 kill했다. Canonical
판정 기준은
[`plans/global/2026-07-30-session-01-motivation-validation.md`](plans/global/2026-07-30-session-01-motivation-validation.md)에
있고 최종 종합은
[`experiment-reports/global/2026-07-31-session-01-motivation-final.md`](experiment-reports/global/2026-07-31-session-01-motivation-final.md)에
있다.

## Canonical 경로

- 운영 규칙: [`PROTOCOL.md`](PROTOCOL.md)
- 원 proposal: [`project/proposals/00.proposal`](project/proposals/00.proposal)
- Session 01 연구 근거: [`project/proposals/sections/01-motivation-validation.md`](project/proposals/sections/01-motivation-validation.md)
- Session 01 실행 계획: [`plans/global/2026-07-30-session-01-motivation-validation.md`](plans/global/2026-07-30-session-01-motivation-validation.md)
- Session 01 global evidence index: [`experiment-reports/global/2026-07-30-session-01-motivation-validation.md`](experiment-reports/global/2026-07-30-session-01-motivation-validation.md)
- Session 01 GH 최종 보고서: [`experiment-reports/global/2026-07-31-session-01-motivation-final.md`](experiment-reports/global/2026-07-31-session-01-motivation-final.md)
- related-work/novelty 경계: [`project/proposals/sections/02-related-work-and-novelty-boundary.md`](project/proposals/sections/02-related-work-and-novelty-boundary.md)
- redacted 서버 인벤토리: [`servers/connection-inventory.md`](servers/connection-inventory.md)
- server1 onboarding record: [`servers/active/server1.md`](servers/active/server1.md)
- 실행·분석 스크립트: `project/run_scripts/`
- raw artifact·dataset·checkpoint·full log·credential: ignored `local/`
- session boundary: ignored `servers/local/session-boundary.env`

## 운영 요약

Git은 plan, instruction, audit, compact metadata, report를 위한 control
plane이다. SSH/Slurm/rsync와 ignored `local/`은 execution plane이다.
`messages/inbox/<server>.md`의 actionable instruction은 해당 server-head가
sync 후 읽어 실행하며, Git message 자체가 실행기가 아니다. Session 01은
사용자 지시와 recorded GH exception 아래 server1에서 완료됐고, `NO MV3`
이후 추가 Motivation job은 제출하지 않는다.

GPU cap과 host-memory request cap은 ignored `servers/local/gpu-caps.tsv`에
있다. Slurm job은 `scripts/check-slurm-resource-cap.sh <server> <gpus>
<mem_mb>`를 먼저 통과해야 한다. GH/SH primary Codex session은 `Sol Ultra`
(`gpt-5.6-sol`, reasoning effort `ultra`)이며 delegated blue/red/analysis
subagent만 `Terra Ultra` (`gpt-5.6-terra`, `ultra`)다. repo-local
[`.codex/config.toml`](.codex/config.toml)과
[`.codex/agents/default.toml`](.codex/agents/default.toml)이 이 split을
고정한다. Primary session은 server record의 session ID, confirmed Sol
profile, repository CWD, Git identity가 모두 일치할 때만 이 repo를
조작한다. Actionable primary session은 먼저
`scripts/check-session-boundary.sh <session_id>`를 통과해야 한다.

원격 저장소와 실제 서버 접속 정보는 이 문서에 기록하지 않는다. raw IP,
username, port, key, token, password와 private dataset secret은
`servers/local/` 또는 `local/`의 ignored private path에만 둔다.

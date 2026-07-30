# GH Session Registry Pre-push Waiver

- 감사 유형: pre-push-sensitive / bootstrap update
- 판정: waived
- 감사 시간: 2026-07-30
- 작성 agent: head-server1-gh (global-head)
- 대상: server별 Codex SH session registry 및 server4 resource-cap update

## Waiver 사유

독립 red-team session은 아직 배정되지 않았다. GH는 이 변경을 역할별 session
registry, CWD, Git identity의 명시와 server4 resource cap correction으로
한정한다. red-team `pass`가 아니며, 이 waiver는 실험 실행·task 승인·Slurm
제출에 적용되지 않는다.

## GH 확인

- server1 GH와 server1 SH, server2/3/4 SH를 별도 registry 행으로 분리했다.
- 미지정 SH ID를 다른 repo session ID로 채우지 않았다.
- server4 cap은 동시 3 GPU, GPU당 65984 MiB로 반영했다.
- staged path/access/whitespace 및 local private path exclusion을 push 직전에
  다시 확인한다.

## 다음 gate

각 server-head Codex session을 만들면 onboarding red audit에서 해당 session ID,
CWD, origin, `servers/local/session-boundary.env` 일치 여부를 반드시 검증한다.

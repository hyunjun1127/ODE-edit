# server4 온보딩 점검 — 2026-08-22

## 범위

- 수행 agent: `server4-server-head` (`server-head`)
- 대상: server4 control-plane onboarding, Git/session 경계,
  Slurm/resource/storage와 Git artifact 정책
- 판정 시각: `2026-08-22T17:51:38+09:00`
- 독립 red-team 감사가 아니라 server-head 자체 온보딩 점검이다. 향후
  scientific task에는 해당 task의 명시적 red-team preflight가 별도로 필요하다.

## 점검 결과

| 항목 | 판정 | 근거 |
| --- | --- | --- |
| Git repository sync | PASS | `git pull --rebase origin main`, conflict 없음, HEAD `96494a05...` |
| Git/agent identity | PASS | `server4-server-head`, `server-head`, `server4` |
| session boundary | PASS | 등록 session/CWD/repository identity checker 통과 |
| protocol full read | PASS | 1,125 lines, SHA256 `a54a4e7...` |
| pending command audit | PASS | `scripts/audit-pending-tasks.sh`: `No pending tasks.` |
| Slurm command access | PASS | `sinfo`, `squeue`, `scontrol`, `sbatch` available; submit은 실행하지 않음 |
| GPU cap config/check | PASS | ignored local cap 2, active project GPU 0, 1 GPU / 65,984 MiB check 통과 |
| CPU/host memory | PASS | 128 logical CPUs, OS memory 503 GiB, available 약 406 GiB |
| storage capacity | WARN | `/data` 사용률 98%, 약 193 GiB 가용 |
| local output boundary | PASS | local boundary/cap config ignored 확인, write probe 삭제, raw artifact Git 추가 없음 |
| destructive/transfer safety | PASS | submit/cancel/retry/rsync/delete 0건 |

## Red/blue 및 실행 경계

- 이번 지시는 control-plane 초기화와 읽기 전용 resource audit 범위다.
- 신규 experiment job은 GH가 명시적으로 금지했으며 실제 제출은 0건이다.
- 향후 scientific task를 받으면 실행 envelope, point-in-time cap/storage,
  data/output boundary, red-team preflight를 확인하기 전 제출하지 않는다.
- raw logs/results/checkpoints/datasets/credentials는 Git에 추가하지 않는다.

## 최종 판정

- `WARN`
- control-plane onboarding과 session/Git/Slurm 경계는 PASS다.
- `/data` 사용률 98%를 storage warning으로 유지한다.
- 현재 상태는 `waiting_for_global_head_task`; scientific submission은 HOLD다.

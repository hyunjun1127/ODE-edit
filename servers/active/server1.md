# server1 active record

## 현재 authority

- server: `server1`
- physical hostname: `devbox`
- host ID: `remote-ssh-codex-managed:lab120`
- repository: `hyunjun1127/ODE-edit`
- 갱신 시각: 2026-08-29 (session authority rotation)

| 역할 | Codex session ID / deeplink | local hard boundary | 실제 CWD | 상태 |
| --- | --- | --- | --- | --- |
| global-head (GH) | `01a04939-8873-7673-8dca-4c7fc5e31af0` / `codex://threads/01a04939-8873-7673-8dca-4c7fc5e31af0` | session ID user-confirmed; app list/read PASS | `/mnt/raid5/janghj/ODE-edit` | active caller |
| server-head (SH1) | `01a04939-f93a-7b50-bca0-65438eab2062` / `codex://threads/01a04939-f93a-7b50-bca0-65438eab2062` | session ID user-confirmed; app list/read PASS | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | app-server direct ACK PASS |

새 session은 2026-08-28 registry의 GH/SH1 authority를 supersede한다. 과거
task, report, audit, receipt와 completed launcher의 session ID는 historical
provenance로 보존한다.

이번 갱신은 session authority만 교체했다. 아래 repository/runtime/Slurm 관측값은
별도 표기가 없으면 2026-08-22 마지막 audit 기준이다.

## Repository 상태

- canonical remote: `https://github.com/hyunjun1127/ODE-edit.git`
- audit 시 `origin/main`: `0d63ad4ec4978be6d04aabb640e17917bd1348d7`,
  tree `652b84489825371558002e24b381dcb777327a7f`
- GH root clone은 detached `858f562cc8282b9ae164d41b85dfad070a8273f2`의
  기존 tracked/untracked dirty 상태다. reset, stash, cleanup 또는 통합에
  사용하지 않는다.
- 이번 registry 통합은 clean worktree
  `/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-session-registry-20260822-v1`
  에서 수행한다.
- SH1 worktree는 detached
  `cdb80bc70032c203334531edb7020ff654f2938d`, tree
  `cc8285949540f70a4afcc0278968c3978068b465`다. tracked clean,
  user-owned untracked path 2개를 그대로 보존한다.

## 연결 및 runtime

- GH에서 SH1 session list/read: PASS
- GH→SH1 app-server direct: PASS; active turn
  `01a04989-e4b9-79e1-978e-c07b21f49330`에 `turn/steer` 후
  `turn/completed` ACK PASS
- current SH1 science task는 중단·변경하지 않았다. 별도 direct turn
  `01a04999-ff23-7823-bf3d-532e33f40b0b`에서 protocol commit
  `6d9e4e625c7ed016742ca3516299eae40d9b4af1`로 ff-only sync PASS,
  clean, ahead/behind `0/0`.
- `PROTOCOL.md` full-read identity:
  SHA256 `a54a4e7c00c36ec9f3b0fe122e5d8735dacc2c21c8604bc598593eb396936663`,
  51,147 bytes, 1,125 lines
- EasyEdit, Python runtime와 Hugging Face Llama/Qwen cache: available
- Slurm query/submit commands: available
- SSH/rsync read-only audit: server2/server4 PASS; server3 SSH PASS이나 target
  repository 없음; server1 loopback alias DNS 실패. 외부 transfer action은 0이다.

## Slurm 및 resource

- audit 시 job `22541`: array 1–3 `COMPLETED(0:0)`, array 0/4
  `RUNNING`; 변경·취소·재제출 0
- active server1 project GPU cap: 4
- 실제 후속 제출 전에는 point-in-time GPU/host-memory cap을 다시 확인한다.

## 판정

- session authority IDs: user-confirmed
- session routing/read: PASS
- app-server direct coordination: PASS (steer/terminal ACK/ff-only sync)
- GH와 SH1 local runtime checker: pending revalidation
- tracked registry update owner: GH clean integration worktree
- SH1 canonical session worktree sync: PASS; 기존 사용자 상태는 삭제·reset하지 않음

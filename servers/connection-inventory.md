# 서버 접속 인벤토리

- 갱신 시각: 2026-08-29 (session authority만 갱신; 물리/runtime 상태는 기존 마지막 audit 기준)
- 작성 agent: `head-server1-gh` (global-head)
- repository: `hyunjun1127/ODE-edit`
- 목적: 새 GH/SH session authority, app-server direct coordination과 안전한
  Git 동기화 경계 공유

이 tracked 파일에는 raw SSH HostName/IP, username, port, key path나 credential을
기록하지 않는다. 실제 접속값은 ignored local-only inventory에 보관한다.

## 마지막 물리 서버 상태

아래 표는 2026-08-22 마지막 audit 관측값이며 이번 session rotation에서 다시
검사하지 않았다.

| 서버 | 실제 host | repository CWD | Git 상태 | audit 시 ODE-edit Slurm |
| --- | --- | --- | --- | --- |
| `server1` GH | `devbox` / `remote-ssh-codex-managed:lab120` | `/mnt/raid5/janghj/ODE-edit` | detached/dirty 사용자 상태 보존; clean registry integration worktree 별도 사용 | `22541_0/4 RUNNING` |
| `server1` SH1 | `devbox` / `remote-ssh-codex-managed:lab120` | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | detached `cdb80bc70032c203334531edb7020ff654f2938d`; tracked clean, untracked 2개 보존 | `22541_1..3 COMPLETED(0:0)` |
| `server2` SH2 | `server2` / `remote-ssh-codex-managed:lab121` | `/mnt/raid5/janghj/ODE-edit` | audit 시 clean `main` `0d63ad4ec4978be6d04aabb640e17917bd1348d7`; registry push 뒤 ff-only sync 대상 | `22600/22601 PENDING` |

Git 업데이트는 clean canonical integration worktree에서만 커밋·push한다. GH root와
SH1 detached worktree의 기존 상태를 reset, stash, revert 또는 cleanup하지 않는다.

## Codex Session Registry

| 서버 | 역할 | Codex session ID / deeplink | hard boundary | Repository CWD | 상태 |
| --- | --- | --- | --- | --- | --- |
| `server1` | global-head (GH) | `01a04939-8873-7673-8dca-4c7fc5e31af0` / `codex://threads/01a04939-8873-7673-8dca-4c7fc5e31af0` | session ID user-confirmed; app list/read PASS | `/mnt/raid5/janghj/ODE-edit` | active caller |
| `server1` | server-head (SH1) | `01a04939-f93a-7b50-bca0-65438eab2062` / `codex://threads/01a04939-f93a-7b50-bca0-65438eab2062` | session ID user-confirmed; app list/read PASS | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | app-server direct ACK PASS |
| `server2` | server-head (SH2) | `01a0493a-074c-7f91-9a13-769116326fef` / `codex://threads/01a0493a-074c-7f91-9a13-769116326fef` | session ID user-confirmed; app list/read PASS | `/mnt/raid5/janghj/ODE-edit` | app-server direct ACK PASS |
| `server3` | server-head | 미지정 | 미지정 | `/data/janghj/ODE-edit` | future target |
| `server4` | server-head (SH4) | `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` / `codex://threads/01a04939-b5c7-7a03-ba2d-ef3343d62cfd` | session ID user-confirmed; app list/read PASS | `/data/janghj/ODE-edit` | app-server direct ACK PASS |

## App-server direct endpoint registry

| 역할 | app host | local Unix control socket | direct 상태 |
| --- | --- | --- | --- |
| GH, SH1 | `remote-ssh-codex-managed:lab120` | `/mnt/raid5/janghj/.codex/app-server-control/app-server-control.sock` | PASS |
| SH2 | `remote-ssh-codex-managed:lab121` | `/mnt/raid5/janghj/.codex/app-server-control/app-server-control.sock` | PASS |
| SH4 | `remote-ssh-codex-managed:lab163` | `/data/janghj/.codex/app-server-control/app-server-control.sock` | PASS |

## Direct coordination verification

| 대상 | 방식 | 결과 |
| --- | --- | --- |
| GH→SH1→GH response | active turn `01a04989-e4b9-79e1-978e-c07b21f49330`에 `turn/steer` 후 `turn/completed` | PASS; current science task unchanged |
| GH→SH2→GH response | `thread/resume` → `turn/start` → `turn/completed` | PASS, turn `01a04994-0ed3-7253-a155-e5aacdaa9943` |
| GH→SH4→GH response | `thread/resume` → `turn/start` → `turn/completed` | PASS, turn `01a04994-02d7-7482-8ec7-4706e96ab414` |

Protocol commit `6d9e4e625c7ed016742ca3516299eae40d9b4af1`의 direct
ff-only 동기화도 같은 transport로 검증했다: SH1 turn
`01a04999-ff23-7823-bf3d-532e33f40b0b`, SH2 turn
`01a04998-3f36-7630-92b1-4738e576fca6`, SH4 turn
`01a04998-37e0-7300-a9e8-91b5f16c03ec`; 모두 clean, ahead/behind
`0/0`.

`send_message_to_thread` dynamic wrapper는 계속 unavailable이며 direct
coordination의 구성요소가 아니다. SH가 GH로 unsolicited push하는 것으로
기록하지 않는다. GH가 exact SH session을 resume/start/steer하고 같은
app-server stream에서 response를 회수한다.

## Superseded current-authority records

2026-08-28 registry의 GH/SH1/SH2 session
`01a04660-f2e5-7583-a9f9-83db20210d72`,
`01a04664-c753-7461-b914-b4ebf24a581a`,
`01a04661-2f02-7d93-b3ba-4c69d113eaee`와 server4 session
`01a028a7-9e3c-7541-81ba-efb40555d17d`, 2026-08-22 registry의 GH/SH1/SH2 session
`01a0278e-3f20-7b12-8c26-cf71deb2708b`,
`01a0278d-456b-7d20-820f-63fc80a57b8d`,
`01a0278d-6e24-7a81-974a-96e4addd7ca5`와 2026-08-17 registry의
`01a00e5f-63ef-7cc2-89ec-f2f7b23df40f`,
`01a00e5d-29e8-7a01-822b-7acf43226035`,
`01a00e5c-f7ae-72a2-98b2-b8b0907168b4`는 inactive/superseded다.
과거 reports, audits, receipts와 completed launchers의 session ID는 실행
provenance이므로 일괄 치환하지 않는다.

## Local Private Inventory

각 agent는 필요 시 아래 파일을 local-only로 유지한다.

```text
servers/local/connection-inventory.private.md
servers/local/ssh_config
servers/local/rsync-targets.tsv
servers/local/gpu-caps.tsv
servers/local/method-runtime.env
servers/local/dataset-roots.tsv
servers/local/session-boundary.env
```

일반 artifact broadcast는 source boundary, resource cap과 peer path를 검증한 뒤
`scripts/rsync-artifact-broadcast.sh`를 사용한다. `--delete`, credential,
private inventory와 민감 경로는 별도 승인 없이 전송하지 않는다.

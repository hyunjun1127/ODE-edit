# 서버 접속 인벤토리

- 갱신 시각: 2026-08-17
- 작성 agent: head-server1-gh (global-head)
- repository: `hyunjun1127/ODE-edit`
- 목적: 현재 GH/SH session authority와 안전한 repository 동기화 경계를 공유

이 tracked 파일에는 raw SSH HostName/IP, username, port를 기록하지 않는다.
실제 접속값은 각 clone의 ignored local-only 파일에 보관한다. 비밀번호, SSH
private key, token, key path, passphrase는 Git에 기록하지 않는다.

## 현재 서버 상태

| 서버 | 실제 host | repository CWD | Git 상태 | ODE-edit Slurm |
| --- | --- | --- | --- | --- |
| `server1` GH | `devbox` / `remote-ssh-codex-managed:lab120` | `/mnt/raid5/janghj/ODE-edit` | GH root는 detached/dirty 보존 상태. canonical `main` 통합·push는 `/mnt/raid5/janghj/.codex/worktrees/odeedit-p2r7-main-publish-v1`에서 수행 | active 0 |
| `server1` SH1 | `devbox` / `remote-ssh-codex-managed:lab120` | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | detached `cdb80bc70032c203334531edb7020ff654f2938d`; tracked clean, user-owned untracked paths 2개 보존; `origin/main=1caed88867db3087fa5db26995c7e3719c064216` 확인 | active 0 |
| `server2` SH2 | `server2` / `remote-ssh-codex-managed:lab121` | `/mnt/raid5/janghj/ODE-edit` | `main` `6145406ae4b11e05b683c46aa604c972eb727f5a`; clean; model gate block로 fetch/ff-only 미실행 | active 0 |

Git 업데이트는 clean canonical `main` 통합 worktree에서만 커밋·push한다. SH1의
detached worktree와 GH root의 기존 dirty state는 reset/revert/delete하지 않는다.
SH2는 새 session-boundary 확인 후에만 `fetch`와 `merge --ff-only
origin/main`을 수행한다.

## Codex Session Registry

아래 값이 이 repository의 현재 primary session authority다. 사용자 메시지에 SH2
ID가 SH1과 동일하게 중복 기재됐지만, SH2 direct ACK와 app host metadata로 실제
SH2 ID를 확인해 정정했다.

| 서버 | 역할 | Codex session ID / deeplink | Required / confirmed model | Repository CWD | 상태 |
| --- | --- | --- | --- | --- | --- |
| `server1` | global-head (GH) | `01a00e5f-63ef-7cc2-89ec-f2f7b23df40f` / `codex://threads/01a00e5f-63ef-7cc2-89ec-f2f7b23df40f` | user-managed; observed `gpt-5.6-sol/xhigh` | `/mnt/raid5/janghj/ODE-edit` | active |
| `server1` | server-head (SH1) | `01a00e5d-29e8-7a01-822b-7acf43226035` / `codex://threads/01a00e5d-29e8-7a01-822b-7acf43226035` | user-managed; observed `gpt-5.6-sol/xhigh` | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | direct ACK PASS; local ID updated |
| `server2` | server-head (SH2) | `01a00e5c-f7ae-72a2-98b2-b8b0907168b4` / `codex://threads/01a00e5c-f7ae-72a2-98b2-b8b0907168b4` | user-managed; observed `gpt-5.6-sol/max` | `/mnt/raid5/janghj/ODE-edit` | direct ACK PASS; ff-only sync authorized |
| `server3` | server-head | 미지정 | `Sol Ultra` / 미지정 | `/data/janghj/ODE-edit` | future target |
| `server4` | server-head | 미지정 | `Sol Ultra` / 미지정 | `/data/janghj/ODE-edit` | registered-pending-clone |

현재 session을 대상으로 actionable instruction을 실행하기 전에는 각 clone의
ignored `servers/local/session-boundary.env`에 실제 session ID, confirmed model,
CWD와 repository identity를 기록하고
`scripts/check-session-boundary.sh <session-id>`를 통과해야 한다. Runtime model
model/profile은 사용자 관리 항목이며 hard boundary가 아니다. Session ID, CWD와
repository identity는 계속 checker의 hard gate다. 개별 scientific contract가
model/profile을 명시적으로 고정한 경우에만 그 task 내부 gate로 적용한다.

## Superseded Session Records

2026-08-17 이전 tracked active registry의 GH/SH authority
`019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`,
`019fc63e-5217-7250-9c22-c5b2ec4248f0`,
`019fc5ec-f85b-7770-a73a-1d19be1cd491`와 각 clone의 old local boundary 값은
inactive/superseded다. 과거 experiment report, audit, receipt, completed launcher에
기록된 session ID는 실행 provenance이므로 일괄 치환하지 않는다.

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

일반 artifact broadcast는 source server의 boundary, resource cap, peer path가
검증된 뒤 `scripts/rsync-artifact-broadcast.sh`로 수행한다. `--delete`,
private inventory, SSH material, credential, 민감 경로와 repo-external 경로는 별도
사용자 승인 없이 전송하지 않는다.

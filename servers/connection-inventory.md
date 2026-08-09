# 서버 접속 인벤토리

- 갱신 시각: 2026-08-09 12:41:38 KST
- 작성 agent: head-server1-gh (global-head)
- 목적: agent 간 SSH/rsync 계획 수립을 위한 redacted 접속 인벤토리 공유

이 tracked 파일에는 raw SSH HostName/IP, username, port를 기록하지 않는다.
실제 접속값은 각 서버 clone의 ignored local-only 파일에 보관한다.
비밀번호, SSH private key, token, key path, passphrase는 Git에 기록하지 않는다.

## 서버 목록

server1의 새 canonical GH와 SH1은 이전 context를 인계받았고, 각 local session
boundary와 repository identity를 확인했다. server2의 새 canonical SH2도 이전 context를
인계받았으며 local session boundary와 Git identity까지 확인해 GH instruction을 받을
준비가 됐다.
server4는 physical host로 등록됐지만 이 repo clone/SH가 없는
`registered-pending-clone` 상태이며 server3는 future target이다.

| Repository server name | Raw connection detail location | 상태/용도 |
| --- | --- | --- |
| `server1` | `servers/local/ssh_config`, `servers/local/rsync-targets.tsv`, `servers/local/gpu-caps.tsv` | GH active on root clone / SH1 active on dedicated worktree; GPU cap 3, memory cap 198117 MiB per GPU |
| `server2` | `servers/local/rsync-targets.tsv`, `servers/local/gpu-caps.tsv` | SH2 active / boundary PASS / ready for GH instruction / clone: `/mnt/raid5/janghj/ODE-edit` |
| `server3` | `servers/local/ssh_config`, `servers/local/rsync-targets.tsv`, `servers/local/gpu-caps.tsv` | future target / clone 전 / Codex session 미지정 |
| `server4` | `servers/local/rsync-targets.tsv`, `servers/local/gpu-caps.tsv` | registered-pending-clone / GPU cap 3, memory cap 65984 MiB per GPU / Codex session 미지정 |

## Codex Session Registry

아래 registry는 **이 repository 전용** session만 기록한다. 각 서버의
global-head와 server-head는 서로 다른 role/session으로 명시하며, `미지정`은
해당 role의 Codex session을 아직 만들거나 배정하지 않았다는 뜻이다. 다른 repo의
session ID를 채우거나 대체 대상으로 사용하지 않는다.

| 서버 | 역할 | Codex session ID | Required/confirmed model | Repository CWD | 상태 |
| --- | --- | --- | --- | --- | --- |
| `server1` | global-head | `019fe491-16f4-7bd3-adf5-4b1eb4a57d1f` | `Sol Ultra` / `Sol Ultra` (`gpt-5.6-sol`, runtime metadata) | `/mnt/raid5/janghj/ODE-edit` | active / 현재 GH session; prior context inherited |
| `server1` | server-head (SH1) | `019fe489-c968-75f3-9965-7cfbc26c0a99` | `Sol Ultra` / `Sol Ultra` (`gpt-5.6-sol`, runtime metadata) | `/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit` | active canonical SH1; prior context inherited; boundary PASS; ready for GH instruction |
| `server2` | server-head (SH2) | `019fe491-954b-70a0-8ba8-0588e9f8d741` | `Sol Ultra` / `Sol Ultra` (`gpt-5.6-sol`, runtime metadata) | `/mnt/raid5/janghj/ODE-edit` | active canonical SH2; prior context inherited; boundary PASS; ready for GH instruction |
| `server3` | server-head | 미지정 | `Sol Ultra` / 미지정 | `/data/janghj/ODE-edit` | future target / clone 전 |
| `server4` | server-head | 미지정 | `Sol Ultra` / 미지정 | `/data/janghj/ODE-edit` | registered-pending-clone |

Canonical SH를 등록할 때 GH는 이 table, `servers/active/<server>.md`, 그리고 해당
clone의 ignored `servers/local/session-boundary.env`에 **동일한** session ID,
confirmed `Sol Ultra` primary model profile, CWD, repository identity를 기록한다.
실제 instruction은 그 ID와 model profile을 envelope에 넣고
`scripts/check-session-boundary.sh <session-id>`를 먼저 실행한다.

Task-local delegated session은 canonical SH assignment를 대체하지 않으며, 해당 envelope와
전용 worktree에만 권한이 있다. 서버를 등록할 때는 `servers/templates/server-onboarding.md`를 바탕으로
`servers/active/<server>.md`를 만들고, 해당 server-head의 heartbeat와
red-team onboarding audit이 `pass` 또는 명시적 `waived`가 된 뒤에만 task를
배정한다.

2026-08-09 사용자 직접 교체로 이전 GH `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`,
SH1 `019fc63e-5217-7250-9c22-c5b2ec4248f0`, SH2
`019fc5ec-f85b-7770-a73a-1d19be1cd491`의 active authority는 종료됐다. 이전 ID가
남아 있는 plan, report, audit, message와 execution lock은 당시 provenance이므로
수정하지 않는다.

## Local Private Inventory

각 agent는 필요 시 아래 파일을 local-only로 유지한다.

```text
servers/local/connection-inventory.private.md
servers/local/ssh_config
servers/local/rsync-targets.tsv
servers/local/gpu-caps.tsv
servers/local/method-runtime.env
servers/local/dataset-roots.tsv
```

`rsync-targets.tsv` 형식:

```text
server<TAB>ssh_alias<TAB>repo_path
```

## SSH Alias Convention

서버 간 artifact broadcast는 tracked raw IP/user/port가 아니라 local-only SSH
alias를 사용한다. tracked 문서와 메시지에는 repository server name, alias,
repo path, relative `local/` path만 기록한다.

일반 project artifact는 source server가
`scripts/rsync-artifact-broadcast.sh`로 broadcast한다. 단, server1/server2 모두 SH
onboarding과 peer rsync verification이 끝나지 않았으므로 artifact broadcast는 금지한다. 각
server에는 local-only `rsync-targets.tsv`의 해당 repo path에 clone을 만든 뒤,
onboarding audit와 active 등록을 마쳐야 한다. `--delete`, private inventory,
SSH material, credential, 민감 경로, repo-external 경로는 별도 user approval
없이는 전송하지 않는다.

## Codex Session Boundary

각 서버 record에는 해당 repo를 담당하는 primary Codex session ID,
required/confirmed `Sol Ultra` profile, CWD를 역할별로 기록한다.
Git/SSH/rsync/Slurm command는 `repository identity + session ID + confirmed
Sol Ultra profile + CWD`가 모두 일치할 때만 수행한다. Delegated subagent는
별도 `Terra Ultra` runtime metadata를 확인하되 server-session authority를
대체할 수 없다. 다른 repo session, 특히 `knowledge-revision` session은 이
repo의 command·message·artifact target으로 사용할 수 없다.

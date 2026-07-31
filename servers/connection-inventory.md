# 서버 접속 인벤토리

- 갱신 시각: 2026-07-30
- 작성 agent: head-server1-gh (global-head)
- 목적: agent 간 SSH/rsync 계획 수립을 위한 redacted 접속 인벤토리 공유

이 tracked 파일에는 raw SSH HostName/IP, username, port를 기록하지 않는다.
실제 접속값은 각 서버 clone의 ignored local-only 파일에 보관한다.
비밀번호, SSH private key, token, key path, passphrase는 Git에 기록하지 않는다.

## 서버 목록

server1에는 GH clone만 초기화되어 있고, server-head는 아직 배정되지 않았다.
따라서 server1은 `pending-onboarding`이며 Slurm 제출, 원격 preflight,
artifact broadcast 또는 실행 inbox를 발행하지 않는다. server4는 physical host로
등록됐지만 이 repo clone/SH가 없는 `registered-pending-clone` 상태다. server2–3는
future target이다.

| Repository server name | Raw connection detail location | 상태/용도 |
| --- | --- | --- |
| `server1` | `servers/local/ssh_config`, `servers/local/rsync-targets.tsv`, `servers/local/gpu-caps.tsv` | pending-onboarding / GH clone: `/mnt/raid5/janghj/ODE-edit` / session ID는 `servers/active/server1.md` |
| `server2` | `servers/local/rsync-targets.tsv`, `servers/local/gpu-caps.tsv` | future target / clone 전 / Codex session 미지정 |
| `server3` | `servers/local/ssh_config`, `servers/local/rsync-targets.tsv`, `servers/local/gpu-caps.tsv` | future target / clone 전 / Codex session 미지정 |
| `server4` | `servers/local/rsync-targets.tsv`, `servers/local/gpu-caps.tsv` | registered-pending-clone / GPU cap 3, memory cap 65984 MiB per GPU / Codex session 미지정 |

## Codex Session Registry

아래 registry는 **이 repository 전용** session만 기록한다. 각 서버의
global-head와 server-head는 서로 다른 role/session으로 명시하며, `미지정`은
해당 role의 Codex session을 아직 만들거나 배정하지 않았다는 뜻이다. 다른 repo의
session ID를 채우거나 대체 대상으로 사용하지 않는다.

| 서버 | 역할 | Codex session ID | Required/confirmed model | Repository CWD | 상태 |
| --- | --- | --- | --- | --- | --- |
| `server1` | global-head | `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2` | `Terra Ultra` / `Terra Ultra` (`gpt-5.6-terra`, runtime metadata) | `/mnt/raid5/janghj/ODE-edit` | active / 현재 GH session |
| `server1` | server-head | 미지정 | `Terra Ultra` / 미지정 | `/mnt/raid5/janghj/ODE-edit` | SH 미배정; GH와 별도 session 필요 |
| `server2` | server-head | 미지정 | `Terra Ultra` / 미지정 | `/mnt/raid5/janghj/ODE-edit` | future target / clone 전 |
| `server3` | server-head | 미지정 | `Terra Ultra` / 미지정 | `/data/janghj/ODE-edit` | future target / clone 전 |
| `server4` | server-head | 미지정 | `Terra Ultra` / 미지정 | `/data/janghj/ODE-edit` | registered-pending-clone |

새 SH를 등록할 때 GH는 이 table, `servers/active/<server>.md`, 그리고 해당
clone의 ignored `servers/local/session-boundary.env`에 **동일한** session ID,
confirmed `Terra Ultra` model profile, CWD, repository identity를 기록한다.
실제 instruction은 그 ID와 model profile을 envelope에 넣고
`scripts/check-session-boundary.sh <session-id>`를 먼저 실행한다.

서버를 등록할 때는 `servers/templates/server-onboarding.md`를 바탕으로
`servers/active/<server>.md`를 만들고, 해당 server-head의 heartbeat와
red-team onboarding audit이 `pass` 또는 명시적 `waived`가 된 뒤에만 task를
배정한다.

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
`scripts/rsync-artifact-broadcast.sh`로 broadcast한다. 단, 현재 server1만
등록 중이고 peer clone이 없으므로 artifact broadcast는 금지한다. 각 future
server에는 local-only `rsync-targets.tsv`의 해당 repo path에 clone을 만든 뒤,
onboarding audit와 active 등록을 마쳐야 한다. `--delete`, private inventory,
SSH material, credential, 민감 경로, repo-external 경로는 별도 user approval
없이는 전송하지 않는다.

## Codex Session Boundary

각 서버 record에는 해당 repo를 담당하는 Codex session ID, required/confirmed
Codex model profile, CWD를 역할별로 기록한다. Git/SSH/rsync/Slurm command는
`repository identity + session ID + confirmed Terra Ultra profile + CWD`가
모두 일치할 때만 수행한다. 다른 repo session, 특히 `knowledge-revision`
session은 이 repo의 command·message·artifact target으로 사용할 수 없다.

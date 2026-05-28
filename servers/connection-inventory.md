# 서버 접속 인벤토리

- 갱신 시각:
- 작성 agent:
- 목적: agent 간 SSH/rsync 계획 수립을 위한 redacted 접속 인벤토리 공유

이 tracked 파일에는 raw SSH HostName/IP, username, port를 기록하지 않는다.
실제 접속값은 각 서버 clone의 ignored local-only 파일에 보관한다.
비밀번호, SSH private key, token, key path, passphrase는 Git에 기록하지 않는다.

## 서버 목록

| Repository server name | Raw connection detail location | 상태/용도 |
| --- | --- | --- |
| `<server>` | `servers/local/connection-inventory.private.md` | active/pending/retired |

## Local Private Inventory

각 agent는 필요 시 아래 파일을 local-only로 유지한다.

```text
servers/local/connection-inventory.private.md
servers/local/ssh_config
servers/local/rsync-targets.tsv
```

`rsync-targets.tsv` 형식:

```text
server<TAB>ssh_alias<TAB>repo_path
```

예:

```text
server1	rke-server1	/mnt/shared/project-repo
```

## SSH Alias Convention

서버 간 자동 artifact fan-out은 tracked raw IP/user/port가 아니라 local-only
SSH alias를 사용한다.

Tracked 문서와 메시지에는 alias 이름, repository server name, repo path,
relative `local/` path만 기록한다. raw HostName/IP, port, key path, credential은
Git에 기록하지 않는다.

## rsync 사용 메모

- Git-excluded project artifacts는 Git이 아니라 rsync로 공유한다.
- 일반 실험 artifact는 per-transfer 사용자 승인 없이 source 서버가 active 서버들로 자동 fan-out할 수 있다.
- `--delete`, repo-external path, 민감 파일, private inventory, SSH material, credentials는 자동 전송하지 않고 사용자/global-head 승인을 요구한다.
- 전송 후 Git에는 raw artifact가 아니라 compact message/status/verification만 기록한다.

# 2026-08-22 GH/SH 새 세션 registry reconciliation

## 목적

새 GH, SH1, SH2 session을 app host metadata, direct inbox ACK, local hard
boundary checker와 repository/runtime read-only audit로 확인하고 current authority
record를 갱신한다. 실행 중인 실험과 과거 provenance는 변경하지 않는다.

## 확인된 현재 session

| 역할 | server/host | session ID | GH direct inbox |
| --- | --- | --- | --- |
| GH | server1 / `remote-ssh-codex-managed:lab120` | `01a0278e-3f20-7b12-8c26-cf71deb2708b` | active caller |
| SH1 | server1 / `remote-ssh-codex-managed:lab120` | `01a0278d-456b-7d20-820f-63fc80a57b8d` | ACK PASS |
| SH2 | server2 / `remote-ssh-codex-managed:lab121` | `01a0278d-6e24-7a81-974a-96e4addd7ca5` | ACK PASS |

## 교차 inbox 판정

서로 다른 nonce로 아래 네 경계를 확인했다.

- SH1→SH2 ping: PASS
- SH2→SH1 ACK: PASS
- SH2→SH1 ping: PASS
- SH1→SH2 ACK: PASS

따라서 GH↔각 SH와 SH1↔SH2의 direct inbox/ACK 연결이 모두 정상이다.

## Boundary 및 repository snapshot

- 세 session 모두 ignored local boundary를 새 session ID로 갱신하고
  `scripts/check-session-boundary.sh <session-id>` PASS
- common `PROTOCOL.md`: SHA256
  `a54a4e7c00c36ec9f3b0fe122e5d8735dacc2c21c8604bc598593eb396936663`,
  51,147 bytes, 1,125 lines
- audit base `origin/main`:
  `0d63ad4ec4978be6d04aabb640e17917bd1348d7`, tree
  `652b84489825371558002e24b381dcb777327a7f`
- GH root dirty/detached state와 SH1 detached/untracked state는 그대로 보존
- SH2는 audit 시 clean main이며 registry push 뒤 ff-only 최신화 대상

## 실행 상태 보존

- server1 `22541`: 일부 completed, 0/4 running 상태를 read-only 확인
- server2 `22600/22601`: pending 상태를 read-only 확인
- submit, cancel, retry, model/GPU/result mutation과 artifact transfer: 0

## 변경 범위

현재 control-plane 파일만 갱신한다.

- `servers/connection-inventory.md`
- `servers/active/server1.md`
- `servers/active/server2.md`
- 이 reconciliation 메시지

과거 plans, reports, audits, receipts, result manifests와 completed launcher의
old session ID는 당시 실행 provenance로 보존한다.

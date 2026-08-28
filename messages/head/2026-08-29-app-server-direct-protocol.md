# 2026-08-29 app-server direct coordination 전환

- 작성: `head-server1-gh`
- 범위: GH·SH1·SH2·SH4 live Codex control transport
- science/source/model/GPU/Slurm/result action: 0

## 결정

GH·SH 모든 등록 세션의 live instruction, response, 경로 요청, 상태 질의,
handoff와 완료 보고는 `send_message_to_thread` 또는 Git inbox가 아니라 Codex
app-server direct peer-to-peer request-response로 처리한다.

각 연결은 target host의 local Unix control socket에서 WebSocket HTTP Upgrade,
`initialize`/`initialized`, `thread/resume`, idle `turn/start` 또는 active
`turn/steer`, `turn/completed` 순서를 사용한다. GH가 exact session과 turn을
검증하고 같은 stream에서 response를 회수한다.

SH→GH와 SH↔SH도 금지하지 않으며 정식 direct 통신이다. 긴 결과는 tracked
report에 쓰고 direct final에는 report path와 compact identity만 남긴다.
App-server 단계가 실패하면 `COMMUNICATION_HOLD`로 기록하며 user 재승인 없이
Git inbox, rsync, base64 또는 commit을 message fallback으로 사용하지 않는다.

SH1→GH 역방향 delivery는 SH1이 GH active turn
`01a049b4-3dc1-7e33-b3e6-ac07423ff10c`에 exact `expectedTurnId`로
`turn/steer`하여 nonce
`ODEEDIT-SH1-TO-GH-REVERSE-20260829-R1`를 전달했고 PASS했다.

## Current sessions

| 역할 | session | host | direct 결과 |
| --- | --- | --- | --- |
| GH | `01a04939-8873-7673-8dca-4c7fc5e31af0` | `lab120` | controller active |
| SH1 | `01a04939-f93a-7b50-bca0-65438eab2062` | `lab120` | request-response ACK PASS |
| SH2 | `01a0493a-074c-7f91-9a13-769116326fef` | `lab121` | request-response ACK PASS |
| SH4 | `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` | `lab163` | request-response ACK PASS |

SH1 steer는 진행 중인 science task를 중단하거나 변경하지 않았고 같은
turn의 terminal ACK를 회수했다. 세 SH session 모두 protocol commit
`6d9e4e625c7ed016742ca3516299eae40d9b4af1`로 direct ff-only sync까지
완료했다. 전환/sync turn IDs와 host-local socket mapping은
`servers/connection-inventory.md`에 기록했다.

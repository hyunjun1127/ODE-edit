# 2026-08-29 GH/SH session registry rotation

- 작성: `head-server1-gh`
- 범위: GH·SH1·SH2·SH4 current session authority와 direct inbox 점검
- 과학 실험, source 구현, model/GPU/Slurm/result action: 0

## 새 authority

| 역할 | server | Codex session ID | deeplink | app registry |
| --- | --- | --- | --- | --- |
| GH | server1 | `01a04939-8873-7673-8dca-4c7fc5e31af0` | `codex://threads/01a04939-8873-7673-8dca-4c7fc5e31af0` | active, host `lab120` |
| SH1 | server1 | `01a04939-f93a-7b50-bca0-65438eab2062` | `codex://threads/01a04939-f93a-7b50-bca0-65438eab2062` | idle, host `lab120` |
| SH2 | server2 | `01a0493a-074c-7f91-9a13-769116326fef` | `codex://threads/01a0493a-074c-7f91-9a13-769116326fef` | idle, host `lab121` |
| SH4 | server4 | `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` | `codex://threads/01a04939-b5c7-7a03-ba2d-ef3343d62cfd` | idle, host `lab163` |

사용자가 위 네 ID를 새 canonical sessions로 지정했다. 이전 current-authority
session은 superseded로 남기고 과거 task/report/audit/receipt의 session ID는 실행
provenance이므로 변경하지 않는다.

## 통신 점검

GH의 `list_threads`와 각 SH에 대한 `read_thread`는 모두 PASS했다. 즉 세 target
session과 host routing은 app registry에서 조회 가능하다.

GH는 nonce `ODEEDIT-SESSION-MESH-20260829-001`로 SH1·SH2·SH4에 각각
`send_message_to_thread`를 호출했다. 세 호출 모두 전송 전에 다음 동일 오류로
종료됐다.

```text
This app tool is no longer available through dynamic tools. Use the codex_app MCP server.
If that server is unavailable on this host, task delegation is unavailable.
```

따라서 GH→SH1/SH2/SH4 메시지는 0/3 전달됐고 ACK는 0/3이다. SH 사이 mesh
instruction도 전달되지 않아 SH1↔SH2, SH1↔SH4, SH2↔SH4는 모두 NOT RUN이다.
이는 SSH 공개키나 thread ID 불일치로 분류하지 않는다. `codex_app` registry read는
정상이고 outbound send capability만 현재 GH task에 노출되지 않은 상태다.

Git inbox, rsync, SSH 또는 다른 fallback 전송은 사용하지 않았다.

## 갱신 파일

- `servers/connection-inventory.md`
- `servers/active/server1.md`
- `servers/active/server2.md`
- `servers/active/server4.md`
- `messages/README.md`

direct `codex_app` MCP send가 노출되면 동일 네 session으로 GH↔SH 및 SH↔SH mesh를
다시 검사한다.

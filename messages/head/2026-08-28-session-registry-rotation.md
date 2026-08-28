# 2026-08-28 GH/SH session registry rotation

- 작성: `head-server1-gh`
- 범위: tracked session authority와 official inbox 연결 상태
- 과학 실험, source 구현, model/GPU/Slurm/result action: 0

## 새 authority

| 역할 | server | Codex session ID | deeplink |
| --- | --- | --- | --- |
| GH | server1 | `01a04660-f2e5-7583-a9f9-83db20210d72` | `codex://threads/01a04660-f2e5-7583-a9f9-83db20210d72` |
| SH1 | server1 | `01a04664-c753-7461-b914-b4ebf24a581a` | `codex://threads/01a04664-c753-7461-b914-b4ebf24a581a` |
| SH2 | server2 | `01a04661-2f02-7d93-b3ba-4c69d113eaee` | `codex://threads/01a04661-2f02-7d93-b3ba-4c69d113eaee` |

사용자가 위 ID를 새 canonical session으로 지정했다. 2026-08-22 registry의
GH/SH1/SH2 session은 inactive/superseded current authority로 분류한다. 과거
task/report/audit/receipt의 session ID는 provenance이므로 변경하지 않는다.

## Official inbox 점검

GH는 새 SH1과 SH2에 각각 official `send_message_to_thread`를 호출했다. 두 호출
모두 전송 전에 다음 동일 오류로 종료됐다.

```text
This app tool is no longer available through dynamic tools. Use the codex_app MCP server.
If that server is unavailable on this host, task delegation is unavailable.
```

따라서 GH→SH1, GH→SH2 메시지는 전달되지 않았고 ACK도 없다. SH1↔SH2
cross-inbox instruction 역시 전달할 수 없어 `NOT RUN`이다. thread ID 불일치나
SSH 인증 실패로 판정하지 않으며, 현재 상태는 `BLOCKED_CODEX_APP_MCP_UNAVAILABLE`로
기록한다. fallback 전송은 사용하지 않았다.

## 갱신 파일

- `servers/connection-inventory.md`
- `servers/active/server1.md`
- `servers/active/server2.md`
- `messages/README.md`

Runtime model/CWD/session checker는 각 새 task에서 별도 재검증해야 한다.

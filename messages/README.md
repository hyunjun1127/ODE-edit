# 메시지 통합 인덱스

- 갱신 시각: 2026-07-30
- 작성 agent: head-server1-gh
- 역할: global-head

서버별 메시지 폴더는 쓰기 충돌을 줄이기 위한 원본 보관 위치다. 전체
흐름을 읽을 때는 이 인덱스를 먼저 보고, 세부 내용은 원문 링크를 따라간다.
모든 active server-head는 자기 서버와 직접 관련 없는 메시지도 이 인덱스에서
확인하고, `messages/acks/<server>/`에 확인 기록을 남긴다.

`messages/inbox/<server>.md`는 global-head가 특정 server-head에게 남기는
durable instruction inbox다. 대상 server-head는 이 파일을 주기적으로 읽고,
실행 상태와 결과는 `messages/server-heads/<server>/`, `messages/acks/<server>/`,
`runs/`, `experiment-reports/`, `audits/`에 남긴다. Inbox 파일 자체를
server-head가 수정하지 않는다.

주의: Git message나 inbox entry는 자동 실행기가 아니다. 실제 실행은 살아있는
agent session, systemd/cron automation, SSH, Slurm job이 읽고 수행할 때만
발생한다.

## 최신 메시지

| 작성 시각 | 작성 agent | 작성 서버 | 범위/수신 | 상태 | 요약 | 다음 행동 | 원문 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-28 | head-server1-gh | server1 | 전체 / GH·SH1·SH2 | blocked | 새 session authority 등록; official inbox는 `codex_app` MCP unavailable로 미전달 | Desktop/host tool 연결 복구 후 양방향 inbox 재검증 | [session registry rotation](head/2026-08-28-session-registry-rotation.md) |
| 2026-07-30 | head-server1-gh | server1 | 전체 / SH 미배정 | info | ODE-Edit Session 01 Motivation Validation 초기화; repo push 완료, SH가 없어 실행 보류 | user가 server/SH·compute context 확인 | [초기화 공지](head/2026-07-30-gh-initialization.md) |

## 갱신 규칙

- server-head는 `messages/server-heads/<server>/` 아래에 원문 메시지를 쓴다.
- global-head는 서버별 직접 지시가 필요할 때 `messages/inbox/<server>.md`에
  append한다.
- global-head는 원격 변경을 동기화한 뒤 중요한 새 메시지를 이 인덱스에 시간순으로 추가한다.
- 작성 시각은 메시지 본문에 명시된 시각을 우선하고, 없으면 해당 파일을 추가한 Git commit 시각을 사용한다.
- 이 파일은 읽기 편의를 위한 요약이다. 최종 근거는 원문 메시지와 관련 `servers/`, `agents/`, `audits/`, `experiment-reports/` 기록이다.

## 확인 상태

| 서버 | 확인 시각 | 확인 agent | 확인한 최신 메시지 | 상태 | 확인 기록 |
| --- | --- | --- | --- | --- | --- |

각 서버는 최신 index를 읽은 뒤 자기 서버의 ack 파일만 작성하거나 갱신한다.

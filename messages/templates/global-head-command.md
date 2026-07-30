# Global-head 명령 템플릿

## 명령 메타데이터

- 명령 ID:
- 작성 시각:
- 작성 agent:
- 대상 서버:
- 대상 Codex session ID:
- 대상 repository CWD:
- 대상 Git repository identity:
- 우선순위: low/normal/high/blocking
- ack 필요 여부: yes/no
- 완료 보고 경로:

## 실행 권한 envelope

- 허용 write path:
- Slurm 제출 권한: allowed/not allowed/requires global-head or user confirmation
- GPU cap:
- host-memory cap / 요청 memory:
- red-team gate:
- artifact broadcast 의무:
- 완료 보고 경로:
- session boundary 확인 command: `scripts/check-session-boundary.sh <session_id>`

## 명령

구체적인 수행 내용을 적는다.

## 근거

왜 이 명령이 필요한지, 어떤 메시지/태스크/보고서에 근거하는지 적는다.

## 수락 기준

- 확인해야 할 명령:
- 갱신해야 할 파일:
- 보고해야 할 메시지:

## 주의 사항

- credentials, token, private key는 기록하지 않는다.
- 실험 완료 후 Git-excluded artifact 공유는 자동 rsync broadcast 정책을 따른다.
- `--delete`, credentials, `local/secrets/`, private inventory, SSH material 전송은
  사용자 명시 승인 없이는 실행하지 않는다.
- actionable GH 명령에 실행 권한 envelope가 없으면 SH는 추측 실행하지 말고
  ack/blocker로 clarification을 요청한다.
- 대상 session ID, repository CWD, Git identity 중 하나라도 불일치하면 다른
  repo session을 조작하지 말고 `block`으로 보고한다.

# Global-head 명령 템플릿

## 명령 메타데이터

- 명령 ID:
- 작성 시각:
- 작성 agent:
- 대상 서버:
- 우선순위: low/normal/high/blocking
- ack 필요 여부: yes/no
- 완료 보고 경로:

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

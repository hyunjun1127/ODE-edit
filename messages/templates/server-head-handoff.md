# 서버 Head Handoff 템플릿

## 헤더

- 시간:
- From:
- To:
- 상태: info/request/blocked/handoff/done
- 우선순위:
- 관련 plan:
- 관련 task:
- 관련 run:

## 요약

대상 server-head가 private local log를 읽지 않아도 행동할 수 있도록,
cross-server 이슈, 요청, handoff 내용을 충분히 자세히 적는다.

## 확인한 맥락

- 확인한 서버:
- 로그 경로:
- 로그 확인 범위:
- artifact 경로:
- 수행한 command 또는 check:

## 근거

요청을 뒷받침하는 key metric, 짧은 error excerpt, file size, checksum,
구체적인 관찰 내용을 적는다. 전체 로그는 붙이지 않는다.

## 요청 행동

대상 agent 또는 global-head에게 요청하는 정확한 행동을 적는다.

파일 전송이 필요한 경우 다음을 포함한다.

- Source server:
- Source path:
- Destination server:
- 요청 또는 제안 destination path:
- overwrite 정책:
- 전송 후 검증 방법:

## Blocker 또는 Risk

누락된 경로, 권한 문제, 알 수 없는 환경 정보, version mismatch, resource
constraint를 적는다.

## 다음 담당자

응답하거나 이어서 처리해야 할 agent 또는 role을 적는다.

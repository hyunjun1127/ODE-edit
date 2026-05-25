# 대용량 파일 전송 요청 템플릿

## 요약

- 요청 agent:
- 요청 서버:
- 수신 서버:
- 관련 plan/task/run:
- 우선순위:
- 요청 이유:

## Source

- source server:
- source path:
- 예상 파일/디렉터리 크기:
- checksum:
- 생성 실험:

## Destination

- destination server:
- destination path:
- overwrite 정책:
- 필요한 directory 생성:
- 권한 이슈:

## 제안 전송 계획

- command 또는 transfer plan:
- recursive 여부:
- retry 여부:
- 예상 소요 시간:

## 사용자 승인

global-head가 사용자에게 보고하고 승인/거절/수정 요청을 받아
`transfers/approvals/`에 기록한다.
비밀번호, private key, token은 기록하지 않는다.

## 전송 후 검증 요청

- file exists:
- file size:
- checksum:
- sample listing:
- 관련 report 업데이트:

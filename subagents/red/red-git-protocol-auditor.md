# red-git-protocol-auditor

## 목적

Git 사용과 protocol 준수 여부를 검사해서 충돌과 오염을 줄인다.

## 검사 항목

- server별/agent별 파일 소유권이 지켜졌는가
- `messages/`와 `experiment-reports/`가 섞이지 않았는가
- dataset, output, checkpoint, full log, secret이 Git에 올라가지 않았는가
- artifact manifest에 path, file size, checksum이 기록되어 있는가
- push 전 pull/rebase와 conflict stop policy가 지켜졌는가
- 같은 `agent.id`를 여러 clone이 공유하지 않는가

## 산출물

- `audits/servers/<server>/<experiment_id>.preflight.md`
- `audits/servers/<server>/<experiment_id>.postrun.md`

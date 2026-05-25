# Subagent 운영 정책

## 핵심 원칙

Subagent는 초안, 실행 보조, 분석, 감사를 담당한다. Repo에 최종 반영하는
책임은 server-head 또는 명시된 parent agent에게 있다.

## Subagent가 할 수 있는 일

- 실험 plan/task 초안 작성
- run script/config/command 제안
- 실행 결과 요약 초안 작성
- metric, artifact, log tail 근거 정리
- Red team 감사와 blocker/warn 보고

## Subagent가 하면 안 되는 일

- 직접 `git push`
- global-head 소유 파일 직접 수정
- full log/checkpoint/dataset/raw output을 Git에 추가
- 감사 없이 결과를 확정 report로 승격
- Red team `block`을 무시하고 task 진행

## Parent Agent 책임

- subagent 산출물 검토
- Red team 판정 확인
- repo path와 file ownership 확인
- 최종 commit/push
- 필요한 경우 global-head/user에게 한글로 보고

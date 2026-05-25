# 충돌 보고 템플릿

## 요약

어떤 repository/clone에서 충돌이 났는지, 어떤 작업이 실패했는지, 자동
sync가 중단됐는지 적는다.

## 맥락

- 시간:
- 보고 agent:
- 서버:
- repository:
- branch:
- remote:
- 관련 plan/task/run:
- local conflict report 경로:

## 근거

핵심 `git status --short --branch` 출력, 실패한 command, 충돌 path, 관련
commit SHA를 적는다. private local scratch log는 붙이지 않는다.

## 영향

어떤 서버 또는 task가 pull/rebase/push를 멈춰야 하는지, global pause
marker `control/sync-paused`가 올라갔는지 적는다.

## 필요한 결정

global-head/user에게 필요한 결정을 구체적으로 요청한다. 예: 한쪽 변경
선택, 파일 수동 merge, 특정 서버 reclone, clean remote state로 복구.

## 다음 담당자

충돌을 해결하거나 이어서 처리할 agent 또는 role을 적는다.

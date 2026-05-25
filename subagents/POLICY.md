# Subagent 운영 정책

## 핵심 원칙

Subagent는 초안, 실행 보조, 분석, 감사를 담당한다. Repo에 최종 반영하는
책임은 server-head 또는 명시된 parent agent에게 있다.

Parent chain은 다음과 같다.

```text
global-head -> server-head -> worker/subagent
```

Subagent의 parent는 해당 subagent를 호출한 `server-head` 또는 `worker`다.
Parent가 subagent 결과를 검토하고, 필요한 경우 red-team 판정을 확인한 뒤
repo에 반영한다.

## Subagent가 할 수 있는 일

- 실험 plan/task 초안 작성
- run script/config/command 제안
- 실행 결과 요약 초안 작성
- metric, artifact, log tail 근거 정리
- Red team 감사와 blocker/warn 보고
- parent에게 transfer request, server lifecycle, conflict report 초안 제안

## Subagent가 하면 안 되는 일

- 직접 `git push`
- `scripts/claim-task.sh`, `scripts/finish-task.sh`, `scripts/sync-agent.sh`,
  `scripts/heartbeat.sh` 직접 실행
- global-head 소유 파일 직접 수정
- `messages/head/`, `transfers/approvals/`, `control/` 직접 수정
- 다른 서버 또는 다른 agent 소유 파일 직접 수정
- full log/checkpoint/dataset/raw output을 Git에 추가
- 감사 없이 결과를 확정 report로 승격
- Red team `block`을 무시하고 task 진행
- Red team이 자기 blocker를 직접 waiver 처리

## Parent Agent 책임

- subagent 산출물 검토
- Red team 판정 확인
- repo path와 file ownership 확인
- 최종 commit/push
- 필요한 경우 global-head/user에게 한글로 보고

## Parent별 접근 제한

- `server-head` parent: subagent는 `plans/updates/<server>/`,
  `tasks/proposed/<server>/`, `messages/server-heads/<server>/`,
  `experiment-reports/servers/<server>/`, `audits/servers/<server>/`,
  `transfers/requests/`에 들어갈 초안을 만들 수 있다. Parent 검토 전에는
  최종 파일로 확정하지 않는다.
- `worker` parent: subagent는 worker가 claim한 task와 관련된 `runs/`,
  log tail, artifact manifest, 결과 report/audit 초안을 만들 수 있다.
  다른 task나 다른 worker의 run 파일은 다루지 않는다.
- Red-team subagent: 문제를 발견하면 직접 수정하지 않고 audit 또는 parent
  보고로 남긴다. 수정은 blue-team 또는 parent가 수행하고, red-team은 재검사한다.

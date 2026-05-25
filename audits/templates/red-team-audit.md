# Red Team 감사 템플릿

## 요약

- 감사 유형: pre-flight/post-run/pre-push-sensitive
- 판정: pass/block/waived
- 관련 plan:
- 관련 task:
- 관련 run:
- 감사 서버:
- 감사 agent:
- 감사 시간:

## 검사 대상

- task 또는 report 경로:
- run script:
- config:
- dataset 경로:
- output/artifact 경로:
- log 경로:

## Data/Evaluation 감사

- train/validation/test 분리:
- leakage 가능성:
- metric 계산 조건:
- 평가 절차 문제:

## Logic/Evidence 감사

- 실험 가정:
- baseline/comparison:
- 해석의 근거:
- hallucination 또는 근거 부족 가능성:

## Git/Protocol 감사

- file ownership:
- messages/report 분리:
- local-managed 파일 유입 여부:
- secret/checkpoint/full-log 유입 여부:
- sync/conflict protocol:

## Blocker

`block` 판정이면 중단해야 하는 이유와 필요한 수정 사항을 구체적으로 적는다.

## 권고

다음 행동과 담당 agent 또는 role을 적는다.

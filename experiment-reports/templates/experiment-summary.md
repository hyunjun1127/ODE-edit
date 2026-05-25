# 실험 결과 정리 템플릿

## 요약

실험의 목적, 가장 중요한 결과, 결론을 짧게 정리한다.

## 식별 정보

- 실험 ID:
- 관련 plan:
- 관련 task:
- 관련 run:
- 실행 서버:
- 실행 agent:
- 실행 시간:

## 실행 설정

- command:
- run script:
- working directory:
- environment:
- GPU/CPU/memory:
- 주요 config:
- git commit:
- seed:

## 데이터와 산출물 경로

- dataset 경로:
- raw output 경로:
- checkpoint/model 경로:
- log 경로:
- artifact manifest:
- file size/checksum:

데이터셋, raw output, checkpoint, full log는 Git에 넣지 않고 경로만 적는다.

## 재현성 정보

- 재실행 command:
- 필요한 environment/module/conda:
- 주요 dependency version:
- random seed:
- 재실행 시 주의할 점:

## Config Diff

이전 실험 또는 baseline 대비 바뀐 config, hyperparameter, dataset, prompt,
method 차이를 정리한다.

## 주요 결과

핵심 metric과 관찰 결과를 표나 bullet로 정리한다.

## Baseline/비교표

비교 대상, metric, 차이, 해석을 표로 정리한다.

## 해석

결과가 기대와 어떻게 다른지, 어떤 원인이 가능한지, 다음 실험에 어떤
의미가 있는지 적는다.

## 문제점과 주의사항

실패, warning, 재현성 이슈, 환경 차이, 데이터 문제를 적는다.

## 실패 분류

실패한 경우 CUDA OOM, import error, data path error, logic error, Git/protocol
error, unknown 중 하나 이상으로 분류하고 근거를 적는다.

## 다음 행동

다음에 수행할 실험, 수정할 config/script, 필요한 server-head/global-head
결정을 적는다.

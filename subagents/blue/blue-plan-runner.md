# blue-plan-runner

## 목적

Plan과 task proposal을 실행 가능한 실험 단위로 정리한다.

## 책임

- plan의 목표, 가정, 입력/출력, resource 요구사항을 확인한다.
- 필요한 `project/run_scripts/`, config, command를 명확히 정리한다.
- 이전 실험 또는 baseline 대비 config diff를 정리한다.
- dataset/output/checkpoint/full log가 Git에 들어가지 않도록 경로만 남긴다.
- 실행 전 red-team pre-flight audit에 필요한 정보를 제공한다.

## 산출물

- `plans/updates/<server>/`
- `tasks/proposed/<server>/`
- 필요한 경우 `project/run_scripts/`

# blue-experiment-runner

## 목적

승인된 task를 실행하고, 실행 상태와 artifact 경로를 기계가 읽을 수 있게
정리한다.

## 책임

- `tasks/pending/`에서 승인된 task만 실행한다.
- 실행 command, environment, resource, exit code를 기록한다.
- full log는 local에 두고, Git에는 short log tail과 경로만 남긴다.
- 결과 확정 전 red-team post-run audit에 필요한 정보를 제공한다.

## 산출물

- `runs/<run_id>/status.<agent>.json`
- `runs/<run_id>/metrics.<agent>.json`
- `runs/<run_id>/artifact_paths.<agent>.json`
- `runs/<run_id>/log_tail.<agent>.txt`

# CPU 분석 r1 경로 처리 오류 및 한정 수리

- MAIN job48303은 COMPLETED/0:0, 11,191 allocated GPU-sec이다. 본 오류는 그 뒤 CPU 분석에서 발생했으며 과학 실행 실패가 아니다.
- 실패 분석 source: `f61207dc` (실행 source `5d149fec254a7a53b6d91790d186880f248676e6`는 변경하지 않았다).
- `initial_cpu_state`가 HF snapshot의 정상적인 `model.safetensors.index.json` symlink를 일반 결과 파일의 non-symlink 규칙으로 거부했다. 원 모델 파일의 부재/변경이 아니며 새 GPU 호출은 0이다.
- 실패 local orchestrator `authoritative/postrun-r1.py` SHA256: `4370324d98671d7db232692cf4ce51eb53ebaa9f27fbed7725de628f39e7a018`. 분석 집계/보고 root 생성 전 중단됐다. 이미 완료한 scheduler/source 검산 receipt는 그대로 보존한다.
- 수리 범위: CPU reader에서 input.lock의 exact path/realpath/symlink-kind/size를 결속한 model member만 resolve한다. 작은 index는 full SHA를 재검산하고 selected W0 tensor SHA도 검산한다. 다른 blob으로 재지정된 symlink는 bytes가 같아도 거부한다. 일반 결과 파일의 symlink 거부 규칙은 유지한다.
- 원 source/runtime/config/precision/tolerance/model/checkpoint/raw/evaluator 및 과학 수치는 변경하지 않는다. HF cache를 수정하거나 큰 pretrained shard를 불필요하게 재해시하지 않았다.
- 추가 fixture 및 전체 CPU 검사: **121 PASS, 13.562s**. 실제 GPU 재실행 0. 별도 CPU attempt `authoritative/postrun-r2.py`, private output `raw/analysis-r2`로 재개한다.
- Allocation receipt SHA256: `2bc477754f5c5eca0800b2b1a4c61ecdcc25eafabb3dfd988b3e31e973b80fa6`; postrun source verification SHA256: `b5f6a342bafb0cb56b82dfa2e724a96785bfe372bae6b176f051548d424d2cff`.

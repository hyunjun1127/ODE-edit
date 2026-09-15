# Single-layer write-coupled z-flow CPU reference

고정 native writer를 이용해 actual-write loss와 변경 비용을 함께 최적화하는 참조 구현이다. 실제 Llama adapter, tokenizer/key extractor, GPU 성능, durable checkpoint transaction은 포함하지 않는다.

- [전체 method 파이프라인](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-single-layer-zflow-pipeline-v1.md)
- [실행 설정](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-single-layer-zflow-contract-v1.json)
- [선행 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-16-single-layer-zflow-design-review-ko.md)

## 흐름

`W_entry/M_entry → K/B/S → all-token affine cache/entry teacher → X=0 → IMEX + fresh suffix L/g → 종료 → 실제 FP32 weight parity → history 1회 확정`

Main reference는 positive price=1, barrier off, 최대 25개 complete logical oracle sweep를 사용한다. Price는 미튜닝 초기값이다. 25회 내 최적 endpoint나 실제 편집 효능을 주장하지 않는다. `RESOURCE_STOP`도 마지막 accepted 후보를 반환하며 commit 결과와 별도로 기록한다.

## 파일

| 파일 | 역할 |
|---|---|
| flow_core.py | Native map, quadratic cost/metric, IMEX, barrier off/fixed/exponential, 상대 budget KKT, step 회복 |
| oracle.py | 전역 token weights, 고정 teacher reverse KL, all-token affine cache, complete X gradient |
| transaction.py | CPU FP32 weight/history materialization 및 in-memory exactly-once |
| demo.py | 비선형 causal toy suffix에 전체 경로 연결, JSON trace/receipt |
| tests/ | 수식·수치 종료·gradient·commit 회귀 검사 |

## CPU 실행

저장소 root에서 cached uv/PyTorch 환경을 사용한다.

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project --with torch python -m unittest discover -s project/run_scripts/single_layer_zflow/tests -v
PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project --with torch python -m project.run_scripts.single_layer_zflow.demo --output /tmp/single-layer-zflow-demo.json
```

이미 PyTorch가 설치된 환경에서는 `uv run --offline --no-project --with torch python` 부분을 해당 Python으로 바꾸면 된다. Demo는 CPU tensor만 만들며 checkpoint 다운로드나 원격 실행을 하지 않는다. CPU suffix callback은 full logits를 반환한다. 실모델에서의 selected-position head 최적화는 별도 adapter에 속한다.

## 해석 경계

- Solver status와 commit status는 별개다. 수렴은 reduced-space 1차 조건이며 edit 성공 인증이 아니다.
- Fresh oracle 1회는 전체 logical request/context/token sweep다. Scalar barrier 계산은 모델 평가 횟수에 포함되지 않는다.
- 작은 budget의 잘못된 boundary 판정을 상대 slack으로 수정했다. Raw 및 normalized residual을 함께 기록한다.
- Teacher/prefix 준비와 terminal parity 비용은 oracle 횟수 밖에 별도로 기록한다.
- Candidate/reject에서 W/M를 변경하지 않는다. 실제 commit cost는 저장될 weight와 entry 차이로 다시 계산한다.
- Transaction은 프로세스 내 exclusive ownership에서 동작한다. 디스크 crash recovery와 분산 실행의 exactly-once 보장은 별도 구현이 필요하다.

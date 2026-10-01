# JLZ fixed-budget B100×10

정본은 `plans/global/2026-10-01-jlz-sequential-bs100x10-v1/`이다. Pilot 원본은 수정하지 않는다.

- `policy.py`: pinned pilot solver를 호출하되 사용자 fixed-budget admission 적용. NONFINITE 이력은 fatal.
- `oracle.py`: 후보 weight 한 번 생성, dL/dW GPU 누적, FP64 adj pullback. full sequence / selected full-vocabulary head.
- `state.py`: W/H/RNG/context/ledger RAM transaction. 전체 checkpoint 저장 없음.
- `observation.py`: 기존 canonical tokenization/NLL kernel MB2 재사용, desired TF 및 true/new NLL.
- `run.py`: 최초 actual BS100 비교3회 + fresh W0/H0 10회. 최초 실패한 batch는 복구하고 종료.
- `collect.py`: afterany CPU 독립 counts/NLL/TF/분모/10commit·50append·9link 검산. agent 독립 reviewer의 사후검토는 사용자 recall 때.
- `bind.py`: 기존 자산 hash/schema 검증, create-once 입력/source/resource lock과 launcher.

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest project.run_scripts.jlz_pilot.test_prompts project.run_scripts.jlz_pilot.test_solver project.run_scripts.jlz_pilot.test_integration project.run_scripts.jlz_sequential.test_core -v
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.jlz_sequential.bind --attempt attempt-r1
```

bind는 Slurm 제출을 하지 않는다. Source commit 후 생성한 source hash lock으로 runner를 실행한다.
new-route token work는 oracle forward 입력 token만 계수한다. legacy/capture/official token work 전체는 NOT_SEPARATED이며 forward-call/time과 구분한다.
과학1200회는 고정 상한이지 ETA 또는 단일요청1200회 비용이 아니다. Post-run 독립 reviewer 미수행을 자동 collector PASS로 대체하지 않는다.

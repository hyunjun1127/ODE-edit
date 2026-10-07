# GPT2-XL W0 cohort reference

USER 승인 단일 cold W0 first2000 관측 profile이다. 기존 W0/profile/helper는 읽기만 한다.
새 관측 raw의 occurrence ordinal로 current100 ×20과 prefix500/1000/1500/2000을
CPU 재집계한다. 추가 LM forward, 편집, fit, solve, H, C0/P load, checkpoint는 없다.

`edits`는 reference cohort 비교축이다. actual/applied/pre/post edits는 항상0,
`reference_only=true`, `evaluation_model_state=W0`이다. 일반 method validator는
바꾸지 않고 task-local schema/worker에서 이 정직한 예외만 검사한다.
SDK 비동기 접수, 원격 identity, finish coverage/value readback은 별도 상태다.

재현 순서(전용 WT, CPU 준비 → source commit → freeze → launchers → held 검사/release):

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.base_model_eval.gpt2xl_server1_cohort tests --attempt /absolute/cpu-receipt.json
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.base_model_eval.gpt2xl_server1_cohort prepare --attempt /absolute/create-once-attempt
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.base_model_eval.gpt2xl_server1_cohort freeze --config /absolute/config.json --source EXACT_COMMITTED_SHA
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.base_model_eval.gpt2xl_server1_cohort launchers --config /absolute/config.json --lock /absolute/execution.lock.json
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.base_model_eval.gpt2xl_server1_cohort submit --config /absolute/config.json --lock /absolute/execution.lock.json
```

등록된 GPU runner가 fresh 관측 및 W&B 기록을, afterany CPU collector가 독립 raw
집계·coverage·partial/failure 보고를 수행한다. old W0 active일 때만 직렬화하며
기존 job/source/raw는 보존한다. 제출 뒤 agent 장기 monitoring/retry는 없다.

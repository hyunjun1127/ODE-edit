# 실험 W&B scalar 기록 공통 API

SH1 소유 공통 helper다. 기존 frozen 실험은 수정하지 않는다. 새 실험은 모델/GPU
작업 전에 cheap online startup 검증을 통과해야 한다. 인증 부재는
`SETUP_READY_NEEDS_USER_LOGIN`, 권한/연결/SDK 오류는 `LOGGING_BLOCKED` 계열이다.
offline 결과로 online 통과를 대체하지 않는다.

## SDK·인증 준비

각 서버 담당자는 별도 local venv에 `requirements-sdk.txt`를 설치하고
`servers/local/wandb.env`에 비민감 설정만 둔다. 과학 Python에는 SDK를 설치하지
않는다. worker는 별도 Python process이므로 torch/transformers/pydantic pin이
섞이지 않는다. 표준 CPython 3.10 이상, Linux 기준이다.

entity=`wkdguswns2256`, project=`layer allocation` 공백을 그대로 유지한다.
UI 주소는 API endpoint가 아니다. default는 SDK 공식 `https://api.wandb.ai`이며
기존 계정의 검증된 별도 endpoint만 담당자가 비민감 config에 명시한다.

키가 없으면 사용자가 해당 서버 terminal에서 SDK venv의 `wandb login`을
실행한다. **키를 명령 인자·채팅·보고서·Git에 넣지 않는다.** 표준 사용자 netrc는
같은 사용자 Slurm `--export=NONE` job에서도 HOME 경로로 읽는다. 허용된 기존
환경 인증/identity token 파일도 SDK가 직접 읽으며 helper는 값을 기록하지 않는다.
다른 서버의 credential을 복사하지 않는다.

## 새 launcher / runtime 통합

`--export=NONE` launcher에서도 아래 비민감 파일/SDK Python 절대경로를 고정한다.
공용 shell env를 source할 필요가 없다. job의 HF offline 설정을 유지한다.
legacy `WANDB_DISABLED`는 scientific parent에서 바꾸지 않고 새 sidecar가 online
설정을 명시한다. 기존 source를 일괄 수정하거나 이미 제출된 job에 붙이지 않는다.

```python
from project.run_scripts.experiment_tracking import init

tracking = init(
    env_file="/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env",
    spool="/absolute/ignored/task/attempt/tracking-new",  # create-once
    config={"server": "server1", "task_id": "my-task", "arm": "A", "attempt": "r1",
            "source_sha": "0" * 40, "config_sha": "0" * 64, "job_id": "12345"},
)
# init is before scientific/model work. Use actual SHA/job metadata, not these examples.
try:
    # Only already-computed Python scalars, every 10–30 s or candidate/batch/phase.
    tracking.log({"fit/loss": 0.25, "candidate": 1}, step=1)
    # No extra forward, tensor.item(), CPU tensor copy, or synchronization for logging.
except BaseException:
    tracking.finish(exit_code=1)  # bounded; preserves the scientific exception
    raise
else:
    tracking.finish()
```

`schema.METRICS`는 명시 허용 이름이다. task별 추가 지표는 의미/분모 검토 후
이 목록을 확장하며 arbitrary config나 tensor를 받지 않는다. 이름이 다른 기존
지표는 caller에서 위 이름으로 mapping한다. phase/status/cohort는 숫자 ID로
로그하고 의미표는 tracked task config에 둔다. source/config SHA와 논리 identity만
run config로 업로드한다. parent run 연결은 `parent_run_id` config 필드다.

한 arm/attempt마다 새 UUID run이다. SDK 자체 네트워크 재연결은 같은 run에
이어지지만 모델 crash resume 권한/보장은 없다. `resume='never'`는 별도 실행이
과거 과학 run을 이어 쓴 것으로 오인되지 않도록 한다.

## 기록 실패·개인정보 경계

parent `log`는 bounded queue에만 넣는다. SDK 지연이 scientific thread를 막지
않는다. 거절/queue 초과는 False 및 dropped count로 표시한다. accepted scalar만
로컬 JSONL에 보존한다. SDK 반환은 remote delivery ACK가 아니며 비동기 연결
재시도는 SDK에 맡긴다. 로컬 오류는 LOGGING_DEGRADED, flush는 최대45초 기본;
scientific 결과 저장/원 예외를 logger가 대체하지 않는다. 끊긴 통신을 이유로
과학 fit을 재실행하지 않는다. 별도 heartbeat/monitor process는 없다.

SDK worker는 CPU affinity2, address-space4GiB, GPU 비가시로 제한한다. 자동
console/code/git/requirements/system stats/machine metadata/job artifact 생성을
끄고 `watch`, `save`, `log_artifact` API를 제공하지 않는다. SDK stdout/stderr는
부모에게 전달하지 않는다. SDK exception 문자열도 기록하지 않는다. SDK 내부
transport bookkeeping은 존재하며 remote 파일 검사는 online smoke에서 수행한다.
noCP 정책은 그대로다. run ID와 transport spool은 ignored local만 사용한다.

## 한 번의 online smoke

```bash
python -m project.run_scripts.experiment_tracking.smoke \
  --env-file /mnt/raid5/janghj/ODE-edit/servers/local/wandb.env \
  --out /mnt/raid5/janghj/ODE-edit/local/wandb-setup/smoke-new \
  --server server1 --source-sha ACTUAL_40_HEX_COMMIT
```

합성 `setup_ok/step` 3점 한 run만 만들고 bounded finish 후 remote history3점을
한 번 읽는다. 새로 같은 smoke를 실행하려면 새 out을 명시해야 한다. 인증이 없으면
run 생성 없이 상태만 기록한다. 단위검사는 fake SDK이며 online 검증과 별개다.

```bash
python -m unittest project.run_scripts.experiment_tracking.test_tracking -v
```

공식 근거: [W&B init](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py),
[Settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py),
[login](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_login.py).
실제 채택은 local SDK0.30.0 source/settings 필드와 결속한다. API 변경시 silent
fallback하지 않고 version 검토부터 수행한다.

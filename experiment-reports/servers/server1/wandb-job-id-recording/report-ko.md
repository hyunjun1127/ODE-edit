# W&B Slurm job ID 기록 구현

Nonce: `USER-GH-ALL-SH-WANDB-JOB-ID-20261007-SERVER1`.

SH1 공통 helper 구현 완료. 부모가 네 Slurm 환경 키만 수집하고 실제 raw job ID,
array parent/task(0 포함), step, 표시 ID를 config/receipt에 전달한다.
run.name에 `job<표시ID>`를 포함하며 UUID/group 정책은 유지한다.
caller ID와 실제 환경 충돌, array 불완전, 가짜 local job ID는 명시 오류다.
기존 startup API readback에서 name/config를 검산한다. sidecar 환경에 전체 Slurm
환경을 복사하지 않으며 code/console/artifact 비공개 정책은 그대로다.

검사: 기존 17 + 신규 7 = CPU/fake-SDK 24 tests PASS. 실제 SDK 0.30.0의
settings/finish signature도 확인했다. fake ID는 CPU fixture 내부에서만 사용했다.
별도 Slurm/GPU/login smoke 실행 0. 아래 승인된 GPT2 CPU 검증 job의 startup은
실패했으며, 실제 Slurm online ID 검증은 NOT_VERIFIED다.

재현: `local/wandb-setup/20261006/venv/bin/python -m unittest
project.run_scripts.experiment_tracking.test_tracking
project.run_scripts.experiment_tracking.test_job_identity -v` (repo root 기준).

호출자는 기존 `init(env_file=..., spool=..., config=...)` 그대로 사용한다.
`job_id='CPU'` 같은 placeholder를 제거한다. 부모 환경으로 자동 결속되며,
호출자가 job_id를 명시할 경우 같은 실제 값만 허용한다.
현재 GPT2 준비 source는 미봉인·미제출 상태에서 채택했다. 과거 실험·run·job 변경 0.
독립 reviewer 사용 0, owner source 및 production assembly 검토/회귀 수행.
원 root dirty와 다른 SH 변경 보존. NO_BROADCAST_NOT_REQUIRED.

## 최초 실제 startup 실패와 미검증 경계

GPT2 task의 CPU jobs 60071/60072/60074는 `bind_job_identity` 안의
`step_id` 일반 identifier 검증에서 `INVALID_IDENTIFIER`로 종료했다.
원 helper 실행 source `fb8457f9c31ebe9376a6781caef9f8497b9eb583`와 로그는 보존했다.
이 경계는 SDK init 이전이므로 새 run 생성·원격 job-ID 확인은 이루어지지 않았다.
실제 step 원문은 당시 기록하지 않아 값/세부 형태는 UNKNOWN이다.

CPU 반례에서 빈 optional step과 signed step sentinel은 기존 일반 identifier에
맞지 않음을 재현했다. 수정에서는 빈 step은 없음으로 처리하고, signed numeric
step은 job ID와 구분해 원 문자열로 보존한다. 실제 job_id는 여전히 양의 정수이며,
caller/env 충돌 및 임의 문장은 차단한다. 이 수정의 실제 Slurm 원인 해결 여부는
NOT_VERIFIED이며 CPU fixture PASS로 대신하지 않는다.
새 Slurm test/자동 재제출 0. 다음 새 source부터 사용하고 frozen job은 수정하지 않는다.

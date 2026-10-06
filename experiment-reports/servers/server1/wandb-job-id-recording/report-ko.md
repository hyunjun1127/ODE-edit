# W&B Slurm job ID 기록 구현

Nonce: `USER-GH-ALL-SH-WANDB-JOB-ID-20261007-SERVER1`.

SH1 공통 helper 구현 완료. 부모가 네 Slurm 환경 키만 수집하고 실제 raw job ID,
array parent/task(0 포함), step, 표시 ID를 config/receipt에 전달한다.
run.name에 `job<표시ID>`를 포함하며 UUID/group 정책은 유지한다.
caller ID와 실제 환경 충돌, array 불완전, 가짜 local job ID는 명시 오류다.
기존 startup API readback에서 name/config를 검산한다. sidecar 환경에 전체 Slurm
환경을 복사하지 않으며 code/console/artifact 비공개 정책은 그대로다.

검사: 기존 17 + 신규 6 = CPU/fake-SDK 23 tests PASS. 실제 SDK 0.30.0의
settings/finish signature도 확인했다. fake ID는 CPU fixture 내부에서만 사용했다.
별도 Slurm/GPU/login smoke 실행 0. 실제 Slurm online ID 검증은 아직 미수행이며
다음 승인된 GPT2 CPU 자산 검증 job의 기존 bounded startup에서 확인한다.

재현: `local/wandb-setup/20261006/venv/bin/python -m unittest
project.run_scripts.experiment_tracking.test_tracking
project.run_scripts.experiment_tracking.test_job_identity -v` (repo root 기준).

호출자는 기존 `init(env_file=..., spool=..., config=...)` 그대로 사용한다.
`job_id='CPU'` 같은 placeholder를 제거한다. 부모 환경으로 자동 결속되며,
호출자가 job_id를 명시할 경우 같은 실제 값만 허용한다.
현재 GPT2 준비 source는 미봉인·미제출 상태에서 채택했다. 과거 실험·run·job 변경 0.
독립 reviewer 사용 0, owner source 및 production assembly 검토/회귀 수행.
원 root dirty와 다른 SH 변경 보존. NO_BROADCAST_NOT_REQUIRED.

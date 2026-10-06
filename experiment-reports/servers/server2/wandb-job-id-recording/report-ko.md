# SH2 W&B job 번호 기록 정책 수락

- nonce: `USER-GH-ALL-SH-WANDB-JOB-ID-20261007-SERVER2`
- 상태: **POLICY_ACCEPTED_FUTURE_SOURCE_ADOPTION_REQUIRED**
- 정본: `50eb29c5659242864411e1a36703423d459d508c`. Envelope 및 policy 전체 읽기와 SHA 결속은 ACK/감사에 기록했다.
- 실제 경계: server2, SH2 session `01a0493a-074c-7f91-9a13-769116326fef`, CWD `/mnt/raid5/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit`.
- 별도 clean worktree/branch에서 본 문서만 작성한다. 원 root dirty 및 봉인 source/job은 보존한다.

## 적용 약속과 현재 구현 경계

향후 새 Slurm-backed run은 실제 SLURM_JOB_ID string을 Config.job_id에, 표시용 job ID를 run.name의 `job<job_display_id>`에 기록한다. Array는 raw job ID와 parent/task를 분리하고 task index 0도 보존한다. Step ID가 있으면 함께 기록한다. Run UUID/attempt/group 정책은 유지하며 job 번호만으로 run을 덮어쓰지 않는다.

Parent에서 whitelist Slurm metadata만 전달하고 caller ID와 실제 환경이 충돌하면 명시 identity error로 처리하는 SH1 공통 helper를 새 source 봉인 때 채택한다. SH2는 공통 helper를 수정하지 않았다. Local CPU 실행은 backend=local, job identity=NOT_APPLICABLE이며 가짜 job 번호를 쓰지 않는다.

정본 source의 좁은 읽기 확인 결과, 현재 worker.py의 run.name은 server-arm-attempt이고 job 번호가 없다. FE telemetry.py는 이미 job_id를 config에 전달하지만 이것만으로 새 요구사항 충족을 주장하지 않는다. 공통 helper의 새 구현·배포 및 실제 online job identity 검증은 이번 SH2 정책 수락과 별개다. 현재 implementation_verified=false, online verification=NOT_TESTED다.

다음 승인된 신규 source에서 SH1 helper exact commit을 결속하고 기존 startup bounded readback 안에서 run.name/Config의 실제 ID 일치를 확인한다. 이 검증을 위한 새 실험·Slurm smoke·가짜 번호 업로드는 하지 않는다.

## 보존·privacy·검사 범위

기존 FE 및 모든 frozen/released job, 과거 W&B run은 hotpatch/restart/rename/backfill하지 않는다. GPU/Slurm 제출·조회·변경, W&B API 호출, 모델 평가, monitoring 재개는 모두 0이다. 과거 실행의 현재 상태를 이번 기록으로 추정하지 않는다.

Entity `wkdguswns2256`, project `layer allocation`(공백 포함), scalar/config whitelist와 noCP 정책을 유지한다. Secret/fullenv/argv/prompt/tensor/code/stdout 자동 업로드는 금지한다.

검사는 owner의 문서·정책 hash·source 읽기·JSON/scope/diff 검사다. 독립 reviewer 및 fake-SDK/GPU/online 새 검사는 수행하지 않았다. 공통 helper 검사는 SH1 소유이며 이를 SH2 PASS로 대체하지 않는다.

NO_BROADCAST_NOT_REQUIRED. 정책 수락 보고 게시 후 STOP; 향후 실제 새 source 적용 의무는 남는다.

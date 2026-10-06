# SH3 향후 W&B job 번호 기록 정책 수락 — 2026-10-07

**POLICY_ACCEPTED_FUTURE_CALLER_ADOPTION_REQUIRED.** 이번 SH3 작업에는 미봉인 신규 과학 launcher가 없다. 정책과 다음 source freeze의 필수 적용 지침을 등록했다. 공통 helper 구현은 SH1 소유이며 SH3는 수정하지 않았다. 정책 수락, helper 배포, 실제 online 기록 검증을 구분한다.

정본 main `50eb29c5659242864411e1a36703423d459d508c`의 서버 공통 envelope와 `control/wandb-policy.json`을 FULL_READ했고, `job_identity_required`가 envelope requirement와 정확히 일치함을 확인했다. Envelope SHA256 `d9b3d7904420d621ba82d25fdfbc42677b679b52a4c358f1e117468c064fb3ca`를 검산했다. 실제 경계는 ubuntu / `/data/janghj/ODE-edit` / SH3 session `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`다. 기존 동일 nonce ACK는 없었다.

## 다음 신규 run에 적용할 계약

| 항목 | SH3 적용 |
|---|---|
| `Config.job_id` | 실제 job 안의 `SLURM_JOB_ID` 문자열. 제출 전 번호 추정 금지 |
| Array | `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`를 각각 문자열로 기록. task index `0`도 유효 |
| Display | 일반 job은 raw job_id, array는 parent_task. raw job_id와 `job_display_id`를 별도 보존 |
| Step | 실제 `SLURM_STEP_ID`가 있으면 기록 |
| `run.name` | 항상 `job<job_display_id>` 포함, server/arm/attempt 구분 유지 |
| Backend/source | `execution_backend=slurm`, `identity_source=SLURM_ENV` 또는 명시 검증된 submission binding |
| 전달 경계 | Parent에서 위4개 Slurm env만 채집하여 sanitized config/receipt/sidecar로 전달. sidecar env에 Slurm이 없다고 local로 오판하지 않음 |
| 불일치 | Caller 실제 ID와 env가 다르면 명시 identity error. Silent overwrite 금지 |
| UUID/group | 기존 attempt UUID/run.id·task group 유지. job 번호만으로 run 덮어쓰기 금지. 같은 job의 여러 arm은 별도 run |
| Local CPU | backend=local, job identity NOT_APPLICABLE. 가짜 job/`0`/`UNKNOWN` 업로드 없음 |

다음 SH3 신규 source를 봉인할 때 **SH1이 게시한 job identity 구현 commit/API/import closure를 결속**하고 caller가 실제 runtime identity를 전달하는지 확인한다. 접수 시점 helper는 job_id가 optional이고 worker의 run.name은 server-arm-attempt여서 아직 이 요건을 충족하지 않았다. 기존 helper 상태를 구현 완료나 online PASS로 표시하지 않는다. 새 helper가 게시되면 그 API에 맞춰 미래 caller만 채택하며, 같은 기능을 SH3 namespace에 복제하지 않는다.

다음 실제 run의 기존 bounded startup API readback에서 name과 config의 raw job/array/step/display ID가 일치하는지 확인한다. 이를 위한 새 Slurm/GPU smoke·과학 재실행은 만들지 않는다. Normal/array0/step/local/missing/mismatch/privacy fake-SDK 검산은 SH1 공통 구현 범위이며 이번 SH3는 수행하지 않았다.

## 보존·검증 경계

기존 frozen/released source/job, 과거 W&B run 이름·config는 그대로 유지한다. Hotpatch/rename/backfill/재시작/취소/기존 run 재조회0, 과학 평가0, GPU0, Slurm조회·제출0, SDK/인증 설정 변경0이다. 현재 scientific task 재개나 장기 monitoring 권한은 없다.

Entity `wkdguswns2256`, project `layer allocation`(공백 유지), online/scalar whitelist/noCP 정책은 그대로다. Full env/argv/API key/prompt/tensor/code/stdout 자동 업로드는 금지한다. 실제 온라인 인증·기록은 이번에 확인하지 않았으며, 과거 인증 상태를 현재 상태로 추정하지 않는다.

Owner가 정책 일치·파일 SHA·소유 경로·문서 검산만 수행했다. 독립 agent reviewer는 사용하지 않았다. [수락 및 source 관측 receipt](../../../../audits/servers/server3/wandb-job-id-recording/acceptance.json)와 [서버 상태](../../../../tasks/status/wandb-job-id-recording/server3.json)에 검증 수준을 분리했다. Own-scope nonforce main 게시 후 ACK 사실 인계 및 STOP.

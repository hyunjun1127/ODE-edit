# SH1 W&B 실시간 기록 설정

상태: **SETUP_READY_NEEDS_USER_LOGIN**. SDK·비민감 설정·공통 helper는 준비됐다.
기존 인증이 없어 실제 online run/프로젝트 권한/3점 remote readback은 검증하지 못했다.
offline run을 만들거나 이를 online PASS로 표시하지 않았다.

| 항목 | 확인 결과 |
|---|---|
| SDK | 별도 venv, wandb 0.30.0 / Python 3.10.12 |
| 설정 | entity `wkdguswns2256`, project `layer allocation`(공백 유지), online |
| API endpoint | 공식 SDK 기본 `https://api.wandb.ai`; UI URL에서 추정하지 않음 |
| 인증 | 표준 환경변수·netrc·설정에서 미확인, SDK noninteractive login도 인증 없음 |
| online smoke | 인증 단계 종료; run 생성0, remote point0, URL 없음 |
| CPU 회귀 | fake SDK16 + 실제 SDK signature1 =17검사 통과; online 검증과 별개 |
| 기존 실험 | GPU/model/Slurm 신규0, 기존 job 조회·변경0 |

## 설치·설정 경로

- SDK Python: `/mnt/raid5/janghj/ODE-edit/local/wandb-setup/20261006/venv/bin/python`
- 비민감 config: `/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env` (ignored)
- 실제 smoke: `/mnt/raid5/janghj/ODE-edit/local/wandb-setup/20261006/smoke-r1/result.json`
- smoke 결과 SHA256: `45eb5fcb43d650b2477936e919072a12bbea1c3cb2a7067d0cf2b69d335a2233`

원 과학 환경 `/mnt/raid5/janghj/EasyEdit/.venv`의 torch/transformers 또는 기타 pin은
변경하지 않았다. credential 값·키 hash·전체 환경·full stdout는 보고/Git에 없다.
Local SDK/credential/spool은 다른 서버로 전송하지 않는다.

## 사용자에게 필요한 다음 작업

server1 terminal에서 아래 명령으로 직접 로그인한다. 키는 숨김 입력으로만 전달하고
채팅이나 command argument에 붙이지 않는다.

```bash
/mnt/raid5/janghj/ODE-edit/local/wandb-setup/20261006/venv/bin/wandb login
```

로그인 후 한 번의 CPU online3점·finish/readback이 남았다. 해당 검증 전에는 새 GPU
실험의 logging-ready를 주장하지 않는다. 인증/프로젝트 권한이 실제로 없는 경우
그 상태를 보고하며 자동 retry/monitor나 기존 실험 재개는 하지 않는다.

## 공통 helper 인계

재사용 source: `93912c3fdb40fa0fe70d115e48b6abd9672af96b`.
실제 auth smoke source는 cleanup 추가 이전 `fe9bbb84f312f69bdd1484265bb7db8a3eac3cd9`다.
후속 cleanup 예외 보존 검사와 실제 SDK0.30 signature 검사를 추가했다.
`Run.finish`에서 지원하지 않는 `quiet` 인자를 제거했고 CPU fixture로 검증했다.
online smoke를 반복하지 않았다.

`project.run_scripts.experiment_tracking.init(env_file, spool, config)`로 startup을
검증한 뒤 반환된 `Tracker.log(existing_scalars, step=...)`, `Tracker.finish()`를 사용한다.
[API·새 launcher 사용법](../../../../project/run_scripts/experiment_tracking/README.md)은
repository의 `project/run_scripts/experiment_tracking/README.md`에 있다.
각 SH는 자기 SDK/config/auth만 준비하며 동일 helper를 재사용할 수 있다.

설정·metric은 명시 allowlist이며 tensor/string prompt/임의 config를 받지 않는다.
SDK는 별도 CPU process에서 동작하고 console/code/git/requirements/system metadata/
watch/automatic artifact를 사용하지 않는다. 과학 thread의 `log`는 bounded queue만
사용한다. SDK queue accepted는 remote 수신 ACK가 아니므로 finish/readback과 구분한다.
네트워크/기록 실패는 LOGGING_DEGRADED 및 local scalar spool 보존으로 처리하며
과학 결과 저장이나 원 예외를 대체하지 않는다. noCP·exact-model-resume 경계는 그대로다.

독립 reviewer agent는 사용하지 않았다. owner 검토 및 fake SDK CPU fixture를 수행했다.
실제 모델/GPU 검증은 이 설정 task 범위가 아니다. 기존 source/dirty/중단 task는 보존했다.

공식 API 근거는 [W&B Settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py)와
[login](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_login.py)이며, 사용한
SDK0.30.0의 실제 local source/Settings 필드를 확인했다. 설치 성공과 인증 성공을
구분했다. NO_BROADCAST_NOT_REQUIRED.

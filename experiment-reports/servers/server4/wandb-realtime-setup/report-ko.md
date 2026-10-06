# Server4 W&B 실시간 기록 준비

상태: `READY_ONLINE_VERIFIED`. 지시 `USER-GH-ALL-SH-WANDB-REALTIME-20261006-SERVER4`.

## 실제 검증

- entity `wkdguswns2256`, project `layer allocation`(공백 유지), mode `online`.
- [CPU smoke run](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/server4-setup-20261006): `setup_ok=1`, step 0/1/2를 기록하고 finish 후 원격 history에서 정확히 3개 확인. remote state `finished`.
- 기존 해당 사용자의 로컬 인증을 사용했다. 키 값·키 hash·전체 환경은 출력/저장/전송하지 않았다. 원격 인증과 지정 프로젝트 기록 권한은 smoke로 검증했다.
- GPU 0, 모델 로딩 0, 신규 Slurm 0, 기존 job/source 변경 0. CPU affinity 0/1, 주소공간 상한 4 GiB. smoke wall 6.269초, Python parent peak RSS 69,560 KiB(자식 합계 peak 아님).
- 자동 code/git/console/system metadata/system metrics/requirements 수집을 비활성화했다. watch/artifact 호출 0. 원격 파일은 `config.yaml`, `wandb-summary.json` 두 개였다.

## 설치와 설정

- Python 3.12.3: `/data/janghj/ODE-edit/local/wandb-setup/sdk/bin/python`
- SDK `wandb==0.30.0`: 같은 환경의 `lib/python3.12/site-packages/wandb/`.
- 비민감 설정: `/data/janghj/ODE-edit/servers/local/wandb.env`(ignored).
- SDK/spool/receipt: `/data/janghj/ODE-edit/local/wandb-setup/`(ignored). 과학 env pin 변경 없음.
- 기본 python3 `venv`의 ensurepip 부재를 확인한 뒤 기존 uv로 독립 환경을 생성했다. 시스템 패키지는 설치하지 않았다.
- API는 SDK 공식 기본값 `https://api.wandb.ai`를 사용했다. 사용자 UI `https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation`를 API 주소로 추측하지 않았다. [공식 SDK 설정](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py).

## 향후 신규 launcher 결속

새 launcher의 `--export=NONE` 내부에서 위 비민감 env 파일을 명시적으로 source하고, 그 **새 job 안에서만** `WANDB_DISABLED`를 unset한다. 기존 해당 사용자 HOME의 SDK 인증 파일을 그대로 발견하게 하며 키를 인자/로그/Git로 전달하지 않는다. 다른 사용자 HOME을 대입하거나 인증 파일을 복사하지 않는다. HF offline은 유지한다.

SDK 전용 Python은 CPU 로깅용이다. 과학 runner의 Python을 이 환경으로 바꾸거나 SDK dependency를 과학 env에 덮어쓰지 않는다. SH1의 공통 helper API가 도착하면 새 source/launcher에서 호환 import 또는 격리 logger 방식을 명시 결속하고 startup online 검증을 수행해야 한다. 설정 파일만으로 기존/미계측 코드에 자동 logger가 붙었다고 주장하지 않는다.

공통 helper `project/run_scripts/experiment_tracking/**`는 SH1 단독 소유다. 확인한 origin/main `d62887b7`에는 아직 없었다. SH4는 이를 복제 구현하지 않았다. 현재 신규 미제출 실험은 이 설정 작업 범위에 없으므로 생산 launcher integration은 `NOT_PERFORMED_NO_NEW_EXPERIMENT_IN_SCOPE`다. 향후 helper는 scalar/config whitelist, watch/artifact 금지, 원 예외 보존, 로컬 spool, bounded finish와 logging failure-isolation을 충족해야 한다. 본 smoke는 공통 helper 생산 검증을 대체하지 않는다.

## 근거와 범위

정본 policy/envelope FULL_READ/SHA 검증 완료. 실제 session/host/origin 일치와 전용 WT boundary PASS. root의 과거 session-boundary와 dirty 파일은 보존했다. 원격 3-point readback 외 추가 실험/장애 주입/반복 감시 없음. 검토 수준은 owner audit이며 별도 독립 reviewer를 사용하지 않았다.

- [실행 receipt](../../../../runs/wandb-realtime-setup/server4.json)
- [검산 및 SHA](../../../../audits/servers/server4/wandb-realtime-setup/verification.json)
- [단발 smoke source](../../../../audits/servers/server4/wandb-realtime-setup/smoke.py)

Git에는 source와 compact receipt만 게시한다. `NO_BROADCAST_NOT_REQUIRED`: SDK/credential/spool/raw는 로컬 보존, 타 서버 복사 없음. `monitoring_active=false`, `automatic_resume=false`.

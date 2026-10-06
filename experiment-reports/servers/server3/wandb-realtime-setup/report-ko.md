# Server3 W&B 실시간 기록 환경 준비 — 2026-10-06

**상태: SETUP_READY_NEEDS_USER_LOGIN.** 격리 SDK와 비민감 설정은 준비했으나 서버3 사용자 인증이 없어 online run을 생성하지 않았다. 인증·프로젝트 접근·3점 원격 readback은 모두 미검증이다. Offline 결과를 online PASS로 대체하지 않았다.

| 항목 | 실제 확인 |
|---|---|
| 담당/경계 | head-server3, ubuntu / janghj, session `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`; 전용 non-main worktree boundary PASS |
| 정본 | `ff43b43887f2aeabba35a04a2769ac045c5c8b86`, 서버 envelope 및 W&B 정책 FULL_READ/SHA 결속 |
| Python / SDK | 격리 Python3.12.3 / wandb0.30.0, import 및 uv dependency check26packages PASS |
| 대상 | entity `wkdguswns2256`, project **`layer allocation`** — 공백 유지 |
| API endpoint | 설치 SDK 기본 `https://api.wandb.ai`; Forge UI에서 API주소 추정하지 않음. 계정 endpoint/권한 검증은 로그인 후 필요 |
| 인증 | SDK `login(prompt=False, verify=False)` 결과 false. 키·키hash·전체환경·인증 파일 내용 출력/복사0 |
| online smoke | 실행 전 인증 gate에서 중단; run0, remote point0, run URL 없음 |
| 검사 | 비민감 env→SDK 설정 검산 및 합성 CPU회귀3 PASS. 온라인 검증 아님 |
| 공유 logger | SH1 단독 소유. SH3는 설치 및 일회 setup smoke만 작성; 신규 실험 source가 없어 launcher integration 미수행 |
| 과학 자원 | GPU0/model0/Slurm조회0/제출0/기존job변경0. 기존 중단 task 재개0 |

## 경로와 재현

- SDK Python: `/data/janghj/ODE-edit/local/wandb-setup/20261006-v1/venv/bin/python`
- SDK CLI: `/data/janghj/ODE-edit/local/wandb-setup/20261006-v1/venv/bin/wandb`
- Nonsecret 설정: `/data/janghj/ODE-edit/servers/local/wandb.env` 및 작업 worktree의 동일 ignored 파일.
- 작업 worktree: `/data/janghj/ODE-edit/local/wandb-setup/20261006-v1/worktree`
- [SDK lock](../../../../audits/servers/server3/wandb-realtime-setup/sdk-lock.txt), [설정 예제](../../../../audits/servers/server3/wandb-realtime-setup/wandb.env.example), [검사 receipt](../../../../audits/servers/server3/wandb-realtime-setup/checks.json), [smoke 코드](../../../../audits/servers/server3/wandb-realtime-setup/smoke.py).
- 기존 과학환경 torch2.9.1+cu128 / transformers4.44.2의 설치를 변경하지 않았다. SDK venv를 과학 Python의 PYTHONPATH에 무조건 주입하지 않는다.

사용자가 **server3 terminal**에서 아래처럼 로그인한다. 키는 대화형 비표시 입력에만 넣고, 채팅·명령 인자·Git·로그로 전달하지 않는다. 현재 기본 cloud 외 검증된 계정 endpoint가 있다면 해당 endpoint를 사용자 설정으로 확인해야 한다.

```bash
/data/janghj/ODE-edit/local/wandb-setup/20261006-v1/venv/bin/wandb login --cloud --verify
```

로그인 뒤 준비한 일회 CPU smoke를 실행할 수 있다. 이는 현재 online PASS를 뜻하지 않는다. 정확 세 synthetic scalar point만 보내며 기존 task-level attempt marker가 있으면 두 번째 run 생성을 거부한다. 원격 readback은 한 번만 수행하고 반복 monitoring하지 않는다.

```bash
cd /data/janghj/ODE-edit/local/wandb-setup/20261006-v1/worktree
set -a
source /data/janghj/ODE-edit/servers/local/wandb.env
set +a
timeout --signal=TERM --kill-after=5s 120s   /data/janghj/ODE-edit/local/wandb-setup/20261006-v1/venv/bin/python   audits/servers/server3/wandb-realtime-setup/smoke.py
```

CPU affinity2/프로세스 주소공간4GiB, init30s/finish30s/readback15s/외부120s 제한을 사용한다. SDK stdout/stderr는 Python stream과 실제 fd 모두 차단한다. 예외 문자열 대신 class만 기록하며 원 오류와 finish 오류를 구분한다. SDK local spool은 ignored 경로에 보존하고 Git에 올리지 않는다.

## 향후 새 launcher 경계

`--export=NONE`인 새 job 내부에서 위 비민감 설정을 명시적으로 source하고, legacy `WANDB_DISABLED`만 해당 새 process에서 해제한다. HF offline 설정은 유지한다. 인증은 동일 사용자의 private netrc/검증된 로컬 credential source를 SDK가 읽게 한다. 키를 sbatch 인자·설정파일 예제·다른 서버에 복사하지 않는다.

공통 logger API는 SH1의 `project/run_scripts/experiment_tracking/**`를 사용하며 SH3 복제 구현은 없다. 신규 science launcher에서 SH1 helper source/API와 실제 과학환경 호환성을 결속한 후 사용해야 한다. 이 설정만으로 과학 코드에 자동 계측이 삽입되지는 않는다. 새 GPU 실험 전 cheap online 확인이 실패하면 `NEEDS_USER_LOGIN/LOGGING_BLOCKED`로 명시한다.

업로드는 기존 scalar/분모/논리ID/source·config SHA whitelist에 한정한다. Console/code/git diff/watch/automatic artifacts, 원문 prompt·dataset·모델·W/H/RNG·activation·checkpoint·fullstdout은 금지한다. 기존 metric만 batch/phase/candidate 또는10–30초 간격 비동기 기록하며 새 평가/forward/강제GPU동기화를 만들지 않는다. 런 중 전송 실패는 spool 보존과 `LOGGING_DEGRADED`로 처리하고 과학 재실행 사유로 삼지 않는다. Finish는 bounded이며 결과 저장/원 예외를 가리지 않는다. 이 공통 logger 런타임 동작은 이번 서버 setup에서 실험으로 검증하지 않았다.

## 검토 수준과 제한

Owner가 설정·SDK·auth gate와 합성회귀3개를 실행했다. 별도 `wandb_server3_audit` reviewer는 정책 및 smoke/config/test의 읽기 전용 검토만 수행했다. 이 reviewer가 찾은 per-output 중복 run 가능성을 고정 task marker로 수리하고 서로 다른 output 경로의 회귀를 통과했다. Reviewer는 인증·네트워크·GPU·실험을 검사하지 않았다.

인증이 없어 remote project 존재/권한/3점 기록·finish/readback은 **NOT_VERIFIED**다. 현재 막힌 부분은 해당 서버 사용자의 로그인이며 추가 과학 승인이나 GH 감사 대기가 아니다. 기존 science source/job/중단상태는 보존한다. 상세 상태와 manifest는 아래 문서에 결속한다.

- [서버 상태](../../../../tasks/status/wandb-realtime-setup/server3.json)
- [소형 산출물 manifest](../../../../audits/servers/server3/wandb-realtime-setup/manifest.json)

범용 `check-agent-access.sh --staged`는 `runs/wandb-realtime-setup/server3.json`의 prefix를 지원하지 않아 exit7/NOT_PASS였다. 이번 서버 envelope의 exact 허용 경로로 게시하며 공용 검사기/공유 identity를 수정하지 않았다. 그 외 변경은 SH3 own scope다.

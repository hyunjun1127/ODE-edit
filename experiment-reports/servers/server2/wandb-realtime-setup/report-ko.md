# Server2 W&B 설정 — 안전 로그인 필요

Nonce: `USER-GH-ALL-SH-WANDB-REALTIME-20261006-SERVER2`.
상태: **SETUP_READY_NEEDS_USER_LOGIN**. SDK 설치/설정 성공과 online 검증을 구분한다.

- 실제 server2 / janghj / session `01a0493a-074c-7f91-9a13-769116326fef` / `hyunjun1127/ODE-edit`.
- 독립 SDK: W&B **0.30.0**, Python `/mnt/raid5/janghj/ODE-edit/local/wandb-setup/sdk-env/bin/python`.
- 과학 venv/Torch/Transformers 수정0. SDK와 dependencies는 ignored 독립 venv에만 설치했다.
- 비민감 설정: `/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env`.
- entity=`wkdguswns2256`, project=`layer allocation`(공백 유지), mode=`online`.
- API endpoint는 SDK 공식 cloud 기본값을 사용했다. UI URL을 API endpoint로 추측하지 않았다.
- console/code/git/system metadata/system metrics 자동 수집은 끄고 watch/artifact 호출은 사용하지 않는다. Config는 server label만 허용했다.
- SDK의 비대화형 credential 탐색 결과 인증정보 부재. 키/credential 객체/전체env/키hash는 출력·기록·복사하지 않았다.
- 따라서 online run 생성0, 3-point upload0, remote readback0, run URL 없음. Offline를 PASS로 보고하지 않는다.
- CPU2/4GiB/90초 제한의 one-shot smoke 준비·실행, GPU0/model0/Slurm0/기존job 변경0.
- SH1 소유 공통 logger는 중복 구현하지 않았다. 당시 main에서 helper는 없었으며 peer app tool도 전달 불가 응답이었다. 직접 전달 성공을 주장하지 않는다.
- Owner review이며 independent reviewer=0. Local receipt: `/mnt/raid5/janghj/ODE-edit/local/wandb-setup/initial/receipt.json`.

## 사용자 조치

Server2의 **본인 terminal**에서 아래 명령으로 안전 로그인한다. 키를 채팅/명령 인자/Git에 넣지 않는다.

```bash
/mnt/raid5/janghj/ODE-edit/local/wandb-setup/sdk-env/bin/wandb login
```

이후 같은 서버에서 새 attempt 이름으로 CPU online 3-point smoke/readback을 한 번 확인한다.
`audits/servers/server2/wandb-realtime-setup/setup_smoke.py --attempt after-login`이 재현 경로다.
그 다음 SH1 common helper API/source를 FE에 연결하고 새 source/input/online lock을 봉인한다.

FE는 정책 수신 시 미봉인/미제출이었다. 구현·CPU·입력 준비는 보존하되 현재 **GPU job ID 없음**이다.
인증과 helper 미결속을 우회해 제출하지 않는다. 기존 frozen 실험에는 hotpatch/취소/재시작하지 않았다.

설치 receipt와 privacy Settings 검산은 online auth/project 권한 검증을 대체하지 않는다.
[SDK privacy 설정 근거](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py).
`NO_BROADCAST_NOT_REQUIRED`: SDK/spool/credential 전송0, compact source/receipt만 Git 게시.

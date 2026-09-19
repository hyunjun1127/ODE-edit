# SH4 수신 / FULL_READ M0

- Instruction: `ODEEDIT-S06-SL-MECHANISM-FIRST-ZHOOK-SH4-V1`.
- 게시 source `be71bfba2454aa4645e28d960d495dfc25a65993`, tree `5e26c38e17074b563f155ad84d461a4874fe3c8d` 포함 확인. 새 branch `codex/server4-single-layer-mechanism-first-v1`.
- 작업 경로 `/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/worktree`; shared root/기존 dirty/실행 산출물 변경0.
- 사용자 원문·설계·contract·cells·두 reference·envelope·PROTOCOL 전체 읽기 완료. exact 사본과 SHA는 local `authoritative/`, `receipts/T0-attempt-v1-input-plan.json`에 결속했다. 원 attachment 정규화 이력은 GH authority manifest 그대로 보존했다.
- 위 input-plan SHA `ac15e7b0a025bf2fce652db593ff43c63bdf0d9e863ff9a21be9b7e940b7e7a2`는 실행 완료/실행 lock이 아닌 CPU 준비 기록이다.
- 원 hooking.py SHA `feb3509940a40e8b8027ee7daae4c7486fc43ec80394383627699be4f34f072b` 일치. Native compute_z SHA `a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f`; actual config loss layer31, L4, lr.1/decay.5/clamp.75/KL.0625/L2=1/25loss·24Adam.

## 범위와 미구현

T0→B1(N4/EN_KL_Q/DEC_LINE/DEC_MODES_CUM)의 구현·actual 검증을 진행한다. B1 STEP은 CUM alias이며 중복 실행하지 않는다. S3/S10은 정본 CUM gate 통과 시에만 제출한다. 과거 initial/PENDING pause, T/FD skip, checkpoint 미저장, storage waiver, EN 삭제/취소 권한은 상속하지 않는다.

M0 시점 새 model/controller/hook/solver/transaction/observer 연결은 구현 중이며 actual model 검증0/Slurm 제출0이다. 복잡한 z hook, basis/QCQP, decision/history 모듈을 파일 소유권을 나누어 세 helper가 구현하고 있다. 독립 red 감사 완료를 의미하지 않는다.

기존 R512/Dev128 generated256 capsule/teacher 및 W0 upstream은 prior READY와 현재 model/token/runtime binding을 검산해 재사용한다. 신규 문서선정/teacher 재생성은 기본0이다. 새 hook은 unhooked native / cache+head batch1 / 고정 소수 요청 batched 경로로 T0 비교한다. batch16 actual 검증이 없으면 성공으로 표기하지 않는다. 본 B1 native는 새 hook을 적용해 공통1회 생성하며 옛 EN endpoint로 대신하지 않는다.

## 자원·저장 및 helper 경계

resource-only 조회에서 현재 사용자 queue는 빈 상태다. 당시 `/data` free 61,164,986,368 bytes(약56.97GiB), free inode225,364,289. 제출 직전 재확인한다. project cap2/task cap2, job당GPU1/CPU8/60416MiB/exportNONE/Requeue0. T0/B1은 공통 native/derivative 중복을 피하는 단일 job 구성을 우선한다.

신규 B1 저장 상한 추정24GiB: CP/weight8 + factor4 + geometry5 + temporary/report7. 기존 약94.5GiB teacher/key/residual은 복제하지 않는다. S3/S10은 앞 stage 실측으로 별도 재산정한다. 현재 estimate는 reserved disk/실측 peak가 아니다. 삭제/이동/대형 원격전송0. `NO_BROADCAST_NOT_REQUIRED`.

T0 약2h/B1 약8h는 준비 추정이며 실제 kernel/회수별 측정 전이다. job wall24h, 사용자 GPUh hardcap 없음. 수치 호출 상한을 실제 GPU시간으로 쓰지 않는다.

registry의 현재 session/host/repo는 일치한다. session helper는 새WT의 local config 부재 및 shared root의 stale session mismatch로 NOT_PASS다. 실제 registry/명시 envelope와 구분하여 기록하고 helper/공유 설정은 변경하지 않는다.

실제 단계·기술오류·B1표·gate·최종보고를 계속 전달한다. GH 중복 raw/GPU 감사를 선행조건으로 요구하지 않는다.

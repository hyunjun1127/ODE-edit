# MEMIT HJ v2

정본은 `plans/global/2026-09-30-memit-hj-experiment-design-v1/`의 v2와
`messages/head/2026-09-30-memit-hj-sh3*.md`이다. 28개 cell을 새 namespace에서 실행한다.
기존 `MEMIT_seq` wrapper code object를 그대로 사용하고, 복제한 execute 함수의
residual 배분과 관측 callback만 바꾼다. 원 BLUE/EasyEdit/env 파일은 수정하지 않는다.

## 실제 physical DAG

|group|내용|선행조건|
|---|---|---|
|P|실제 T0a BS100 native/adapter, RAM·임시 CP 복원, W0 전체10k 관측|기존 project GPU 작업 afterany|
|A|W0 writer4(E0/T1 포함), 000/001, 첫1k의 T0b, 나머지 writer/history anchor|P afterok + bound readiness|
|B|100/101|P afterok, A afterany + readiness|
|C|010/011|P afterok, B afterany + calibration PASS|
|D|110/111|P afterok, C afterany + calibration PASS|
|CPU|독립 scalar/identity 검산, 보고/그림/manifest, 실패·차단 수집|P/A/B/C/D afterany, GPU0|

각 GPU job은 1GPU/8CPU/119GiB이며 동시에 하나만 실행된다. 분기·anchor는 해당
persistent job 안에서 독립 CPU RAM clone으로 실행한다. 다른 job에서 edited state를
이어붙이지 않는다. Z 교정 실패는 네 Z cell만 BLOCKED로 남긴다.

## 준비와 실행

고정 source archive를 만든 후 같은 frozen source에서 실행한다.

```sh
PY=/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover -s project/run_scripts/memit_hj -t . -v
"$PY" -m project.run_scripts.memit_hj.freeze --source ABS_FROZEN_SOURCE --commit EXACT_SHA --archive ABS_ARCHIVE --attempt NEW_TASK_ATTEMPT
"$PY" -m project.run_scripts.memit_hj.submit --lock NEW_TASK_ATTEMPT/execution.lock.json
```

`submit`은 여섯 job을 모두 held 등록·검사한 뒤 역순으로 release한다. 최초 reservation이
남으면 자동 재제출하지 않는다. 실제 job·argv·dependency·resource·source receipt가
있어야 REGISTERED/RELEASED라고 보고한다. 모의 Slurm fixture는 실제 제출이 아니다.

## 복원·수치·관측 경계

장기 000/100/110/111의 임시 CP만 1k 간격으로 허용한다. payload에는 W/H/RNG/context/
native cache binding/ledger/sample reference/dispersion/trigger/control cursor/source inputs가
들어간다. 최신2개를 유지하되 refcount가 있으면 보존한다. atomic SHA/실제 reload와 output
parity 후 manifest에 기록하며, 해당 path만 owner/hash/refcount 확인과 tombstone 후 삭제한다.
실패한 CP는 자동 정리하지 않는다.

동일 frozen 계약의 운영상 복원은 **새 output namespace**의 lock으로 다음 CLI를 사용한다.

```sh
"$PY" -m project.run_scripts.memit_hj.runner --lock NEW_RECOVERY_LOCK --group A --resume-checkpoint OLD_OWNED_LONG_CP
```

source/config/input binding이 다르면 거부한다. 기존 raw는 보존하고 검증된 prefix receipt만
hard link하며 CP 이후의 불완전 raw를 새 trajectory에 섞지 않는다. 장기111의 pending parent는
pin된110 CP로 복원한다. 단기/진단의 disk checkpoint 예외는 없다. source를 수리한 수치
trajectory는 이 recovery 경로가 아닌 fresh W0/H0로 시작한다.

native key는 context group의 평균을 다시 평균한다. dispersion은 정본 reference의
`sqrt(mean_context ||k_c-mean_context(k)||²)/||mean_context(k)||`를 그대로 유지한다.
두 정의를 임의의 가중 threshold로 합치지 않는다. 모든 occurrence는 H rebuild에 남는다.

Oracle는 pinned z-hook의 token/position/suffix/loss closure를 사용한다. KL teacher는
native full-model zero-delta forward에서 고정하며 prefix와 같은 요청 안에서만 재사용한다.
FP64 probe는 FP32 값과 cache를 승격하고 Llama의 강제 FP32 norm/softmax 연산도 task-local
함수에서 FP64로 실행한다. finally에서 원 함수·dtype·parameter identity를 복원한다.
PG error/call cap을 교정하기 전 Z production은 실행하지 않는다. FP32 oracle의 수치 차이와
actual T0 증거는 CPU PASS와 별개다.

RS/PS/NS preference, TF micro/macro/strict, NLL, at-write/past/continuation/W0-correct
전이를 원 forward의 scalar raw에서 집계한다. 공식 관측은 controller에 들어가지 않는다.
SPG·native·추가 probe·shadow·observer·CP 비용을 분리하며 nested timer를 더하지 않는다.

초기 인계는 대표 E0/T1의 실제 첫 BS10 commit/observer→next entry 또는 전량 등록 후의
확인된 resource pending에서 수행한다. 그 뒤 agent polling은 중단하며 프로그램과 CPU
collector만 자연 진행한다. owner audit와 CPU fixtures가 독립 reviewer 또는 실제 GPU PASS를
대체하지 않는다. 완료 상세 리뷰는 사용자 recall 때 수행한다.

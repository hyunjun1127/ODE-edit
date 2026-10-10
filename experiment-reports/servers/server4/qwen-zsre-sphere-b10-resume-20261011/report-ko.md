# Qwen zsRE AlphaEdit+SPHERE: server4 B10 재개

62538 `s2-qwen25-zsre-sphere-resume-b9`의 실패 원인은 GPU OOM이 아닌 **저장 공간 하한**이다.
child 로그의 최종 예외는 `RESOURCE_BLOCKED_STORAGE_KEEP_CHECKPOINT`이며,
B10/1,000건의 평가와 checkpoint 저장은 완료됐다. B11 편집 전 여유 공간 32 GiB 검사에서 중단됐다.
원62087의 B10 CUDA OOM과 구분한다.

사용자 지시에 따라 server4에 **63303 / s4-qwen25-zsre-sphere-resume-b10**을 실제 held 등록·검사 후 release했다.
관측 상태는 **PENDING**, 의존관계는 `afterany:63284`다. server4 기존 두 lane 중 짧은 lane의
마지막 작업 뒤에 연결했으며 기존 job의 취소·의존관계 변경은 0이다. GPU 동시 사용 cap은 2다.

- 자원: server4 RTX PRO 6000 96 GB 한 장, CPU 6개, RAM 58 GiB, wall ceiling 48시간, QOS lab_gpu_s4, no-requeue.
- 시작: **B10/1,000 edits**. 새 편집은 B11–B20, 1,000건이다. 새 W0/첫1K 재편집은 없다.
- ancestry: **62087 B1–B9 → 62538 B10 → 63303 B11–B20**. 원 checkpoint·commit·raw는 보존한다.
- frozen main source: `16512a230aa8ceb319680839c01ee41a1bf67640`. 실행은 그 commit의 official archive다.
- 원 scientific config SHA: `dcc7018f54b1f63c8c295ebe5d82b32fb933856e0d0b94b8659c9238ef41a814`. 하드웨어 경로·checkpoint I/O·서버/attempt binding은 별도 transfer config `ba9451090eb9da9bbbc0483c93fab39c77c11f7df3bf74e79a816d6c989f6605`에 기록한다.
- model/tokenizer/P는 server4의 기존 자산을 전체 SHA로 대조했다. Python numeric packages는 source와 동일하다.
- SPHERE·AlphaEdit helper·state adapter·기존 loader/evaluator 등 **23개 파일 byte가 62538 실행본과 동일**하다. 편집 hparams, FP32/eager, TF32 off, seed0, 공용 context를 유지한다.
- zsRE는 기존 public token-prefix/loc_ans 평가기를 사용하고 all-seen 1.5K/2K를 계산한다. 기존 B10 raw는 과거 관측으로 보존한다.

## Checkpoint 및 저장 공간

복사한 B10 checkpoint는 **8,535,444,185 bytes**, SHA256
`6ab0bea43a595f97df631176c0d5a6254eea90038b0c7c6eead8165c575abbd8`이다.
전체 SHA, FP32 편집층 가중치/history 형상·finite, 첫1K 순서, context, RNG 및 B9→B10 ancestry를 CPU에서 검증했다.
GPU 시작 시에도 정확한 W/H/context/RNG/cursor 복원을 확인한 뒤 B11을 수행한다.

server4 `/data`의 낮은 여유 공간을 피하기 위해 별도 root 파일시스템의
`/var/tmp/ode-edit-janghj-qwen-sphere-resume-20261011/`을 사용했다.
입력 B10과 출력 두 slot, raw 2 GiB, 여유 8 GiB를 포함해 34.04 GiB를 계산했고,
사전 여유 39.18 GiB를 확인했다. 출력 slot 두 개(각 8,602,553,049 bytes)를
**posix_fallocate로 실제 사전 할당**했다. 기존 파일 삭제는 0이다.

매 batch마다 비활성 slot에 쓰고 fsync·SHA 검증 후 `output/checkpoint/latest.json`만 atomic 교체한다.
이전 활성 slot은 쓰기 또는 pointer 발행이 실패해도 보존된다. 공통 checkpoint.load와 호환되는
가중치/history/RNG/context/identity payload다. 최종 W20과 입력 B10을 자동 삭제하지 않는다.
저장 경로는 위 root의 `output/checkpoint/`, 로그는 `logs/`다. 원 server2 B9/B10도 유지한다.

## 검증과 관측 범위

CPU unittest **26 PASS**: B10/ancestry 검증, malformed prefix/state 거절, 저장 실패 시 활성 checkpoint 보존,
두 slot 교대와 공통 loader 호환, synthetic resume W/H/RNG exact 일치, cap-2 DAG, 이전 B9 복원·SPHERE lifetime·mask·zsRE 평가 테스트.
server4 실제 tokenizer로 전체 2K query 대조도 PASS이며, 별도 pretrained GPU qualification은 `NOT_RUN_USER_DISABLED`다.
server4 GPU 실행/2K 완료/W&B online은 등록 시점에 아직 미관측이다. GPU가 바뀌므로 향후 계산의 bitwise 동등성을 주장하지 않는다.

[실제 제출](../../../../audits/servers/server4/qwen-zsre-sphere-b10-resume-20261011/submission.json) ·
[원 실패·B10 상태](../../../../audits/servers/server4/qwen-zsre-sphere-b10-resume-20261011/parent-terminal.json) ·
[CPU 검증](../../../../audits/servers/server4/qwen-zsre-sphere-b10-resume-20261011/cpu-validation.json).

# JLZ v4 compute-r1 A/B 2k — 준비 상태

상태: IMPLEMENTING_NOT_SUBMITTED. 실제 GPU 결과와 job ID는 아직 없다.

Instruction: `ODEEDIT-USER-GH-SH4-JLZ-V4-COMPUTE-R1-2K-20261002-R1`.
원 method와 별도 experiment는
[정본](../../../../plans/global/2026-10-02-jlz-native-joint-v4/GH-HANDOFF.md)을 따른다.

archive 33 member SHA/size, case schedule 2000행, CPU tokenizer 22 pack 및
CPU 회귀 6 tests를 확인했다. source/CPU 검산과 실제 GPU 검증은 별개다.
baseline 신규 실행0, 기존 중단 task 재개0. 새 runner만 독립 namespace에 구현했다.

예정 DAG: shared W0 → A/B BS2 commit·B2 entry → A/B B100 timing4·cold BS100×20 → CPU collector.
각 lane은 1GPU/8CPU/60416MiB, 동시에 최대2GPU다. 성능에 따른 분기·선택은 없다.

Local root: `/data/janghj/ODE-edit/local/jlz-native-joint-v4/20261002-compute-r1/`.
save_checkpoints=false; exact_resume=NOT_AVAILABLE. Raw/model/teacher/tensor/prompt/fullstdout는 Git0.
NO_BROADCAST_NOT_REQUIRED: 동일 서버에서 수행하며 소형 source/receipt만 게시한다.

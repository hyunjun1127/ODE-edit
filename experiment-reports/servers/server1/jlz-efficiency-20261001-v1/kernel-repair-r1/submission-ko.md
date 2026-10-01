# JLZ efficiency kernel repair 재제출

Instruction ODEEDIT-GH-SH1-JLZ-EFFICIENCY-KERNEL-REPAIR-20261001-R1.
본 문서는 재제출 기록이며 actual repair gate/전체 benchmark 완료가 아니다.

원56704 FAILED1:0,401 GPU초(program393.51547831110656초);
56705 CPU collector COMPLETED0:0. 원 source/raw/실패 및 UNQUALIFIED 보존.
원인: kernel name label reference/candidate와 timing 통계 동일 키 충돌.
수리: reference_kernel/candidate_kernel 필드, production assembly/선택 필터/collector 연결.
Owner CPU21회 및 독립 focused3회 PASS. 실제 GPU numerical PASS로 확대하지 않는다.

2026-10-01 22:17 KST에 **GPU56758 / CPU collector56759**를 held 검사 후 release했다.
GPU dependency=null, collector afterany:56758.
Admission 기존56684 한GPU + 신규 한GPU=2 (projectcap2/taskcap1).
이 확인은 resource-only이며 56684의 기존 pause/source/config/job은 유지했다.

| Identity | 값 |
|---|---|
| Source | 1d1e47b457838825605ad8850c5041857bf5e5a9 |
| Tree | 95360bf7c988434a0320c0d07d311a7670c0ede7 |
| Lock SHA256 | 7981db6043d9a46736ef274eed15fd609b402cae1cb6450f44117a2ab6a137fd |
| GPU | 1GPU/8CPU/131072MiB/12h/devbox/exportNONE/requeue0 |
| Collector | 0GPU/4CPU/16384MiB/2h |
| Parent contract SHA | 4485c11ffc9b0dbdf04dc5d672e284254fc15a37bc469271aa0166e8ffb48772 |

Run `/mnt/raid5/janghj/ODE-edit/local/jlz-efficiency/20261001-v1/attempt-kernel-r1`.
Execution WT `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-jlz-efficiency-kernel-repair-20261001-v1`.
원21 JSON size/SHA, source8 numerical module, input seal을 결속했다.
기존fixed32/short84/native결과 재사용(UNQUALIFIED 유지).
RAM entry/adj와 reference-returnR를 새coldW0/H0에서 재계산하고 REF12 trace exact를 요구한다.
새 native fit0; kernel12/probe/B1008/observer는 미완료 범위만 수행한다.
Kernel RAM timing은 원실패에서 미저장되어 재사용을 주장하지 않는다.

초기 인계 경계는 **kernels.json 저장 + B100 reference/candidate warmup2 PASS**다.
소형4요청 재확인만으로 repair 완료라 하지 않는다. 이후 pause, 프로그램은 자연 진행.
Checkpoint 없음, exact_resume=NOT_AVAILABLE, 신규 scientific chain0,
SERVER3 조회0, NO_BROADCAST_NOT_REQUIRED.

# Qwen official 12개 등록

2026-10-09 KST. 상태: RELEASED_RESOURCE_DEPENDENCY_PENDING.

- 실행 source: `d614add5e4c650821ed8d2503c071a1e02605ca8`; 제출 제어부 CPU range 표시 교정: `6f51777f` (frozen 실행 source 불변).
- GPU cap2. tuning61674 KEEP. 첫 baseline afterany:61674; 이후 GPU main → CPU archive → 다음 main 순차 afterok.
- FT/MEMIT/AlphaEdit/BLUE/FE/SPHERE × CF/zsRE; BLUE L2=1, 추가 grid0.
- 별도 GPU qualification/smoke/resume/native재평가: NOT_RUN_USER_DISABLED. CPU59 및 source157/imports0 PASS는 GPU PASS가 아님.
- 각 main GPU1/CPU8/59392MiB/48h 상한; 각 archive GPU0/CPU2/4096MiB/4h. exportNONE/Requeue0.
- 기존 W0/평가 일정/native hparams/20batch 및 checkpoint 보존 계약 유지. CF W0/W20 generation 완료가 archive 조건이며 다른 consumer가 남으면 KEEP.
- 독립 receiver fullSHA 검증 후 fresh writer/consumer 증거와 source 재검사를 거쳐 정확한 W20 payload만 정리. raw/metadata/logs KEEP. 실제 payload 전송·삭제0.
- 등록 중 Slurm NumCPUs=8-14 표시로 검사 중지 → 실제 CPUs/Task와 ReqTRES 검사로 control-only 교정. 61743을 보존하고 exact held prefix에서 계속, 중복/취소0.
- GPU 결과·실제 W&B run identity 아직 NOT_OBSERVED. GPU 완료 대기/recurring monitor/자동 재제출 없음.

| method | CF job | zsRE job |
|---|---:|---:|
| FT | 61743 | 61755 |
| MEMIT | 61745 | 61757 |
| ALPHAEDIT | 61747 | 61759 |
| ALPHAEDIT_BLUE | 61749 | 61761 |
| MEMIT_FE | 61751 | 61763 |
| SPHERE | 61753 | 61765 |

각 main 직후 CPU archive IDs: 61744, 61746, 61748, 61750, 61752, 61754, 61756, 61758, 61760, 61762, 61764, 61766.

현재 관측은 전부 PENDING. README는 GH가 이 실제 제출 영수증으로 통합하며 과거 완료 수치/미관측 수치를 새 결과로 쓰지 않습니다. 이전 준비/저장 blocker 보고는 이 등록 receipt로 supersede됩니다.

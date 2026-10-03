# JLZ v10 T′ 제출

Source: c2d5fb107a0435491d8b4705b43b75f6177bbb5c
Lock: cf03a94ccf873d93f36bdfe3c0a378998bc2340ffbb4cda4d5b18914c31cbe23

| 단계 | Job | Dependency | GPU/CPU/메모리 | wall 상한 |
|---|---:|---|---|---|
| A | 57699 | afterok:57698 | 1/8/59GiB | 7d |
| B | 57700 | afterok:57698; afterany:57699 | 1/8/59GiB | 7d |
| collector | 57701 | afterany:57698:57699:57700 | 0/8/24GiB | 4h |
| q1 | 57698 | 없음 | 1/8/59GiB | 24h |

전량 held owner/source/fullargv/GPU/CPU/memory/dependency/exportNONE/Requeue0 검사를 마친 후 정상 release했다. Q1 RUNNING, A/B/collector dependency pending을 최초 관측했다. GPU qualification/main 초기 및 완료 PASS는 아직 없다.

직전 attempt-v1은 로컬 admission 선행 검사에서 중단되었고 sbatch0/GPU0이다. 별도 ST4KE 작업은 경로로 분리하여 변경하지 않았다. 제출 helper가 없는 admission을 읽기 전에 lock을 만들던 순서만 고쳤다. 과학 source/계수/허용오차는 동일하고 attempt-v1 원본을 보존한다.

현재 production18 tests 및 추가 missing-admission regression1 PASS. Mock job 번호900…903은 CPU fixture만이며 실제 mapping은 위 표이다. 실제 총 GPUh/ETA는 미측정; wall을 예상 종료시간으로 해석하지 않는다.

원자료/실행경로: /data/janghj/ODE-edit/local/jlz-realized-subject-v10/20261003-v1/attempt-v2

Q1 source/config-bound READY 후 main A와 독립 cold B가 자연 진행한다. Q2는 A B1 자체이며 추가 B100fit0. main A B1→B2 초기 receipt 확인 후 능동 모니터링을 중단한다.

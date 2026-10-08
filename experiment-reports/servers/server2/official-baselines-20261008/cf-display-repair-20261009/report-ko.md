# Server2 CF 표시 transport 수리

권한: USER-SH1-GH-CF-DISPLAY-REPAIR-KEEP-HEALTHY-20261009-R1.
main7337967a 정본과 shared34e4d52d/5a942b26 수리를 읽고 own CF caller를 연결했다.
현재 단계: exact health/영향 pending 취소·CPU/실제 raw 검산·새 held 등록/검사/release 완료.

2026-10-09 05:17 KST: CF MEMIT61725는 실제4commit/accepted scalar/dropped0로 KEEP.
이는 미래 endpoint의 무오류 보장이 아니다. 완료 CF FT61650과 모든 zsRE도 KEEP.
W0_CF61723은 계산 완료지만 logging accepted0/rejected1/dropped1이며 온라인 정상으로 표시하지 않는다.
미시작 영향 CF61727/61729/61731/61733 및 collector61736만 exact owner/source/Command/state
대조 후 pending hold→downstream-first 취소했다. 취소직전 RUNNING이면 보존하는 guard를 적용했다.
zsRE61735는 취소된 CF61731 resource edge를 새 FE로 재연결하기 위한 pending control hold만 했다.
zsRE source/argv/과학 상태는 변경하지 않으며 등록 후 release한다.

own caller는 actual case strict NLL bits → request별 NumPy mean → cohort mean*100 → around(2)의
Efficacy/Generalization/Specificity_AlphaEdit_display를 동일 endpoint에 추가한다.
원 E/G/S/Score/Score_AlphaEdit_display와 공통 harmonic tolerance는 불변이다.
실제2000 W0 raw SHA e9265e1491f19c3fc3efb45cf79c483a5ad63c542028b9a09b56081a7084420b:
legacy mismatch 재현, repaired caller→shared schema PASS, 원 raw/summary 비변이.
이 CPU read-only 검사는 모델/forward/SDK/온라인 업로드0이다.

W0 producer47846468의 모델 payload/tokenizer/runtime/ordered stream/scorer source SHA가
새 consumer와 같음을 별도 compatibility에 봉인한다. producer raw/READY/identity는 수정하지 않는다.
새 CF chain이 원 W0 측정 summary를 수정된 scalar 형식으로 자신의 새 run에 전달하며,
원 failed online history를 overwrite하지 않고 W0-provenance.json에 original members를 남긴다.
추가 W0 GPU 관측·qualification·fit/generation은 없다. CF generation DEFERRED/CP KEEP 유지.

새 후보: ALPHAEDIT/ALPHAEDIT_BLUE/MEMIT_FE/SPHERE 4cold chains. healthy MEMIT lane 뒤 BLUE→SPHERE,
빈 lane AlphaEdit→FE→보존 zsRE SPHERE로 연결해 cap4 유지. 다른 두 zsRE lane은 불변이다.
CPU mixed-DAG 모든 도달 상태 폭≤4, own62 tests PASS. independent reviewer0/owner review.
기존원자료·CP·source·online/cost 모두 보존, README는 GH 단독 통합.

## 실제 제출 및 인계

실행 source/main `d31d582ffd425fe3a56a7d23b68d3c58e7e3d955`,
official tree `ba8f0e36d5551f3ec1d61e50c8996040a16cb6d3`, 구현 `6c38a3b4`.
manifest SHA `2753c82e21e59023dc5f031fea3fab77ddfeb76278ac013d61b1327f46a046a5`,
lock SHA `7d6395a08042cf649a658507de1b6e5d72d46008f1132d773eafe70ec1f4a323`.

| 새 CF job | 방법 | dependency |
| --- | --- | --- |
| 61778 | AlphaEdit | 없음, cap4 빈 lane |
| 61779 | BLUE | afterany61725(정상 MEMIT 유지) |
| 61780 | FE | afterany61778 |
| 61781 | SPHERE | afterany61779 |
| 61782 | CPU collector | 새4개+보존 MEMIT/zsRE7개 afterany |

zsRE61735는 exact pending owner/Command/WorkDir/source 확인 후 resource dependency만
afterany61780으로 변경하고 hold를 release했다. scientific argv/source와 다른 zsRE는 변경0.
취소5개는 sacct CANCELLED by1025/elapsed0/None assigned로 확인됐다.

각 GPU1/CPU6/59392MiB/48h, collectorGPU0/CPU6/24576MiB/4h, exportNONE/Requeue0,
node server2/QoS lab_gpu_s2. 실제 할당+pending DAG cap4/메모리·disk reserve를 검사했다.
2026-10-09 05:24:18 KST 최초 snapshot: 새5개 PENDING, 보존61725/26/28 RUNNING.
전량 held source/fullargv/input/resources/dependencies/checkpoint/tracking 검사 후 release했다.
원료 model/C0/P/token/scorer·공통 표시 helper는 봉인된 source/manifest에 결속했다.
새 온라인 startup/수리된 W0 metric remote delivery는 아직 NOT_OBSERVED이며,
SDK 접수나 CPU 통과를 remote PASS로 표기하지 않는다.

로그/실제원본 receipt 경로:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/registration-cf-display-r1/`.
CPU collector는 새 CF와 보존 source별 실제 raw/CP를 별도로 검산하고 logging 문제도 따로 기록한다.
raw/CP/old history 이동·삭제0, NO_BROADCAST_NOT_REQUIRED(compact Git만 게시).
GH에 `SH2-GH-CF-DISPLAY-SUBMITTED-20261009-R1`로 실제 ID와 scope를 직접 전달했다.
이후 GPU완료 대기/반복 monitor/자동재시도 없이 sealed runner/collector가 진행한다.

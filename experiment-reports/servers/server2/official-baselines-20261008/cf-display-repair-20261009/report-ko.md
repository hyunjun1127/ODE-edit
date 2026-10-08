# Server2 CF 표시 transport 수리

권한: USER-SH1-GH-CF-DISPLAY-REPAIR-KEEP-HEALTHY-20261009-R1.
main7337967a 정본과 shared34e4d52d/5a942b26 수리를 읽고 own CF caller를 연결했다.
현재 단계: exact health/영향 pending 취소·CPU 및 실제 raw 검산 완료, 새 등록 준비.

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

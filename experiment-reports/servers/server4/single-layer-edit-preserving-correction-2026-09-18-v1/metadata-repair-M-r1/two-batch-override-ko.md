# 최신 사용자 override: M B1·B2만 유지

사용자: “이거 batch 2개만 보고 지금 pending 중인 job들은 모두 취소시키자. batch 2개만 봐도 어느정도 나올 것 같아.”

2026-09-18T05:34:13Z(KST14:34:13), exact `50050_2`–`50050_9` 8개 PENDING child만
owner/name/state 검사 후 scheduler-side PENDING filter로 취소했다. 취소된8개 allocation elapsed=0.
`50050_0`(B1/ordinal[0,100))와 `50050_1`(B2/[100,200))는 RUNNING 그대로 유지했다.
다른 job 변경0, 새 제출0, 기존 자료 삭제·이동0. B3의 보존 native capsule도 그대로 둔다.

현재 실행범위는 **2 independent cold100 episodes / unique200 / 각8arm / 16finalL4 endpoints**다.
두 episode 모두 native REUSE이며 이후 신규 native fit0이다. 순차200-edit chain으로 부르지 않는다.
취소된 B3–B10은 NOT_RUN_USER_CANCELLED이며 과학 실패/0점 또는 완료 데이터로 집계하지 않는다.
취소된 episode의 자동 재제출·복구 권한도 없다.

원 M10/80endpoint 제출 report와 frozen source87f65ea2/lock635dd6e9는 과거 사실로 보존한다.
이번 사용자 override가 향후 범위만 축소하며 실행 중 두 프로그램의 방법·수치·저장 의무는 변경하지 않는다.
cap2, T=SKIPPED_USER_DIRECTED, full_numerical_validation=NOT_ESTABLISHED, S/R/L0을 유지한다.
본 취소는 actual M initial 통과 또는 두 batch 완료 보고가 아니다.
기존 대표 EN-F post-seal observer/reset 초기 경계 후 pause 규칙은 유지한다.

확인 receipt:
`/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/receipts/two-batch-override-r1/confirmed-state.json`
SHA `a212b280320fb6b5da67e929c554fb9b24e62cc2a35d1918ffcfdba3027c5dbd`.
Compact scope record: `runs/odeedit_single_layer_edit_preserving_correction_s4_20260918/two-batch-override-r1.json`.
GH direct 수신 완료. GH 중복 raw/GPU 검사 요청0.

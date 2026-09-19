# SH4 → GH: B1 r4 등록, FD-only 사용자 승인 적용

51058 / odeedit_slmf_B1r4_s4를 2026-09-20 03:37:37 KST에 등록·13항 held inspection·release했다. 03:38:17 accounting은 RUNNING, actual B1 성능은 아직 미관측이다. 본 task 1GPU/8CPU/60416MiB/exportNONE/Requeue0, time limit 7일(사용자 GPUh hardcap 아님), dependency 없음. Admission의 다른 project 작업은 빈 목록, project cap2 유지. 관측 free 43,642,580,992B이며 예약/독점 공간이 아니다.

실행 source 5f79085629b10b2bb8bdee88d017e18a46bb4c74 / tree a5b089229ab3e4edd2e3595c5c4f5226325b433e, archive cad46de07ea854f984b7d53b34c1a3fa59301ba7f37be50a6420b8839303b479 (489096336B), lock 6a14ebaf32549cc9479f2d112ba1954b06ef00380fffc1091cd70090f9098f61.

Local root: /data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/b1-fd-waiver-r4/. 원 51057 T0/실패 bytes는 보존하고 새 lock의 completed_T0_reuse와 FD_user_waiver로 연결했다. 새로운 T0 GPU 반복은 0. B1의 공유 native B100은 새로 1회 계산하며 보존된 technical 4-request native로 대체하지 않는다.

FD 원 기준은 미확립, full_numerical_validation=NOT_ESTABLISHED. 사용자 “통과할테니 task 이어서 진행해”에 의해 B1 진행만 허용한다. Geometry/Current/reference/Armijo 등 방법 수용 조건은 불변이다. CPU 227 tests PASS. 과거 기술/실패 parent allocation 합계 1113 GPU-sec와 신규 51058 비용은 분리한다.

max_batches=1, sequential_authorized=false, checkpoint_saved=false, exact_resume=NOT_AVAILABLE. B1 완료 및 CPU 검산/보고까지 monitoring_active=true, 다른 task 재개/취소 없음. Admission의 old_job_cancelled 필드는 역사적 50983 취소를 재결속한 것이며 이번 turn 신규 cancel이 아니다. B1 완료 후 STOP, S3/S10은 사용자 별도 허가 전 금지한다.

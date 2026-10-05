# CD_Q/CD_C 신규 실행 결속

Nonce `ODEEDIT-USER-GH-SH4-CD-CUMULATIVE-2K-20261005-R1`, 정본 source `99a62de2`를 따른다. 기존 V13/V14 source와 raw는 수정하지 않는다.

새 namespace의 frozen `S/C/J`를 batch당 한 번 만들며, `Delta=double(W_entry)-double(W_initial)`를 사용한다. Fit에는 전체 `Y=D*S`를 주입하고 모든 owner의 SUM adjoint를 모은다. Norm과 allocation gradient는 각각 한 번 더한다. Common candidate 0–24에서 모든 native Jr가 .05 미만일 때만 멈추며 terminal backward는 없다.

λ는 main B1 최초 비영 projected-action 후보에서 한 번 계산한다. 두 arm은 scalar/hash 영수증만 공유한다. 각각 독립 cold W0/H0에서 20개 fresh fit/commit을 수행한다. Commit은 원 CD writer의 실제 하층 write 이후 upper key를 재측정하며 rewrite-only native H를 한 번 누적한다.

현재 receiver는 검증된 `full-evaluate-replay` reference 경로를 사용한다. 후보 native no-grad 관측과 continuing 후보 backward replay의 실제 호출/시간을 따로 기록한다. 광범위 최적화나 별도 full-B calibration fit은 수행하지 않는다.

서버4 resource override는 GPU1/CPU8/default59392MiB/hard60416MiB/exportNONE/Requeue0/유한48h이다. 정본의 121856MiB 및 원문 bytes는 바꾸지 않는다. 두 arm은 upfront 등록한다. 제출 당시 한 슬롯만 가용하면 CD_Q 이후 CD_C를 resource/scalar dependency로 직렬화한다. 과학 지표 PASS는 dependency가 아니다.

NoCP/exact resume NOT_AVAILABLE. Source와 compact 영수증만 Git, 원시 metric/JSONL/stdout는 ignored local에 유지한다. 불필요한 대형 raw broadcast는 하지 않고 `NO_BROADCAST_NOT_REQUIRED`로 기록한다. 초기 handoff 뒤 agent monitoring/automatic resume를 중단하며 sealed runner/collector는 W20까지 진행한다.

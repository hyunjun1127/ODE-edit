# CD_Q/CD_C sequential 2k 실행 영수증

Nonce: `ODEEDIT-USER-GH-SH4-CD-CUMULATIVE-2K-20261005-R1`.
Task: `jlz-cd-cumulative-allocation-bs100x20-s4-20261005-r1`.
정본: `plans/global/2026-10-05-jlz-cd-cumulative-allocation/`.

현재 단계는 구현/CPU·source preflight이며 제출 및 실제 GPU qualification은 아직 관측하지 않았다. 각 arm은 first2000 BS100×20, cold W0/H0, L4–L8, budget .75, max25평가/24update다. 기존 V13/V14 실험은 변경하지 않는다.

원 CPU11/design `PASS_WITH_WARNINGS`는 합성 설계 근거로 재사용한다. 새 production CPU 회귀와 source cross-review는 별도 기록한다. 실제 target GPU 검산은 sealed job에서 main 외 2개 native 요청의 same-candidate check만 수행하며 fit/update는 없다. 기술 READY 뒤 main B1부터 cold chain을 진행한다.

산출물 예정 경로: `/data/janghj/ODE-edit/local/jlz-cd-cumulative-allocation/20261005-r1/attempt-r1/`.
Raw/tensor/model/prompt/fullstdout는 local KEEP, checkpoint/복원 bundle은 저장하지 않는다. `exact_resume=NOT_AVAILABLE`.
Artifact broadcast: `NO_BROADCAST_NOT_REQUIRED`; 이 단계는 source/compact receipt만 공유하며 대형 raw는 전송하지 않는다.

최종 W20 지표/분모는 아직 `NOT_OBSERVED`다. CPU collector가 각각 20commit/19join/100H 및 R2000/P4000/N20000의 실제 저장 raw를 확인한 뒤 complete/partial/technical-blocked를 구분해 사실 보고서를 작성한다.

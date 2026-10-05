# Causal Allocation Editing 구현 결속

정본 수학은 변경하지 않았다. 새 namespace에서 native mean-key ridge의 actual all-token NLL/KL과 full causal task/Q adjoint를 구현했다. Raw A 및 Aᵀ를 유지하고, G는 prox에서만 처리한다. 이전 EfficiencyAdam/shared budget/subject-only loss는 실행하지 않는다.

원 native KL의 전체 유효 토큰을 새 task-local entry capture에 보존한다. 요청 단위 원 forward shape, context/token/lookup과 FP32 anchor 순서를 유지한다. Native reference는 반환 delta를 그대로 R로 평가하며, 가격 산술의 radial 좌표는 FP64 R/a다. 추가 scientific fit나 reference commit은 없다.

검증 job 뒤 본선은 `afterany`로 연결하되 모델 로드 전에 exact source/config READY를 검사한다. 검증 실패 시 본선은 typed failure로 종료하여 afterany CPU collector가 실패도 수집한다. `afterok` 미충족으로 영구 pending되는 경로를 피하기 위한 failure-coverage 구현이며, 낮은 품질을 통과 조건으로 사용하지 않는다.

BB1/Armijo 후보의 forward와 accepted backward를 분리했다. 최대 50 logical evaluations, 24 accepted updates, 25 full gradients이며 마지막 accepted gradient를 terminal mapping에 재사용한다. 현재·trial 외 불필요한 초기 payload 보유를 제거했다. Batch마다 BB/entry teacher/anchor/factor를 새로 만들고 가격만 고정한다.

Source/CPU 검사와 실제 GPU qualification은 별도 상태다. 현재 실제 양수 λ, 실제 peak/속도 및 W20은 미확립이다. GPU 검증은 두 native 요청의 한 고정 후보에서만 수행하며 별도 full-B100 fit는 없다. 본선은 fresh W0/H0 20배치이며 noCP/noB21이다.

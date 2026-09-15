# SL-ZFlow SH2 M0 / pre-GPU 구현 경계

- instruction_id: ODEEDIT-S06-SINGLE-LAYER-ZFLOW-SEQ1000-SH2-V1
- reference publication: 69b467d380715f236fad81141d00b53723e01ba7
- reference tree: c0ae949fd666d20b08c2137bc373e78cffbfa244
- original source-input-manifest SHA256: e0f64857e070e4e2123c57adc77e757225d1d36eba86c58a18d8718ce454e423
- original16 및 inherited12 full size/SHA 대조 PASS. Dispatch/latest user/detailed proposal/pipeline/review/config/CPU checks/demo/source/tests FULL_READ.
- 별도 worktree: /mnt/raid5/janghj/.codex/worktrees/odeeditsh2-single-layer-zflow-seq1000-v1
- local output: /mnt/raid5/janghj/ODE-edit/local/single-layer-zflow/20260916-v1/

기존 canonical root 및 다른 task/source/jobs는 보존했다. Source/input freeze는 실제 새 commit에서 생성하며 reference CPU SHA와 혼동하지 않는다. 기존 CPU30/demo25 oracle18accept6reject는 과거 reference 결과이고 실제 Llama 결과가 아니다.

새 native binding/adapter/durable/runner 및 집중 CPU 검사를 작성했다. Native group key(clean1/2, generated각1/10)와 edit context(각1/6)를 구분했다. 실제 Llama weights는 FP32, nonsymmetric N은 LU로 처리한다. Prefix/teacher는 고정하며 suffix는 fresh, full-vocabulary head는 필요한 위치에만 적용한다. 원 core/in-memory transaction은 수정하지 않았다. Production config가 off/null, on/positive-budget를 fail-close 검증한다.

N4는 job38997의 동일 first1000/order/source/context/config를 우선 비교 후보로 결속했다. 기존 W10 998/1000,1943/2000,8072/10000은 신규 결과가 아니다. 문항별 final raw는 정확한 path/size/SHA를 owner에게 요청했으며 수신 전 paired 결과를 주장하지 않는다. 신규 N4 chain은 아직 없다. N4 TF32 cudnn=True와 새 runtime=False 차이를 기록하며 수치 동등성을 주장하지 않는다.

자원은 project cap2, 1GPU/8CPU/60416M/exportNONE/no-requeue다. 실제 admission 직전 재검사한다. 전체 repository memory audit의 과거 Server4 sbatch6개 초과는 이번 scope 밖이므로 변경하지 않는다. 이번 run.sbatch를 독립 검사한다. GPU-hour hard cap은 null; 실제 기술 batch 비용으로 MAIN 계획을 봉인한다.

이 문서는 M0/pre-GPU 사실 기록이며 actual 8B parity/resume/SEQ1000 PASS가 아니다. 이 task에 한해 초기 gate 이후 계속 진행하는 사용자 override를 적용하며, 다른 task의 monitoring pause는 그대로 유지한다. scientific_promotion=false. Raw broadcast는 승인된 필요한 N4 한정 allowlist 외 NO_BROADCAST_NOT_REQUIRED.

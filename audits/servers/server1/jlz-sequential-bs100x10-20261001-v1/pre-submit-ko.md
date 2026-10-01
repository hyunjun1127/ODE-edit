# JLZ pre-submit CPU/source 검토

Instruction: ODEEDIT-GH-SH1-JLZ-BS100X10-20261001-R1.
독립 reviewer `/root/jlz_red`를 명시 사용자 요청에 따라 사용했다. reviewer는 코드/과학/Slurm/GPU/원격/Git을 변경하지 않았다.

최종 판정: **PRE_SUBMIT_CPU_SOURCE_PASS_WITH_WARNINGS**.

- native KL 방향, global active-request SUM, C0 FP32/count→FP64, adj/FP32 materialization 식 검토.
- 120 최종call 예약, rejected candidate 배제, 모든 NONFINITE fatal 검토.
- 원자 W/H/RNG/context/ledger 및 중간 history/observer/publish 예외 rollback 검토.
- 첫 검토 BLOCK: ledger rename 후 fsync 실패 ambiguity, collector 분모/label/NLL/TF 확인 부족.
  `reconcile_ledger`와 candidate exclusion, collector terminal과 ledger 대조, 전체 expected denominator/NLL/TF 교차집계로 수정했다.
- owner old21+신규·상속 포함39실행 PASS. reviewer test_core 독립18실행 PASS(상속된 pilot tests 포함).
- reviewer의 실패 preflight/science 비용분류 WARN은 `science_started`로 별도수리했다.
- capture/legacy/observer token work 전체는 NOT_SEPARATED; 신형 oracle token work/호출/단계 시간과 분리해서 남긴다.
- bind C0 count/dtype/shape/finite, fixed1000·10cell hash, noCP·storage 검사 경계 확인.

실제 BS100 모델수치/Slurm held argv·admission/endpoint는 이 CPU 검토에서 NOT_OBSERVED.
수치 허용치/방법/표본/solver 예산 변경0. postrun 독립 reviewer는 사용자 완료 recall 때 수행하며 자동 collector로 대신하지 않는다.

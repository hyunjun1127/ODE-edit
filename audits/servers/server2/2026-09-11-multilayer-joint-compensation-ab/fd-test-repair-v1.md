# B 초기 FD test-only repair

승인: `ODEEDIT-GH-SH2-B-FD-TEST-REPAIR-20260912-R1`.
이전 source `d7699572d7172ffeb14bc3809e72a5b70cff19a2`, job44991은 FAILED1:0,
98 allocated GPU-seconds, B correction/endpoint0으로 보존한다.
기존 실패는 NUMERICAL_VALIDATION_UNRESOLVED이며 비선형 probe 오차는 가설이다.

변경은 초기 test harness, exact teacher 재사용 binding 및 duplicate-job admission에 한정한다.
기존 eps1/.5/.25 뒤에 .125/.0625/.03125/.015625를 사전 고정해 추가한다.
첫 요청·모든 context·평균 NLL·projected-gradient 방향·.001 WN-norm 스케일·AD식은 동일하다.
원 teacher SHA `1bbf237b12e05a14bf948c46fcd0d50953f62394650d6f42b59ce0241ce646de`를 검증하고 재사용한다.
44991이 direction tensor hash를 저장하지 않았으므로 소급 byte identity는 주장하지 않는다.
이번에는 WN/direction hash와 각 부호 FP32 perturbation norm/rounding norm/nonzero fraction을 기록한다.

허용오차는 그대로 `0.02 max(|AD|,|FD|) + 64 eps32 max(|L+|,|L-|,1)/h`.
신호 판정은 |AD|·|raw FD|가 기존 roundoff보다 크고,
양쪽 actual perturbation이 nonzero이며 그 norm이 rounding-error norm보다 큰 것이다.
두 인접 raw FD가 신호/허용오차 모두 만족해야 한다. Extrapolation gate 영향0.
Grid 소진 시 HOLD; 추가 eps 탐색0. 기존 GGN/physical parity/restore 검사는 유지한다.

검증: FD/actual-callable test/Slurm admission CPU10 tests PASS (1.091s),
compile/bash PASS, memory audit160 checked/failure0.
functional/PCG/elastic/protocol/native-adapter/history/banks의 이전 source 대비 diff0.
기존 test-only synthetic cubic 및 wrong-AD/rounding/non-adjacent rejection을 검증했다.
GPU PASS는 아직 아니다. Retry는 한 번만 허용되며 initial-valid 뒤 agent는 중지한다.

로컬 승인·grid lock:
`local/multilayer-joint-compensation/20260911-v1/track_b/control-v1/fd-repair-instruction-20260912.md`
`local/multilayer-joint-compensation/20260911-v1/track_b/control-v1/fd-test-repair.lock.json`
과학적 승격0, 다른 task/job 변경0, main 통합0.

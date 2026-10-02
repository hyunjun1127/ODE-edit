# JLZ v9 구현과 실행 결속

SH4는 사용자 nonce `ODEEDIT-USER-GH-SH4-JLZ-V9-RIDGE-500-EXACT-PILOT-20261003-R1`에 따라 새 namespace를 구현했다. 정본 authority는 `e147e55139d9a9c1a568a812020633b579c0b767`이다. 원 sealed 문서와 이전 실험은 변경하지 않았다.

## 구현 범위

Native BLUE `311b076a92e4ed0f14f5c8b4909732da781bc5f7`의 FP32 context group mean, group stack mean, CPU FP32 H 의미를 사용한다. 기존 source에서 prompt packing, Llama adapter의 순수 forward, normalized C0 loader, observer tokenization만 명시 재사용했다. v5 writer, v7 fullcontext Gram, replay, raw D Adam은 새 실행 경로에 없다.

새 writer는 whole B mean key와 실제 하층 write로 상층 key를 다시 만들며 D, P, input, solve의 gradient를 보존한다. q Adam LR .1, 25후보와 24업데이트, 물리 D 사후 clamp를 사용한다. Arm 차이는 단일 .1 merged geometry의 층 합과 전체 root 집계뿐이다.

Raw A는 CPU FP64로 보존하고 A 곱의 P adjoint도 계산한다. GPU factor를 재사용하되 원 A residual이 1e-8을 넘으면 같은 A의 LU를 한 번 사용한다. A/H symmetrization, jitter, 과학 허용치 변경은 없다. 실제 비용은 별도로 기록한다.

Q2는 A main B1의 동일 fit 안에서 수행한다. Terminal D와 ridge RAM weight를 봉인한 뒤 frozen operator와 독립 causal exact shadow를 구분한다. Exact upper key는 frozen upper 실패를 상속하지 않는다. Exact unsupported는 ridge 계산을 중단하지 않지만 복원 실패는 기술 실패다. finally에서 W/H/RNG/nonselected를 복원한 뒤 동일 ridge weight를 commit하고 H를 한 번 갱신한다.

## 검산과 한계

Owner CPU 13 tests는 mean order, dense 및 direct D/P/input gradient, q bridge, zero root, 25후보와 24업데이트, native H, rollback, Q2 cache 및 upper 판정 분리, partial collector를 포함한다. 실제 Llama GPU 검증은 Q1에서 별도로 수행한다. 독립 reviewer는 사용하지 않았으며 independent PASS를 주장하지 않는다.

정본 v8/v9 CPU 결과 JSON은 모든 field를 파싱했고 보관된 증거로만 연결했다. 이 작업의 실제 GPU 검증으로 간주하지 않는다. 준비 중 이전 config 경로의 날짜 오타를 CPU 단계에서 바로잡았으며 GPU 실행이나 자료 변경은 없었다.

## 자원과 진행

두 Q1이 모두 같은 source/config의 READY를 만들면 두 독립 cold main이 진행한다. Collector는 4개 GPU job afterany로 실패도 수집한다. 최대 동시 2 GPU, 각 8 CPU와 host 60416 MiB, collector GPU0과 24576 MiB다. Host 계획 56.55 GiB, GPU 계획 81.86 GiB이며 실제 peak는 아직 미측정이다. Disk reserve 30 GiB를 검사한다. Q1 wall 24h, main wall 7일, collector 4h는 ETA가 아니다.

새 full W0 평가와 baseline은 없다. 기존 W0 first500 N 4392/5000은 token/input 결속 후 역사 reference로만 재사용하며 layout bitwise parity는 주장하지 않는다. 각 main W5의 R500/P1000/N5000 후 종료하며 B6는 준비하지 않는다.

Entry 및 terminal의 M/planned D/realized Y만 승인된 local telemetry 예외로 저장한다. W/H/P/K/optimizer 및 복원 bundle은 저장하지 않는다. Raw/tensor/prompt/fullstdout는 Git에 넣지 않는다.

대표 main B1 commit과 B2 자기 entry 또는 정식 main resource pending 확인 후 agent monitoring과 automatic resume를 중지한다. Q1 PASS만으로 main 초기 완료라고 쓰지 않는다.

## 게시 경계

원 root의 무관 dirty 1390개는 보존한다. 전용 worktree와 명시된 own scope만 commit한다. Generic access helper가 새 project namespace를 기본 차단하면 해당 결과를 보존하고 이번 envelope의 exact path 승인을 좁은 예외로 기록한다. NO_BROADCAST_NOT_REQUIRED는 같은 서버 실행 및 local raw 보존 범위에 따른다.

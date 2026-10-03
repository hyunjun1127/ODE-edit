# JLZ v10 T′ 구현·제출 전 검토

상태: IMPLEMENTING_NOT_SUBMITTED. CPU 18 regression 및 정본 두 proof PASS는 실제 Llama qualification이 아니다. 사용자 nonce `ODEEDIT-USER-GH-SH3-JLZ-V10-TPRIME-500-20261003-R1`, authority `7ff7b51e3aec293d462681875a4d8b1704f3d2d1`의 v10 전용 예외를 적용한다. 기존 STOP task는 재개하지 않는다. Owner 직접 검토이며 독립 reviewer는 사용하지 않았다.

## 실제 source·입력 결속

새 namespace `project/run_scripts/jlz_realized_subject/`의 adapter를 실행한다. V9 b8c4c96의 native prompt/readonly Llama layout/covariance loader 일부를 재사용하되 v10 Tprime builder와 adjoint·optimizer는 새 경로다. BLUE311b076의 compute_z/compute_ks/seq/repr는 native 의미 참조 SHA로 결속한다. BLUE writer나 과거 V9 runner를 호출해 v10 실행이라고 하지 않는다.

S3 Python3.12.3/torch2.9.1+cu128/transformers4.44.2, eager FP32, autocast/matmulTF32/cuDNNTF32 off. 설계 참조 transformers4.57.1과 다르다. 4.44.2 decoder tuple/attention 반환을 adapter에서 처리하며 actual Q1으로 검증한다. 모델 revision8afb486c, L4–L8, C0 5개 및 snapshot closure를 기존 S3 실물에서 SHA/size 결속했다. 모델·C0 전송/재생성0.

정본18개 SHA/size 및 CSV 원 CRLF를 보존했다. 승인 archive74785B를 S1에서 1회 pull하고 exact18regular를 검산했다. first500 case-ID SHA `0be7d88c759e7f65a690514035f40a18c5c19d8591ddd33f97b2fa6187b64a95`; 실제 tokenizer의 5개 batch native packed identity도 정본 reference와 일치했다. W0 기존 MEMIT-HJ first5006500RPN scalar rows를 재사용한다. 원 token-ID는 미저장이므로 prompt/target identity 및 count를 대조해 현재 token identity를 새 결속했으며 bitwise cross-host/MB 인증은 아니다. Cold actual 모델 5개 W SHA를 실행 시 원 W0와 대조한다. W0 원 cuDNN TF32 true와 이번 false를 명시한다. 새 W0 전체10k 평가0.

## 구현 요구와 검증

| 요구 | 실제 경로 | CPU/실행 증거 |
|---|---|---|
| whole-B RW nested FP32 key / KL 하층 경로 / upper K,P total gradient | causal_builder, geometry | dense-vs-staged, reversed MB1, KL row, stopped-geometry negative control; Q1 실제 반복 |
| Tprime linear(k,Weff)-linear(k,Wentry), 같은 v의 fit/norm, 두 input VJP | physical_linear, causal_builder, profile, subject | 직접 FP32 weight adjoint를 dense reference와 검산; full-block native hook Q1 |
| NLL 1/6, norm .5, allocation .1, KL .0625, B SUM 1회 | subject, optimize, allocation | component 합/zero subgradient/25후보24update/terminal backward tests |
| 첫층만 cache / upper layer 매후보 갱신 / G-only allocation | geometry, causal_builder | 전층 solve residual1e-8; same-A LU fallback, jitter/pinv0 |
| evaluated Weff exact commit / terminal actual RW Gram once | terminal, writer | actual key/write probe Q1; H once/duplicate commit/rollback/연속2batch fixture |
| RAM W/H/RNG/context/ledger transaction / observer 비변이 | writer, run, observe | IO 오류 주입 복구, context 복구, nonselected guard, runtime aux hashes |
| Q1 cold A/B BS2x2 및 B1/B3 shape / Q2=mainA B1 / main각5batch | run | 명시 routing, input/25·24 counter receipt; B6 경로 없음 |
| R/P/N preference 및 TF micro/macro/strict, NLL, active, paired | observe, collect | ties/N direction/token identity/분모 CPU 검사; W5 collector 동일 raw 사용 |
| source/config/input freeze / cap1 held inspection / upfront DAG | freeze, submit | mocked 실제 argv와 모든 held검사→release 순서, 중복 호출 차단, 실패 collector |
| noCP | 모든 새 source | torch.save/state payload 없음; scalar/hash/token 평가만 JSON; RAM activation checkpoint는 디스크 CP 아님 |

초기 CPU negative-control fixture는 무작위 tiny Llama key와 λ15000 조합에서 차이가 FP32에서 소실됐다. Synthetic fixture prior만 .001로 바꾸어 경로 검출을 확인했다. Production λ15000 및 과학 허용오차는 변경하지 않았다. 최초 실패 log도 local 보존한다. 이후 18 tests PASS. 정본 proof2개는 scratch copy로 실행하여 원 result bytes를 유지했다.

## 자원·실행 계획

1GPU/8CPU/host60416MiB cap1. Q1≤24h, main각7days, collector0GPU/8CPU/24576MiB/4h. 이는 wall 상한이며 실측 GPUh/ETA가 아니다. GPU 자원이 즉시 있지 않으면 정상 dependency/resource pending이다.

FP32 model 약29.9GiB GPU, CPU H 약3.83GiB와 rollback H 약3.83GiB, CPU FP64 A 약7.66GiB, GPU factors 약7.66GiB, editable entry/rollback/terminal W 및 cache를 포함한다. Host peak 계획46GiB/59GiB, GPU peak100GiB/H200143771MiB, cold model CPU load transient는 H 생성 이전이다. 실제 Q1 및 B100 peak를 별도 기록하며 예측을 측정으로 표시하지 않는다. Disk output2GiB + 여유20GiB를 예약 계획; model/reference 복제0, CP0. 최초 관측 free240775438336B는 비독점값이므로 최종 admission에서 다시 확인한다.

Q1 → A afterok(Q1) → B afterok(Q1)+afterany(A), collector afterany(Q1,A,B). 세 GPU job은 최대동시1. Q1 기술실패는 main의 readiness를 fail-closed하며 collector는 coverage를 분리한다. B는 A 품질로 선택하지 않는다. Main각cold W0/H0이며 Q1 state 전달0. 전체 main250후보240update, Q1추가100후보96update. Fixed-candidate qualification·component backward·terminal 실제 forward/energy·observer·IO 비용을 별도 기록하고 무료로 세지 않는다.

현재 실제GPU PASS/초기PASS/제출은 미확인이다. 정상등록 후 Q1 및 main A B1 observer→B2 entry까지만 관찰하거나 전량 released resource-pending을 확인해 인계한다. 이후 자동모니터/재제출/다른task재개0.

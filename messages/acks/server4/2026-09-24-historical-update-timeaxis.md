# SH4 수신 및 구현 ACK

Nonce: `GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1`.

실제 server4 / session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` / repository `hyunjun1127/ODE-edit`를 결속했다. 별도 clean worktree는 `/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/worktree`다. main73d76459의 실행 envelope를 적용하며 이전 E3 pause를 상속하지 않는다.

승인 소형 3파일·29 regular member / 5,238,704B를 receiver full SHA/size 검증했다. 압축의 정확 설계-root 접두사만 제거하고 traversal/link/device/중복/미승인 member를 거부했다. 원 bytes는 그대로 보존한다. Receipt SHA `1ff354e07cc2f76588a1b831e1469be12530bf1f03e01e13f0f86a15ff34961e`.

T0는 CP24/선택 tensor120/model shard4/W0 선택 weight/고정10k/토큰을 새로 검증했다. Runtime-binding SHA `717808cb2460b034d7f526bb14cf325e0ad817f678d6fa7cd98844649f051e41`; CPU188.255초·process peak2125.289MiB. GPU/model forward0. CP 신규전송0.

초기 상태 `IMPLEMENTING_NOT_SUBMITTED`, job_ids=[]였다. 실제 이후 제출은 별도 submission/release receipt로 구분한다. Whole U·FP64 subtraction·FP32 one-cast·MB16·기존 evaluator 그대로, project/task cap2/각1GPU·8CPU·60416MiB, 신규 checkpoint0. T0→T4 완료 또는 technical blockage까지 계속한다.

GH direct 수신 ACK도 같은 nonce로 회수했으며, GH 감사/ACK를 단계 선행조건으로 사용하지 않는다. NO_BROADCAST_NOT_REQUIRED.

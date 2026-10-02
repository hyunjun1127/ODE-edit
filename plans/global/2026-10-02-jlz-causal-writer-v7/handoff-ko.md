# GH와 SH4를 위한 JLZ v7 변경 요약

2026-10-02. 사용자가 SH4 작업을 직접 중단했다고 알렸고, GH가 v7 실험을 SH4에 맡기도록 명시적으로 지시했다. 새 v7 구현·pilot·500-edit 실험 실행을 위한 전달문이다.

> 내가 임의로 sh4의 작업 중단했는데, GH에게 이 정보까지 전달해서 v7 실험 진행하도록 하자. sh4에게 task 진행시키자.

[사용자 실행 명령](execution-command.json)의 instruction ID는 `ODEEDIT-USER-GH-SH4-JLZ-V7-CAUSAL-500-20261002-R1`다. 기존 SH4/v6 작업은 USER 중단으로 기록하고 그대로 재개하지 않는다. 세션 중단과 scheduler job 취소는 다르므로, 실제 v6 job 존재와 상태를 확인해 그 task에 한정하여 중단 상태를 정합화한다. 관련 없는 job은 변경하지 않는다.

정본: [method](method-ko.md), [구현 계약](implementation-ko.md), [contract](contract.json), [TeX](../../../docs/methods/jlz-causal-writer-v7.tex), [실험 설계](experiment-500/experiment-ko.md), [실험 계약](experiment-500/experiment.json), [CPU 검증](math/README.md).

담당 경로는 기존 사용자 지정인 GH→SH4/server4다. GH `01a04939-8873-7673-8dca-4c7fc5e31af0`, SH4 `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, repository `hyunjun1127/ODE-edit`, server4 `/data/janghj/ODE-edit`. 기존 두 arm·각BS100×5·warmup0·native LR .1·W5 종료·noCP·신규baseline0 범위는 유지한다.

교체되는 핵심은 다섯 가지다.

1. 모든 후보에서 실제 current RW 입력을 전체-B layer barrier로 모아 current key→P→U를 구성한다. 첫층 P만 재사용한다.
2. Dynamic solve는 key를 detach하지 않으며 actual linear는 D/P/input gradient 모두 반환한다. 정책 G/E의 P/K/T gradient도 유지한다.
3. Builder에는 subject δ hook이 없다. Native 주경로는 기존 subject δ 공동 학습이고 teacher만 stop-gradient한다.
4. Candidate25도 같은 causal writer로 구성한 weights를 전체 actual forward로 검증하고 그대로 commit한다. 최종 모델 key와 builder key를 비교한다.
5. 비용 원장을 새로 측정한다. 상층4개×25 solve 및 매후보 key builder가 필요하므로 v6 ETA·40% 목표를 복사하지 않는다.

A/B 차이는 norm aggregation뿐이다. Native 목표·입력·clamp·memory·계수·probe 시점을 바꾸지 않는다. Absolute virtual z residual 보상이나 층별 독립 최적화를 추가하지 않는다.

새 코드 namespace는 `project/run_scripts/jlz_causal_writer/`, 새 task/출력은 `jlz-causal-writer-v7-bs100x5-20261002-v1`로 구분한다. 기존 v6 구현/queue/job 상태를 먼저 기록하며, 그 산출물과 v7 결과를 같은 attempt로 합치지 않는다. V7 main은 fresh W0/H0로 시작한다. 문서에 적힌 기존 실행 권한과 실제 제출/취소/수신 상태는 별도 receipt로 구분한다. 이번 사용자 지시로 v7 구현·검증·실험을 진행한다. 설계 작성 당시 NOT_RUN 상태는 과거 기록이며 현재 실행 금지가 아니다. 실제 job 확인·정리·제출 상태는 별도 receipt로 기록한다.

Small pilot은 A/B 각각BS2×2이며 native만 사용한다. 이어 새 B100 builder 경로를 한 번, 후보5/Adam4까지 점검하고 후보5 actual probe backward 후 step 없이 종료한다. Technical parity와 자원 한도를 확인한 실제 실행 source를 기록한다. PS/NS·층 집중·clamp 비율을 통과선으로 삼지 않는다.

CPU 증거는 작은3층 surrogate의 대수와 전체 gradient 검증이다. 실제 모델·FP32 cast·checkpoint·처리량 검증을 대체하지 않는다. Source/문서SHA·입력 identity·pilot/GPU검증·job ID·500완료 상태를 각각 보고한다.

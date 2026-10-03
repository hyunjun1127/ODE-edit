# GH 전달: JLZ v10 T′를 SH3에 배정

2026-10-03. Instruction/nonce: **ODEEDIT-USER-GH-SH3-JLZ-V10-TPRIME-500-20261003-R1**.

사용자 원문:

> GH에게 이 method 설계와 실험 설계 전달해. SH4이 아닌 SH3에게 task 진행시키자.

GH는 [실행 명령](execution-command.json)을 근거로 **SH3/server3**에 구현→qualification/pilot→A/B 각500 main을 배정하고 직접 담당 수락 ACK를 회수한다. 대상은 GH `01a04939-8873-7673-8dca-4c7fc5e31af0`, SH3 `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`이다. SH4에 새 task를 배정하거나 기존 SH4 job을 취소하지 않는다.

반드시 전달할 정본:

- [Method](method-ko.md), [method 계약](contract-draft.json), [수정·결정](changes-and-decisions-ko.md)
- [실험 설계](experiment-500/experiment-ko.md), [실험 계약](experiment-500/experiment.json)
- [입력 identity](experiment-500/input-reference.json), [first500 일정](experiment-500/case-schedule-first500.csv)
- [Baseline 비교 정책](comparison-ko.md)
- CPU 대수·T′ adjoint 검산 코드와 결과. `extract_strength_reference.py`는 역사적 v9 값의 추출기일 뿐 실행 prerequisite가 아니다.

핵심 변경은 **actual key로 만든 v^a의 subject 주입**이다. 매 후보 actual lower all-token write→상층 key→whole-B ridge→v^a와 full K/P gradient를 유지한다. 기존 V9의 R 직접 주입이나 T의 Uk^s 주입을 그대로 사용하지 않는다. 기존 builder에 빠진 KL rows도 actual 하층 경로로 처리하되 ridge key 평균과 history에는 넣지 않는다. v 경계에서 adjoint를 누적하여 builder에 반환한다.

Norm은 actual v의 native key-group 가중 평균, λ_n=.5 고정이다. NLL은 기존 context 균등 평균이다. Allocation은 G-only, A=sum(c_l), B=sqrt(sum(c_l²)), λ_alloc=.1이다. E/pulse/replay/active clamp/warm-up은 없다. 마지막에 평가한 U 그대로 commit/H 한 번 갱신한다.

**철회 확정:** B1 계수3점×A/B 보정6 fit 없음, v9 S_mean≈.185 matching 없음. Native 기본값의 BLUE/MEMIT-H/AlphaEdit가 주 비교다. V9 값은 참고다. 다른 계수·layer subset·추가 reference·history replay를 도입하지 않는다.

기존 순서first500, 독립 cold A/B 각각B100×5,25후보24갱신, 전체L4–L8, FP32 model/FP64 geometry, noB6/noCP/신규baseline0/exactprobe0이다. 작은pilot은 main 밖 case541/6693/16935/17306, A/B각BS2×2다. Whole-B100 검증은 main A B1 자체에 포함하여 별도 B100 fit을 추가하지 않는다. Technical PASS 뒤 stage마다 새 사용자 승인을 기다리지 않는다.

새 namespace `project/run_scripts/jlz_realized_subject/`와 전용 non-main worktree를 사용한다. V9 production reference는 commit `b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66`이며 read-only 참고다. Shared runtime·input·native writer·evaluator는 SH3에서 실제 경로와 hash를 재결속한다. Task cap1 및 더 엄격한 현행 cap을 준수한다. 기존 task나 중단 job을 재개하지 않는다.

이번 사용자 명령은 이 신규 method 실행 범위의 명시 권한이다. 문서의 NOT_RUN/production未검증은 현재 사실을 나타내며 구현 금지가 아니다. GH는 정본 검증과 main publication 및 SH3 직접 수락을 진행하고, 수락/구현/GPU검증/제출을 분리해 응답한다. 실제 수락/제출 전에는 완료로 기록하지 않는다.

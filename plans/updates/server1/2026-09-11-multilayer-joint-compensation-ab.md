# SH1 A 구현 대응표

Instruction ODEEDIT-S06-MULTILAYER-JOINT-EDIT-A-SH1-V1. 기준 main a9b696bd.
기존 v2/ABC/BLUE 원본은 읽기 전용이며 새 A와 SH2 B는 독립이다.

|설계|구현 소유/경로|검증|
|---|---|---|
|공통 We/Current/latest-wins/네 bank|SH1 contracts.py/fixture.py/banks.py|고정10k SHA/order, metadata-only 선정, nested1/7/100|
|L8 history 재구성·공통 N4/BLUE|SH1 history.py/common_reference/|같은 We, 원 M4 보존, endpoint append once|
|A0 writer-aware 공동 RHS|SH1 track_a/planner.py 및 모델 binding|R0, joint full-weight forward, q image metric, 25 logical updates|
|Full P*·native SPD writer/metric|SH1 track_a geometry|raw sum/mean 모든항 변환, 실제 전체 P*|
|출력 functional GGN/PCG/elastic|SH2 functional.py/linear_solve.py/elastic_qp.py|tuple support, 전체sequence JVP/VJP, cross block, exact dual domain|
|A OS/BF/Frozen|SH1 track_a/|동일 snapshot joint equality, native-risk budget, reference path 고정|
|평가·signed attribution·chain|SH1 evaluation.py/observations.py/track_a|이번 delta만 제거, 모든 분모, 자기 W/M/We teacher|

공통 kernel API는 WeightTree=tuple, support순서4,8 또는8 또는4.
logits_fn(weights)는 전체sequence forward 후 prediction logits[T,V]를 반환한다.
token_mean_weights=1/L, context_weights=1/(B·context수)는 전체 logical 분모이며
chunk마다 재정규화하지 않는다. SH2 수신 VERIFIED 전 common parity 완료 주장0.

Middle 공통 준비/A0/A-OS의 실제 비용을 먼저 기록한다. SH2 B-OS 완료나 성능을
기다려 A 준비를 멈추지 않는다. GPU cap2/mem182272M, hour cap null(이전 예산 상속0).
성능 gate/추가 BF1/sweep/10k chain0. 지정 A12+공통6과 세10-batch short chains만 실행한다.

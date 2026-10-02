# JLZ v7 current key와 writer 공동 최적화 설계

2026-10-02. 실제 하층 write가 바꾼 key로 상층 writer를 매 후보 다시 풀고, 그 의존성을 native subject δ 공동 학습에 연결한다. V6의 fixed-P 근사를 교체하며 native 문장·Adam·사후 clamp·두 norm arm·각500-edit 범위는 유지한다.

- [Method 정본](method-ko.md): 실제/가상 경로, causal writer, 현재 geometry와 전체 gradient.
- [구현 계약](implementation-ko.md): layer barrier, implicit solve/direct D·P VJP, cache/checkpoint, 수치 검증.
- [기계 판독 계약](contract.json).
- [TeX](../../../docs/methods/jlz-causal-writer-v7.tex).
- [CPU 수학 검증](math/README.md)과 [결과](math/validation-results.json).
- [500-edit 실행 설계](experiment-500/experiment-ko.md)와 [실행 JSON](experiment-500/experiment.json).
- [GH와 SH4 전달용 변경 요약](handoff-ko.md).

CPU 합성 대수·gradient 검증은 실행했다. Production 구현·실제 모델 FP32/GPU parity·처리량·PS/NS 개선은 미검증이다. TeX 정적 검사는 수행했으며 PDF는 컴파일하지 않았다. 최신 사용자 지시로 기존 SH4 작업 중단 사실과 함께 v7 실행을 GH→SH4/server4로 전달한다. [실행 명령](execution-command.json)의 권한과 실제 전달·접수·제출 상태를 구분한다.

자료는 v6 전달본을 수정하지 않고 새 revision으로 작성했다. 비용에는 매 후보 전체 current rewrite key-builder와 상층 solve/backward가 추가되므로 이전 v6 시간 추정은 사용할 수 없다.

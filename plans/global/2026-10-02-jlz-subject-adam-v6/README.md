# JLZ v6 method 패키지

2026-10-02. Native subject δ 주입·Adam·모든 편집층 공동 학습을 유지하면서, writer의 물리 부담·실현오차와 native 보존 입력을 배분에 연결하는 설계다.

읽는 순서는 다음과 같다.

1. [Method 정본](method-ko.md): 목적, 두 arm, 모든 수식과 유한 update 일정.
2. [구현 계약](implementation-ko.md): 모듈별 책임, gradient·cache·commit·telemetry.
3. [기계 판독 계약](contract.json): profile과 변하지 않아야 할 실행 의미.
4. [TeX 수식 문서](../../../docs/methods/jlz-subject-adam-v6.tex).
5. [CPU 수학 검증](math/README.md)과 [결과](math/validation-results.json).
6. [첫 500-edit 실험 설계](experiment-500/experiment-ko.md)와 [실행 계약](experiment-500/experiment.json).
7. [GH→SH4 전달문](GH-HANDOFF.md).

v6 A/B의 유일한 차이는 geometry 배분 norm의 형태다. A는 층별 norm의 합, B는 층 전체의 결합 norm을 쓴다. 두 arm 모두 동일한 native subject 학습, actual writer 보조 학습, 과거 native 보존을 사용한다. 이전 v5 A/B의 보존항 on/off와 구분한다.

상태: 문서 및 CPU 수학 검증 완료. Production 구현·실제 모델 qualification·성능 pilot은 미실행이다. TeX는 원문과 정적 검사를 제공하며, 로컬 컴파일러가 없어 PDF는 만들지 않았다. 최신 사용자 지시로 warmup을 제거하고 두 arm 각 500 edits의 구현·pilot·실험을 GH→SH4/server4에 전달한다. 실제 전달/접수 상태는 별도 dispatch receipt에 기록하며, 설계 완료를 제출 또는 실행 완료로 간주하지 않는다.

기존 source와 실험을 수정하지 않았다. 구현을 진행할 때는 `project/run_scripts/jlz_subject_adam/`의 새 경로를 사용하고, 이 package의 fingerprint와 실제 실행 source를 receipt에 결속한다. Batch size·benchmark·model에 대한 실제 실행 profile과 stream은 별도 실험 설계에서 정한다.

첫 실행은 native LR을 첫 update부터 고정한다(현재 profile 0.1). Native clamp는 사후 투영으로 유지하며, clamp 기반 LR/gate는 추가하지 않는다. Method의 가변 B·model·benchmark 계약은 그대로이며 BS100×5는 이번 실행 범위다.

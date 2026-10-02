# JLZ v5 재설계 패키지

핵심 결정은 **batch entry에서 고정한 writer로 실제 모델을 공동 학습하고, 마지막 accepted 가중치를 그대로 commit**하는 것이다. 모든 eligible layer와 한 층 집중 가능성을 유지하며, 현재·과거의 native 문장만 사용한다.

- [Method 설계문](method-ko.md): 변수·수식·두 arm·보존·solver·계산량·한계의 정본.
- [구현과 검증 명세](implementation-ko.md): 새 모듈의 책임, 수치 정합, 작은 pilot, 비용 장부.
- [기계 판독 계약](contract.json): 구현자가 고정해야 할 조건과 provisional profile.
- [TeX method 원고](../../../docs/methods/jlz-writer-coupled-v5.tex): 동일 방법의 수식 원고. 현 환경에 TeX engine이 없어 PDF compile은 수행하지 않았다.
- [Geometry와 physical gradient 검산](math/validation.json): stdlib CPU fixture, 실제 모델·성능 검증 아님.
- [Prox와 보존 목적 검산](math/policy-validation.json): 영점 처리·재활성화·예산 내 backtracking·정규화·context별 악화 비용.

v5 A는 physical joint-target control, v5 B는 같은 구조에 과거 native KL anchor와 context별 target NLL 악화 비용을 추가한다. V4 B의 quadratic V 항과는 다른 method다. 현재 요청의 native 문장·readout·loss 항은 보존하지만, 직접 subject injection에서 physical writer graph로 바뀌므로 native compute_z와 동일한 최적화 문제라고 부르지 않는다.

기존 v4 chain에 hotpatch하거나 이어서 실행하지 않는다. 이 패키지는 재설계이며 source runner, 실제 GPU 정합·성능 검증, GH/SH4 수락·제출 영수증은 아니다.

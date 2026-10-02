# JLZ v9 실현량을 구분하는 공동 배분 설계

2026-10-03. 사용자 선택은 **MEMIT-H ridge writer 유지와 exact writer의 작은 pilot 비교**다. 주 방법은 모든 eligible layer의 subject local δ를 공동 최적화한다. Ridge의 보존 비용과 미실현 비용을 한 배분 항으로 합치고, 계획한 δ와 실제 평균 key에서 쓰인 변위를 구분한다. Exact writer로 본선을 대체하거나 absolute tracking을 추가하지 않는다.

- [방법 설계](method-ko.md)
- [첨부 리뷰의 검증과 교정](review-ko.md)
- [구현 계약](implementation-ko.md)
- [기계 판독 계약](contract.json)
- [Pilot과 두 arm 각각 500 edit 실험](experiment-500/experiment-ko.md)
- [실험 설정](experiment-500/experiment.json)
- [수식 문서](../../../docs/methods/jlz-realization-v9.tex)
- [CPU 수학 검증](math/README.md)

이는 설계 산출물이다. Production 구현, 실제 모델 pilot, 성능 개선, GH 전송 및 GPU 제출 완료를 뜻하지 않는다. 기존 v8 문서와 실행 중인 실험을 덮어쓰지 않는다. Manifest는 이번 산출물만 봉인한다.

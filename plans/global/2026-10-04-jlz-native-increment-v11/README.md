# JLZ v11 설계·2k 실행 패키지

사용자 승인: GH가 SH4에 구현→작은 pilot→2,000-edit 실행을 배정한다.

- [Method](method-ko.md) · [수식/실행 계약](contract.json) · [구현 계약](implementation-ko.md)
- [실험 설계](experiment-2k/experiment-ko.md) · [실험 JSON](experiment-2k/experiment.json)
- [Method TeX](../../../docs/methods/jlz-native-increment-v11.tex)
- [GH 전달문](handoff-ko.md) · [사용자 실행 명령](execution-command.json)
- [CPU 검산](math/README.md) · [artifact manifest](artifact-manifest.json)

MAIN은 direct local increment, frozen entry G+E root-sum 비용 .1이다. NOALLOC은 비용0 대조군이며 uniform이 아니다. MEMIT-H는 같은 조건의 native 비교다. 각각 같은 첫2k를 독립 cold 상태에서 BS100×20으로 편집한다. 실제 GPU qualification은 SH4가 수행한다.

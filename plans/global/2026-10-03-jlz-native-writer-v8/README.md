# JLZ v8 plain 방법 설계

사용자 결정 세 가지를 반영한 revision이다: replay 제거, MEMIT-H writer/history 정합, 첫 Adam 업데이트의 전면 clamp 포화 교정. 공동 local-z와 동적 배분의 목적은 유지한다.

- [방법](method-ko.md)
- [구현 계약](implementation-ko.md)
- [기계 판독 계약](contract.json)
- [TeX](../../../docs/methods/jlz-native-writer-v8.tex)
- [CPU 수학 검증](math/README.md)

현재 상태는 설계와 CPU 수학 검증 단계다. Production 구현, GPU pilot, 성능 검증, GH 전달, job 제출·취소는 이번 revision의 완료 항목이 아니다. 기존 v7 source/결과를 보존하며 새 writer history는 W0/H0에서 시작해야 한다.

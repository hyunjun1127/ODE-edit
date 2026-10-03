# JLZ 실현 주입 교정안 — v10 T′

2026-10-03, revision 3. 상태: **B1 강도 보정 철회, native norm 계수 .5 고정. Production 구현·실험 미실행.**

V9 구조 리뷰와 사용자가 전달한 실현 주입 제안을 검토한 결과를 정리한다. 핵심은 **실제로 만드는 subject 변화로 편집 이익을 계산하는 것**, 그리고 **writer를 어떤 모델 상태의 key로 만드는지 구분하는 것**이다.

- [수식과 실행 경로](method-ko.md)
- [필수 수정 / 반영한 결정 / 리뷰 주장 교정](changes-and-decisions-ko.md)
- [SH3 실행 실험 설계](experiment-500/experiment-ko.md), [실험 JSON](experiment-500/experiment.json)
- [GH 전달문](handoff-ko.md), [현재 사용자 실행 명령](execution-command.json)
- [Native 기본값의 baseline 비교 계약](comparison-ko.md), [V9 실현량 참고 기록](math/v9-b1-strength-reference.json)
- [CPU 수학 검산](math/validate_contract_math.py), [계산 결과](math/results.json)
- [T′ whole-B adjoint·유한차분 검산](math/validate_tprime_adjoint.py), [결과](math/tprime-adjoint-results.json)
- [이전 v9 구조 리뷰](../../../experiment-reports/global/2026-10-03-jlz-v9-deep-review/structural-review-ko.md)

사용자의 actual key 재계산 선택과 추가 리뷰 반영 지시에 따라 **매 후보 actual key/U 갱신 → actual context별 v^a를 subject fit에 주입 → 최종 U를 그대로 commit**하는 T′를 채택한다. Native 그룹 가중 actual norm, active clamp 없음, G-only 두 arm, detached pulse 제거를 반영했다. Subject-only 기저는 actual all-token 기저와 달라 전체 gap은 남는다.

**A/B 모두 λ_n=.5를 B1부터 끝까지 고정한다.** B1 계수 탐색·6회 추가 fit·v9 강도 matching은 사용자 지시로 철회했다. 주 비교 대상은 BLUE·MEMIT-H·AlphaEdit이며 v9는 내부 참고 결과다. Norm/key의 그룹 평균(.5/.1)과 native NLL의 context 균등 평균(1/6)을 구분한다. 같은 native 계수는 다층 공동 목적의 유효 규제 강도까지 같음을 뜻하지 않는다. [기계 판독 계약](contract-draft.json)은 구현 인계를 위한 설계이며 production qualification 통과를 뜻하지 않는다.

Production 코드·GPU 실험·기존 실험 상태는 아직 변경하지 않았다. 검산은 합성 행렬 및 작은 CPU causal 모델이며 실제 편집 성능 검증이 아니다. **현재 사용자 지시로 GH→SH3 구현·검증·실험 실행을 승인한 전달 패키지**를 작성했다. 실제 전달·담당 수락·제출은 별도 receipt로 구분한다. SH4는 이번 새 task 담당이 아니다.

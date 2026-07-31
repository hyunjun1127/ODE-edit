# MV-2 공통 JSON key-order 실패 RCA

## 범위와 증거 한계

이 감사는 두 분석 보고서, 두 실패 요약의 기술 필드, 사전등록/프로토콜,
MV-2 producer·analyzer 및 직접 관련 테스트만 읽었다. raw는 Llama와 Qwen의
`analysis_cases.jsonl` 첫 행에서 wrapper/payload의 키와 `technical` 객체의
키·타입만 투영했다. `effect`, `case_effects`, `per_case` 및 어떤 과학적 결과값도
읽지 않았다.

두 요약은 모두 analyzer가 수치 estimand 전에 exit 2로 fail-closed 했고,
12개 case가 같은 `technical` 순서 불일치의 영향을 받았다고 기록한다. 따라서
기존의 `BLOCK_TECHNICAL_INVALID` 표기는 정상 분석 결과가 아니라 입력 거부의
기술적 disposition이다.

## 정확한 불일치와 원인

| 객체 | analyzer의 요구 순서 | producer 소스의 구성 순서 | 두 raw 첫 행에서 확인한 저장 순서 |
| --- | --- | --- | --- |
| `technical` | `exact_panel`, `lineage_exact`, `matched_second_c`, `rollback_exact`, `firewall_pass`, `receipt_before_outcome` | 동일 | `exact_panel`, `firewall_pass`, `lineage_exact`, `matched_second_c`, `receipt_before_outcome`, `rollback_exact` |

Analyzer는 `_require_exact_keys()`에서 `tuple(value.keys())`를 요구 tuple과
동일 비교한다
([mv2_refresh_analysis.py:75](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/mv2_refresh_analysis.py:75),
[mv2_refresh_analysis.py:237](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/mv2_refresh_analysis.py:237)).
반면 producer의 `analysis_case["technical"]` literal은 analyzer의 요구 순서와
동일하다
([mv2_refresh.py:1184](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/mv2_refresh.py:1184)).
그 뒤 stream writer로 저장된다
([mv2_refresh.py:1226](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/mv2_refresh.py:1226)).

두 raw 첫 행은 wrapper와 payload도 모두 알파벳순이며, `technical`의 여섯 값은
모두 boolean 타입이었다. 즉 이 불일치는 case별 생산 계산이 아니라 producer의
구성 이후 JSONL 직렬화 경계에서 발생한 공통 순서 변환이다. Qwen 보고서도 12행
전체의 같은 `technical` 순서와 `lineage`·`compute`의 알파벳순 저장을 독립적으로
기록한다
([Qwen 분석 보고서:18](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/2026-07-31-mv2refresh-qwen-e0-v1-analysis.md:18)).

이는 `technical`만 고치면 끝나지 않는다. Analyzer가 순서를 강제하는 객체는
`lineage`, `technical`, 각 arm, `compute`, `budgets`다
([mv2_refresh_analysis.py:204](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/mv2_refresh_analysis.py:204),
[mv2_refresh_analysis.py:259](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/mv2_refresh_analysis.py:259)).
Producer는 `lineage`와 `compute`도 analyzer의 insertion 순서로 구성한다
([mv2_refresh.py:1211](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/mv2_refresh.py:1211)).
따라서 `technical` gate를 통과시켜도, 현 analyzer는 이후 저장된 nested object의
순서에서 다시 거부될 수 있다.

## 어느 쪽이 계약을 위반했는가

현 잠금 계약을 문자 그대로 적용하면 **영속 raw artifact를 만든 producer/직렬화
경계가 위반자**다. Producer 소스는 요구 순서대로 mapping을 만들지만, 실제
artifact는 그 순서를 보존하지 않아 analyzer 입력 계약을 만족하지 않는다. Analyzer는
자신이 선언한 엄격한 조건을 일관되게 fail-closed로 적용했다.

다만 이 계약 자체가 JSON 객체의 비의미적 member 순서를 과도하게 semantic하게
만든 설계 결함이다. 사전등록은 case 수, 고정 arm **배열** 순서, lineage, equal-C,
기술 gate를 고정하지만 JSON 객체 member 순서를 scientific invariant로 정하지
않는다. 그러므로 장기 canonical contract는 다음처럼 명시해야 한다.

- mapping: 정확한 key 집합과 각 key의 타입·값 검증, member 순서는 비의미적
- 배열: `branch_order`와 fixed arm 순서는 계속 exact하게 검증
- wrapper: schema/event/run_id/연속 sequence 검증은 계속 유지

수리 owner는 analyzer다. Analyzer의 객체-order 검사를 exact-key-set 검사로
바꾸면, 이미 immutable한 raw의 바이트·해시·값을 건드리지 않고 producer가 실제
사용한 JSONL canonicalization과 호환된다. 이는 raw를 재직렬화하거나 사후
normalization하는 수리가 아니다.

## 과학적 영향 및 재실행 판정

Analyzer-only 수리는 안전하다. 변경 범위는 `_require_exact_keys()`의 member-order
동등성뿐이며, 이름·집합·타입·finite 검사, case/ID/lineage 검증, fixed arm array
order, matched-C, technical fail-closed 및 output 계산 경로는 그대로 남겨야 한다.
두 raw artifact의 기존 SHA-256은 불변 evidence로 보존한다. 새 summary/report는
기존 실패 summary를 덮어쓰지 말고, 새 analyzer schema/revision과 원본 raw hash를
명시한 별도 산출물이어야 한다.

따라서 다음은 변경하지 않는다.

- threshold/noise envelope, bootstrap 설정 또는 metric 정의
- salted selection, 모델·데이터·lineage, case 수 및 denominator
- six-arm contrast와 `branch_order` 배열 순서
- C-budget, receipt/rollback/firewall gate 및 pair red gate

필요한 것은 **동일 raw에 대한 새 analyzer 재분석**뿐이다. model 실행, editing,
selection, raw 생성의 scientific rerun은 필요 없다. 새 analyzer가 order 이외의
schema/값/technical gate를 발견하면 그 시점부터 이 결론은 적용되지 않으며,
사전등록의 fail-closed 규칙을 따른다
([구현 사양:263](/mnt/raid5/janghj/ODE-edit/plans/global/2026-07-31-session01-mv2-implementation-spec.md:263)).

## 최소 수리 테스트와 중단 조건

1. Analyzer에 order-insensitive exact-key-set validator를 도입하되, 기존 모든
   호출에서 missing/extra key와 타입 검사는 유지한다. `branch_order`의 list 비교와
   sanitized wrapper의 schema/run/sequence 검사는 완화하지 않는다.
2. 직접 관련 analyzer test에 canonical insertion JSON과 재귀적 key-sort JSONL
   wrapper를 각각 만들고, 사람이 effect 값을 출력하지 않은 채 두 입력의 분석
   summary 동등성을 프로그램적으로 비교한다. 이 test는 `technical`, `lineage`,
   각 arm, `compute`까지 포함해야 한다. 현재 test는 insertion-order wrapper만
   unwrapping하므로 이 회귀를 잡지 못한다
   ([test_mv2_refresh_analysis.py:433](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_edit_motivation/tests/test_mv2_refresh_analysis.py:433)).
3. 별도 negative tests로 key 하나의 누락/추가, boolean·hash·finite type 오류,
   arm 배열 순서 오류, sequence 오류가 여전히 거부됨을 확인한다. 기존 technical
   fail-closed test도 그대로 통과해야 한다.
4. 두 raw의 전 행을 **키·타입만** 구조 검증하여 documented order 차이 외의
   schema 차이가 없음을 확인하고, 전후 SHA-256이 일치함을 확인한다. 그 뒤에만
   새 analyzer를 동일 raw에 read-only로 실행하고, 두 model의 새 compact summary를
   pair gate에 넘긴다.

다음 중 하나면 즉시 중단하고 analyzer-only repair를 승인하지 않는다: raw hash
변화, key set/type 변화, wrapper identity/sequence 불일치, 어떤 non-order schema
오류, 값/technical gate 실패, 혹은 수리 diff가 validation order 외의 상수·선택·
metric·denominator·contrast를 변경하는 경우다. 그 경우 새 과학적 실행 여부는
별도 red review에서 다시 판정한다.

SAFE_ANALYZER_REPAIR

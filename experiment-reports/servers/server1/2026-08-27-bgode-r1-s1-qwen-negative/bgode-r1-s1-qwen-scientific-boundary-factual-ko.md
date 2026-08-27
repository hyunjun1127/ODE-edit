# BGODE-R1 S1 Qwen 과학 경계 사실 보고서

## 판정

Qwen TECH-R6 job `26605`은 Official AlphaEdit fidelity를 **정확히 통과**한 뒤, 최초 Rayleighian arm인 `fisher-only-dynamic`의 node 0에서 `NumericalRankBoundary: equality direction is outside range(G)`로 fail-close되었다. 이는 잠긴 singular-aware 과학 경계이며 기술 수리 대상이 아니다. damping, ridge, 허용오차 완화, fallback, pseudoinverse cutoff 변경 및 재제출은 모두 0이다.

`scientific_promotion=false`이며, 6-arm panel terminal-valid denominator는 **0/6**이다. 따라서 Qwen 성능 수치나 Llama 대비 endpoint 비교를 만들지 않는다.

## 무엇이 유효한가

| 항목 | 사실 |
|---|---|
| job | 26605, scheduler `FAILED`, exit `1:0`, elapsed `00:15:27` |
| source | `44c569a50c178dda655c0d16395dc0087069c2ba` / `4104b58bd5860022c7717a8044c32cb70522d55e` |
| FULL-FP32 preflight | PASS; model=`qwen2.5-7b-inst`; boundary=`<|im_end|>`/151645 |
| Official dense RHS/native adapter | exact PASS |
| global relative Frobenius error | 0.0 |
| layer 4–8 relative error | 모두 0.0 |
| layer 4–8 cosine | 모두 1.0 |
| terminal boundary | equality direction outside `range(G)` |
| scientific result denominator | 0/6 |
| scientific promotion | false |

Official fidelity receipt는 `4d61b6ada645949e258b556eb6bdfc7583539ac65a0fedcf4852af09baa827e9`이고 identity는 `b33925fefb2af705254844de8a5c820e98bd887161267943fc907a6d2342a820`이다. 이는 exact dense RHS identity-basis factor가 Official FP32 update를 byte/numerical 관점에서 정확히 재현했음을 증명하지만, six-arm scientific panel 완성을 뜻하지 않는다.

## arm 경계

고정 arm 순서는 Native → Plain → Fisher → Full-moving → One-step-Full → Frozen-field-N4이다. Plain은 Rayleighian solver를 호출하지 않는다. 관측된 예외는 Rayleighian solver 내부에서 발생했으므로, source order상 최초 가능한 위치는 **Fisher node 0**이다. solver는 coefficient 구성과 `trajectory.apply`보다 먼저 호출되므로 Fisher arm writer action은 0이다.

Plain은 제어 흐름상 Fisher 전에 실행됐지만 전체 panel publication 이전에 중단되었으므로 Plain endpoint도 유효 denominator에 넣지 않는다. Plain 내부 physical action count는 별도 raw receipt에 기록되지 않았고, `AtomicWeightTrajectory.__exit__`가 예외 유무와 관계없이 pointer를 유지한 채 W0 bytes를 복원·SHA 검증한다. 따라서 보고서는 Plain 성능을 추정하거나 공개하지 않는다.

## lineage 분리

| lineage | job | 분류 | scientific denominator |
|---|---:|---|---:|
| TECH-R1 | 26587 | technical exclusion: vocabulary binding | 0 |
| TECH-R2 | 26588 | technical exclusion: native fidelity | 0 |
| TECH-R3 | 26595 | technical exclusion: native fidelity | 0 |
| TECH-R4 | 26599 | technical exclusion: pre-fidelity validator | 0 |
| TECH-R5 | 26602 | technical exclusion: rank-one representation | 0 |
| TECH-R6 | 26605 | locked scientific boundary: `range(G)` | 0 |

TECH-R1–R5 root와 TECH-R6 raw root는 모두 immutable이며 imputation과 partial endpoint 재사용은 0이다.

## Llama/Qwen 결속 범위

두 모델은 sealed Phase123 B1 request ordinal 0, case `19795`, request SHA `285a3add6f31d8546b0d76689a016bc0f9f7d8e58c95f54e87bba48ad5cabc65`, stream root `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a`, order root `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3`를 공유한다. tokenizer/model/termination은 모델별로 별도 봉인된다. Llama canonical 분석 소유자는 GH이며 이 package는 Llama 성능 표나 판정을 복사·결합하지 않는다. Qwen denominator가 0/6이므로 matched cross-model endpoint comparison은 `NOT_COMPARABLE`이다.

## 비주장

- H1/H2/H3의 Qwen terminal 판정은 만들지 않는다.
- barrier attribution, ODE attribution, efficacy/locality/heldout 성능을 추정하지 않는다.
- rank gate를 피하기 위한 수치 보정이나 fallback을 제안하지 않는다.
- Llama GH package 또는 P1R55 source/result를 변경하지 않는다.

모든 raw 경로와 SHA는 `qwen-artifact-inventory.json`, lineage는 `qwen-technical-history.json`, 공통 입력은 `qwen-llama-matched-inputs.json`에 결속되어 있다.

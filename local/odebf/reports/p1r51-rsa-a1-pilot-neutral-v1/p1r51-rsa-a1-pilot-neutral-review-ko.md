# P1R51 Phase-B Neutral pilot 독립 검토 보고서

## 범위와 경계

이 검토는 이미 기록된 P1R51 Phase-B Neutral case01 영수증과 불변 P1R43 Neutral case01 영수증만 읽었다. 모델·evaluator·GPU·Slurm 실행 또는 재실행은 0회이며, 실험 source/result를 변경하지 않았다.

- P1R51 source HEAD/tree: `9b4a88b73bd0c0fdd0b1470d77ea4354bd0e3486` / `4a3b5e2dbd7e15bf8da32117c25fd88ec2eaafc7`
- P1R43 parent HEAD/tree: `11508b6da11d606521b703037034e1814b70d8a8` / `a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b`
- P1R51 contract SHA256: `e19617806b9f60945291885d5fcc91383ee4f2c5198aaf40f77ab38bb9e1562a` (15841 bytes, 520 lines)
- 비교는 alias별 case01, request order/capture plan/objective plan/evaluator/aggregator/target span/evaluation-case identity 일치 조건에서 수행했다. action-freeze hash는 각 방법의 서로 다른 action 때문에 일치 조건이 아니다.

## 기술 무결성

| 항목 | Llama | Qwen |
|---|---:|---:|
| 완료 endpoint / 실패 endpoint | 1 / 0 | 1 / 0 |
| accepted K / BF16 materialization | [1, 2, 3, 4, 5, 6, 7, 8] / 8 | [1, 2, 3, 4, 5, 6, 7, 8] / 8 |
| 최대 RSA energy relative error | 7.046e-12 | 1.004e-11 |
| W0 pointer+bytes restore | PASS | PASS |
| action-freeze / heldout controller access / inner-step heldout | PASS / 0 / 0 | PASS / 0 / 0 |
| retry / cross-case state / nonfinite | 0 / 0 / 없음 | 0 / 0 / 없음 |
| PRIMARY / RESCUE / CURRENT request-step | 80 / 0 / 0 | 80 / 0 / 0 |

각 모델의 accepted index는 `1..8`이고 `h=0.125`는 8회, second-h·remaining horizon·semantic debt·persistent hold·trust/radius·KL/decay/preservation decision counter는 모두 0이다. `alpha_apply/alpha_req`는 각 step에서 1의 수치 오차 범위에 있다.

## P1R43 exact-matched case01 비교

| 모델 | 방법 | z8 full-six NLL | W8 full-six NLL | W−z gap | z Eff/Gen | W Eff/Gen | W Loc |
|---|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | P1R51 RSA | 0.061622 | 0.061190 | -0.000432 | 10/10 / 19/20 | 10/10 / 19/20 | 92/100 |
| llama3-8b-inst | P1R43 불변 | 0.100729 | 0.134933 | 0.034204 | 10/10 / 18/20 | 10/10 / 18/20 | 92/100 |
| llama3-8b-inst | Δ RSA−P1R43 | -0.039107 | -0.073743 | -0.034636 | Δz Eff +0, Gen +1 | ΔW Eff +0, Gen +1 | ΔLoc +0 |
| qwen2.5-7b-inst | P1R51 RSA | 0.034808 | 0.070956 | 0.036148 | 10/10 / 18/20 | 10/10 / 18/20 | 92/100 |
| qwen2.5-7b-inst | P1R43 불변 | 0.323804 | 0.487130 | 0.163326 | 10/10 / 18/20 | 10/10 / 18/20 | 93/100 |
| qwen2.5-7b-inst | Δ RSA−P1R43 | -0.288996 | -0.416174 | -0.127177 | Δz Eff +0, Gen +0 | ΔW Eff +0, Gen +0 | ΔLoc -1 |

## Target allocation·writer·보존·비용

세부 per-step 및 per-request 값은 동봉 JSON 표에 있다. 표에는 각 step의 nominal/RSA energy, allocation entropy/top-1/effective support, PRIMARY/RESCUE/CURRENT, alpha_req/apply, predicted/actual/realization, Structural-P/capacity/BF16 step energy 및 request별 target/W-only NLL·allocation share·accepted path를 기록한다.

## Phase-B 판정

**기술 판정: PASS.** 두 endpoint 모두 K8, action-freeze, W0 restore, input/evaluator identity 및 target energy 보존 receipt를 만족한다. 계약 §6의 Phase-B 중단 조건(nonfinite, request order/data mismatch, energy conservation 위반, target/terminal tensor contract 오류, BF16 materialization 실패, 명백한 catastrophic divergence, writer/router 계약 변경)은 이 검토의 영수증에서 관측되지 않았다.

**Phase-C release gate: RELEASE.** §6의 lenient pilot 규칙에 따라 Neutral independent B10×10은 release 가능하다. 이 판정은 Phase-B 기술/중단 조건에만 근거하며, 단일 batch 결과를 일반화한 과학적 우열 또는 promotion 판정이 아니다.

`scientific_promotion=false`.

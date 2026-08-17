# P1R51 Phase-C Neutral B10×10 독립 검토 보고서

## 범위 및 검토 경계

본 검토는 이미 terminal인 P1R51 Neutral 20 case와 불변 P1R43 Neutral exact-matched 20 case의 raw-free 영수증을 읽었다. 모델/evaluator/GPU/Slurm 실행 또는 재실행과 scientific source/result 변경은 수행하지 않았다.

- P1R51 source HEAD/tree: `9b4a88b73bd0c0fdd0b1470d77ea4354bd0e3486` / `4a3b5e2dbd7e15bf8da32117c25fd88ec2eaafc7`
- P1R43 parent HEAD/tree: `11508b6da11d606521b703037034e1814b70d8a8` / `a0e71bbb27cf36b3b51bf6cde4c92cdc9a47879b`
- 계약 SHA256: `e19617806b9f60945291885d5fcc91383ee4f2c5198aaf40f77ab38bb9e1562a` (15841 bytes, 520 lines)
- 비교 조건: alias·case·request order·capture plan·objective plan·evaluator·aggregator·target span·evaluation-case identity 일치. 모든 20 paired case가 이 조건을 만족했다.

## 무결성 사실

| 항목 | Llama | Qwen |
|---|---:|---:|
| 시도 / 완료 / 실패 | 10 / 10 / 0 | 10 / 10 / 0 |
| accepted K / BF16 materialization | 80 / 80 | 80 / 80 |
| 최대 RSA energy relative error | 1.636e-11 | 1.004e-11 |
| W0 pointer+bytes restore / action-freeze | 10/10 PASS / 10/10 PASS | 10/10 PASS / 10/10 PASS |
| retry / cross-case / nonfinite | 0 / 0 / 없음 | 0 / 0 / 없음 |
| heldout controller / inner-step heldout | 0 / 0 | 0 / 0 |

각 160 step에서 physical h=1회, second-h·remaining horizon·debt·persistent hold·trust/radius·KL/decay/preservation decision counter=0이다. 모든 route는 `JOINT_WRITE`, `alpha_apply/alpha_req` 수치 일치, hard P budget/veto와 retry/backtracking 영향=0으로 기록되었다.

## P1R43 exact-matched aggregate

| 모델 | P1R51 z8/W8 full-six NLL | P1R43 z8/W8 | Δz/ΔW | z Eff/Gen | W Eff/Gen | W Loc |
|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 0.064860 / 0.098573 | 0.236418 / 0.315244 | -0.171557 / -0.216671 | 10.0/10, 18.3/20 | 10.0/10, 18.1/20 | 87.1/100 |
| qwen2.5-7b-inst | 0.562288 / 0.992543 | 0.789759 / 0.894635 | -0.227471 / +0.097908 | 9.7/10, 15.8/20 | 9.6/10, 15.3/20 | 84.2/100 |

## 요청별 및 step별 기록

per-request 표(1600행)는 target current/selected NLL, gradient, nominal/RSA velocity, allocation amplitude/share, PRIMARY/RESCUE/CURRENT, accepted path, writer current/next NLL 및 target-to-terminal residual을 P1R43 same request/step과 함께 기록한다. per-step 표(160행)는 target energy, allocation concentration, alpha_req/apply, predicted/actual/realization, layer/P/capacity/BF16 energy의 직접 산술 차이를 포함한다.

## Phase-D Soft trigger 사실

- Qwen terminal z8 full-six NLL p90의 paired Δ 평균: -1.494489; 감소 case: 9/10.
- Qwen terminal z8 full-six NLL worst의 paired Δ 평균: -0.800305; 감소 case: 7/10.
- Qwen W8 full-six NLL의 paired Δ 평균: +0.097908; 감소 case: 7/10.
- Qwen z-Gen/W-Gen 정확 count의 case 평균 Δ: -0.300 / -0.100 (각 denominator 20).
- Llama z8/W8 full-six NLL의 paired Δ 평균: -0.171557 / -0.216671; 감소 case: 10/10 / 10/10.
- P1R51 CURRENT hold 합계: Llama 4 vs P1R43 8; Qwen 29 vs P1R43 37.
- 계약에 특정된 ‘기존 네 hard request’의 request-ID 목록은 contract 및 immutable P1R43 raw-free aggregate에서 `NOT_RECORDED`; 이 표는 source-recorded terminal p90/worst와 전체 paired request NLL을 대신 기록한다.

**Phase-D trigger 판정: RELEASE.** 계약 §6의 조건 중 Qwen hard-tail terminal NLL 감소가 전체 paired B10 case에 기록되어 있으며, 같은 target operator를 P1R43 Soft router에 연결하는 Phase-D 조건이 충족된다. 이 문장은 계약상 release taxonomy이며 성능 우열·promotion 판정은 아니다.

`scientific_promotion=false`.

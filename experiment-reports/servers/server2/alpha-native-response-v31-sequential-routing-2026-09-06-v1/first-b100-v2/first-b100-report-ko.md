# Alpha-JV sequential — 첫 B100 및 비용 예고

KST: 2026-09-06T19:04:53.238011+09:00

세 cell의 첫 B100이 실제 완료됐다. Qwen JV는 cap3에 따라 queued이며 아래 성능 표에는 추정하지 않는다. 이는 at-write current-B100이며 sequential final-W10 retention 결과가 아니다.

| Model | Arm | RS | PS | strict PS | NS | B100 total min |
|---|---|---:|---:|---:|---:|---:|
| llama3-8b-inst | O_NATIVE | 100/100 | 185/200 | 88/100 | 869/1000 | 18.59 |
| qwen2.5-7b-inst | O_NATIVE | 99/100 | 195/200 | 96/100 | 849/1000 | 13.51 |
| llama3-8b-inst | JV_NATIVE | 100/100 | 177/200 | 81/100 | 858/1000 | 27.45 |

Llama O/JV는 첫 W0/M0, request/order와 accepted-z가 exact 일치한다. 이후에는 각 arm의 자기 W/M에서 z를 다시 계산하므로 shared-z 인과 비교로 부르지 않는다.

RS/PS/NS는 NLL preference이며 tie는 실패다. Token correctness는 CSV 별도 열이며 자유 생성 정확도와 혼합하지 않는다.

## 첫 비용 기반 계획 (측정값과 proxy 분리)

6 chains의 단순 proxy 합: 23.31 GPU-hours. cap3와 main→L8 우선순위의 단순 wall proxy: 11.26 hours. 실제 보장/새 hard cap이 아니다.

Qwen JV는 아직 B100 실측이 없어 Llama JV write 비용에 두 모델의 이미 완료된 2-batch smoke writer 비율을 적용한 engineering proxy다. L8-only는 20→4 JVP 감소만 반영한 proxy이며 실제 overhead를 숨기지 않는다. 과학 metric imputation0.

각 chain에 10×첫 B100 cost와 additional seen-prefix 약 40,200 prompt-row 비용을 W0 평가 row당 시간으로 더했다. 최초 cold setup 및 checkpoint 비용을 10번 세어 보수적으로 중복 포함했고, history 성장·동시 workload·prompt-length 변동은 아직 실측되지 않았다. B5/terminal 실측이 이를 대체한다.

실제 checkpoint 크기에 기반한 6-chain W/M checkpoint1/5/10 예상 합: 115.84 GiB. Smoke/raw receipts/logs는 별도다. Full pretrained model은 중복 저장하지 않는다.

## 불변조건

2-model 2-batch smoke terminal-valid; actual W/M continuity, append1, commit F/B/solve0, checkpoint tensor reload, W0 restore PASS. Main source의 observer-only timing child는 수식/seed/sample/precision을 바꾸지 않았다. Main chains와 모든 finite endpoint는 계속하며 L8-only는 네 main chain 완주·main table 뒤 두 모델 모두 실행한다.

Source `77358b1546d1baf83b3e251afcce663b08d7bfd7` / tree `b64e84f2af7c708405dd6a8a9f018d3ece985c58`. Sample `40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd`.

실행 root `/mnt/raid5/janghj/ODE-edit/local/alpha-native-response-v31-sequential-routing/run-20260906-main-v2`. Main merge/push0; dedicated branches only.

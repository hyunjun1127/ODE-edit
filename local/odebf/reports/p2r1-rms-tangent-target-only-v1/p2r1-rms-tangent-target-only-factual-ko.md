# P2R1 Baseline-Calibrated RMS-Tangent Target Flow — 사실 기록

- instruction: `ODEEDIT-S05-P2R1-BASELINE-CALIBRATED-RMS-TANGENT-TARGET-FLOW-V1`
- source HEAD/tree: `8f817e13167289dac190fe74bfa42e2b3e01372d` / `abe577756ea82e1fed0847b6329c82dc49184e9f`
- exact parent: `11508b6da11d606521b703037034e1814b70d8a8`
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- numerical lock SHA/root: `688305826cb5c702e90da194ee2739bfc21b260ce6be1776f81019436e21b471` / `c770fa4b0944944912187105d67ea12c40725f09a27beab08b03d82847e73b1d`
- source manifest SHA/root: `9350d009ca14ffe5774d92fc8bd18e0167bc247647c0f07626a2dd81280db264` / `b8fddb8c40d9ebdf76bfbc4e5ea2d304da9ddc29cd73229584724147063bc892`
- P1R43 report SHA: `ed60bd3bc57bf2ee14d35d558674734b5ab7c446125131b2a281f58b7d3bfbf7`; Official identity SHA: `e12b19ff941f39a594d2ff0f8718849d102644fd2ec79f53ccb6d7efc9fd8d3c`
- P1R48-old: `SUPERSEDED_BEFORE_SCIENTIFIC_EXECUTION_BY_P2_RENUMBERING`; scientific_failure=false
- report policy: SH_FACTUAL_ONLY_REPORTING; scientific_promotion=false

## 1. 실행·무결성

| 모델 | scheduler | attempts/endpoints/failures | requests | target microsteps | W0 | writer/materialization |
|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 19885 COMPLETED 0:0 | 10/10/0 | 100 | 240 | 10/10 | 0/0 |
| qwen2.5-7b-inst | 19886 COMPLETED 0:0 | 10/10/0 | 100 | 240 | 10/10 | 0/0 |

- integrity counters: `{"case_W0":20,"case_manifest":20,"case_terminal":20,"task_W0_restored":2,"task_failed_case_zero":2,"task_terminal":2}`
- native endpoint runtime access=0; heldout controller access=0; retry/hold/debt/hard-P-budget=0.
- initial attempt 19881/19882: 24/24 target microsteps 후 terminal adapter가 outer K budget에 24를 전달하여 TECHNICAL_FAIL. 예외 SHA `7c7b851d19cecb9a9fb2a9188389d240713631da45671178eead02e6f20c311a`. TECH-R1은 evaluator freeze를 K8/snapshot9로 교정했다.

## 2. terminal z-inject 패널

| 모델 | Eff | Gen | Loc | Eff NLL case-mean(mean/med/p90/worst) | Gen NLL case-mean(mean/med/p90/worst) | full-six target NLL(mean/med/p90/worst) |
|---|---:|---:|---:|---|---|---|
| llama3-8b-inst | 100/100 | 197/200 | 892/1000 | 0.135148/0.000883269/0.382582/9.4375 | 1.18592/1.15298/1.68393/12.5625 | 0.093352/0.00113426/0.274297/0.588581 |
| qwen2.5-7b-inst | 100/100 | 193/200 | 849/1000 | 0.00111858/0.00110421/0.00153023/0.00842285 | 1.66472/1.78849/2.30423/11.875 | 0.000644457/0.000588986/0.000863496/0.000902363 |

## 3. target field·clamp·compute

| 모델 | clamp hits / request-microsteps | repeated-clamp requests | max tangent/rate residual | q mean | gamma mean | field cosine mean | target F/B | KL F/B | target/eval wall sum(s) | MaxRSS KiB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 1046/2400 | 100 | 1.24345e-14/9.09495e-13 | 139.559 | 0.465672 | 0.768405 | 1250/1200 | 1250/1200 | 262.546/12.2736 | 9642144 |
| qwen2.5-7b-inst | 0/2400 | 0 | 9.99201e-16/2.84217e-14 | 5.86258 | 0.459305 | 0.761395 | 1250/1200 | 1250/1200 | 213.641/11.5164 | 6508296 |

## 4. matched comparator arithmetic

| 모델 | comparator | cases | ΔEff correct sum | ΔGen correct sum | ΔEff NLL case-mean | ΔEff margin case-mean | identity |
|---|---|---:|---:|---:|---:|---:|---|
| llama3-8b-inst | OfficialAlphaEdit | 10 | 0 | 12 | 0.133979 | 1.70134 | MATCHED |
| llama3-8b-inst | P1R43-Neutral-EFF_Z_INJECT | 10 | 0 | NOT_APPLICABLE | -0.0811779 | 5.97728 | MATCHED |
| llama3-8b-inst | P1R43-Neutral-GEN_Z_INJECT | 10 | NOT_APPLICABLE | 15 | -0.962917 | 5.11139 | MATCHED |
| llama3-8b-inst | P1R43-Soft-EFF_Z_INJECT | 10 | 0 | NOT_APPLICABLE | -0.0242358 | 5.94237 | MATCHED |
| llama3-8b-inst | P1R43-Soft-GEN_Z_INJECT | 10 | NOT_APPLICABLE | 13 | -0.977264 | 5.14647 | MATCHED |
| qwen2.5-7b-inst | OfficialAlphaEdit | 10 | 0 | 1 | -0.0315274 | 2.84147 | MATCHED |
| qwen2.5-7b-inst | P1R43-Neutral-EFF_Z_INJECT | 10 | 4 | NOT_APPLICABLE | -0.882247 | 6.73965 | MATCHED |
| qwen2.5-7b-inst | P1R43-Neutral-GEN_Z_INJECT | 10 | NOT_APPLICABLE | 32 | -2.15723 | 6.04373 | MATCHED |
| qwen2.5-7b-inst | P1R43-Soft-EFF_Z_INJECT | 10 | 4 | NOT_APPLICABLE | -0.828401 | 6.33432 | MATCHED |
| qwen2.5-7b-inst | P1R43-Soft-GEN_Z_INJECT | 10 | NOT_APPLICABLE | 36 | -2.22247 | 6.06533 | MATCHED |

- Official Gen continuous NLL/margin per-case는 `NOT_RECORDED`; correct-count 비교만 포함했다.
- Native endpoint vectors/RMS의 runtime decision influence와 runtime access는 0이다.

## 5. gate·failure taxonomy 사실

- Gate A0/A1 technical/numerical: PASS (두 모델 10/10, 24 microsteps/case, certificate/W0/firewall PASS).
- ClampDominated telemetry: Llama repeated-clamp requests=100; Qwen=0.
- NonFinite=0; TargetWeak scientific classification=GH_RELEASE_PENDING; P2R1_TARGET_GATE_PASS는 SH가 발행하지 않았다.
- P2R2 scientific GPU execution authorization=0.

## 6. 산출물

- `p2r1-per-case.json`: SHA256 `38819beef6677e01a2139d85b3fbc2f91254868e1cadf1f4315c48a2f28f02d8`, bytes 52640, lines 1
- `p2r1-per-step.json`: SHA256 `74a57c1a7eba1d3bbe6fa33c2b2595825a8dbc586c36db27f5a2cfc58b61fb1d`, bytes 531110, lines 1
- `p2r1-per-request.json`: SHA256 `19f8ed012fd3efe99766fc36926f863949323b010669543b60c7df5f02872a11`, bytes 3077455, lines 1
- `p2r1-per-panel.json`: SHA256 `58a9dadcc21f67fa76ad3050627b1acf7500b68da2401d2b9bda8de46a5a4dd0`, bytes 40021, lines 1
- `p2r1-case-paired-comparisons.json`: SHA256 `283665b0b4e397890b78373f369506a05da7012b59c0acb7079a666f57e40170`, bytes 44251, lines 1
- `p2r1-aggregates.json`: SHA256 `706a49d7e27a18a4af82c2e554e62ad436fb9b8d371699971e1eb8825685bacf`, bytes 5682, lines 1

## 7. 경계

- FACT: 위 표의 값, 분모, hash, scheduler/W0/firewall 상태.
- NOT_RECORDED: Official latent trajectory, Official per-case Gen continuous NLL/margin, writer/P/capacity/energy (target-only gate).
- scientific_promotion=false.

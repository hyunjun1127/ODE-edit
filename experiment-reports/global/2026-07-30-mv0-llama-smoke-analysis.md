# MV-0 Llama 1-case smoke 독립 결과 분석

- 분석일: 2026-07-30
- 대상 run: `mv0_llama_smoke_v1`
- 대상 model: `llama3-8b-inst`
- 분석 역할: 실행에 참여하지 않은 독립 result-analysis agent
- 최종 판정:
  - **Llama 1-case implementation-fidelity/trace-neutrality smoke: PASS**
  - **MV-0 전체 및 research motivation: 미판정**
  - **Qwen smoke 실제 제출: HOLD**

## 1. 증거 범위와 claim boundary

이 분석은 다음 네 파일만 읽었다.

1. `local/results/raw/session01_motivation/mv0_llama_smoke_v1/manifest.json`
2. `local/results/raw/session01_motivation/mv0_llama_smoke_v1/summary.json`
3. `audits/global/2026-07-30-session01-mv0-execution-preflight.md`
4. `plans/global/2026-07-30-session-01-motivation-validation.md`

추가 scheduler fact는 다음과 같이 제공받았다.

```text
job 15500
State=COMPLETED
ExitCode=0:0
Elapsed=00:02:50
MaxRSS=9734028K
allocation: gpu=1, cpu=8, mem=65000M
```

`events.jsonl`, direct-z artifact, prompt/context 원문, Slurm log, 다른
report/code는 읽지 않았다. 따라서 이 보고서는 compact manifest/summary와
제공된 scheduler fact에 대한 독립 일관성 분석이다.

이 run은 case 한 건의 implementation fidelity와 trace neutrality를 확인하는
smoke다. 연구 가설, layer-routing motivation, 성능 우위, 두 모델 MV-0 완료,
후속 MV-1 진입을 지지하는 증거로 사용하면 안 된다.

## 2. 파일 무결성과 실행 identity

| 점검 항목 | 관측값 | 판정 |
| --- | --- | --- |
| 실제 `manifest.json` SHA-256 | `9fa3b63752a9f398dc2e97e5b7021218136a399c62ecaf179ef999d1ee94b635` | PASS |
| summary가 기록한 manifest SHA-256 | `9fa3b63752a9f398dc2e97e5b7021218136a399c62ecaf179ef999d1ee94b635` | 실제 hash와 exact match |
| 실제 `summary.json` SHA-256 | `3609cf679ad0987083ade08b2f11c9f9a28ceec95c26ededb114d998b9156c61` | 기록용 digest 계산 완료 |
| run ID | 양 파일 모두 `mv0_llama_smoke_v1` | PASS |
| model alias | 양 파일 모두 `llama3-8b-inst` | PASS |
| provenance ID | 양 파일 모두 `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b` | PASS |
| context ID | manifest context manifest ID와 summary `context_id`가 모두 `3020b3f5cea62e6cfbd173f0c99a4348cecf7e84425bb087720ced7f395482e5` | PASS |
| selection manifest ID | 양 파일 모두 `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce` | PASS |
| selected/attempted case | manifest `18447`, summary case result `18447` | PASS |

summary 자체의 digest는 이 분석에서 계산한 값이며, 허용된 증거 안에 별도의
외부 anchor는 없다. summary에 기록된 events SHA-256은
`22a6d989e79081b4d837c4913f10d5081c78d4976ac92414c8c9276b8a7e0635`지만,
금지된 `events.jsonl`을 열지 않았으므로 이 값은 재계산하지 않았다.

manifest의 model revision
`8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`와 runtime pin은 canonical
plan/preflight와 일치한다. manifest는 offline 실행, float32, `cuda:0`,
visible GPU 1개, `use_cache=false`, tracked clean commit
`471ace2591ef325c7f009333715de9d5dc991c5e`를 기록한다.

## 3. denominator와 종료 상태

| 항목 | 값 |
| --- | ---: |
| planned | 1 |
| attempted | 1 |
| case count | 1 |
| pass | 1 |
| failure | 0 |
| not run due to abort | 0 |
| abort failure type | `null` |
| summary run status | `completed` |
| summary all pass | `true` |
| scheduler state / exit | `COMPLETED` / `0:0` |

사전 계획된 한 case가 attempted denominator에 그대로 남았고, 실패나 abort로
사후 제외된 case가 없다. 따라서 이 run의 intention-to-diagnose denominator는
`1/1 attempted`, 결과는 `1/1 pass`다. 이 분모는 보존됐다.

## 4. fidelity, trace neutrality, rollback

summary의 compact metric은 다음과 같다.

| metric | 값 | 판정 |
| --- | ---: | --- |
| `all_rollbacks_exact` | `true` | PASS |
| max final-weight relative L2 error | `0.0` | exact zero |
| max teacher-logits relative L2 error | `0.0` | exact zero |
| max teacher-NLL absolute error | `0.0` | exact zero |

기술 실패 없이 완료됐고, 한 case에서 기록된 rollback과 세 비교 오차가 모두
exact zero다. compact 결과가 표현하는 범위에서는 implementation
fidelity/trace neutrality smoke 조건을 충족한다.

이는 한 case에서 관측된 exact result다. dtype-aware bound의 일반적 검증,
paired equivalence CI, deterministic 3-edit calibration, Qwen 재현을
대체하지 않는다.

direct-z는 manifest에서 `computed-once-and-frozen-under-this-local-run`으로
고정됐고 summary의 artifact count는 1이다. precomputed covariance 다섯
파일은 load됐으며 projector load는 0건이다. 이는 MV-0의 covariance
read-only 및 projector non-deserialization 경계와 일치한다.

## 5. artifact firewall

허용된 compact 증거 안에서 다음을 확인했다.

- manifest의 firewall 선언은
  `case IDs/hashes/metrics only; no raw edit or eval fields`다.
- `raw_templates_persisted=false`다.
- request는 원문이 아니라 SHA-256 ID로만 기록됐다.
- 두 compact JSON에서 raw prompt, generated context 원문, evaluation field를
  관측하지 않았다.
- summary의 `git_output_written=false`다.

따라서 **compact artifact firewall은 PASS**다. 다만 지시상 열지 않은
`events.jsonl`, direct-z artifact, 기타 raw namespace에 대한 byte-level
firewall audit까지 했다는 뜻은 아니다.

## 6. resource envelope

| 항목 | 요청/가용 | 관측 | 판정 |
| --- | --- | --- | --- |
| GPU 수 | 1 | scheduler allocation 1, summary visible 1 | PASS |
| CPU | 8 | scheduler allocation 8 | PASS |
| host memory | `65000M` | scheduler MaxRSS `9734028K`; summary host max RSS `11195260 KiB` | cap 이내 |
| elapsed | time limit `06:00:00` | scheduler `00:02:50`; summary `165.689 s` | cap 이내, 약 4.31초 scheduler overhead |
| GPU peak allocated | manifest total `50,899,386,368 B` | `40,711,586,304 B` | cap 이내 |
| GPU peak reserved | manifest total `50,899,386,368 B` | `42,689,626,112 B` | cap 이내 |

scheduler MaxRSS와 process-side host RSS는 계측 범위가 다를 수 있어 서로 exact
일치할 필요는 없다. 두 값 모두 `65000M` 요청보다 충분히 낮다. summary의 GPU
peak reserved도 manifest에 기록된 GPU total memory보다 낮다. 제공된 scheduler
fact와 compact resource record에서 resource envelope 위반은 없다.

## 7. 독립 판정과 다음 gate

### Llama smoke

**PASS**다. 근거는 manifest hash exact match, identity 일관성, denominator
보존, `1/1` completion, exact rollback, 세 exact-zero error, compact artifact
firewall, scheduler `COMPLETED/0:0`, resource cap 준수다.

이 PASS의 정확한 명칭은
`Llama 1-case implementation-fidelity/trace-neutrality smoke PASS`다.
`MV-0 전체 PASS`, `motivation supported`, `research evidence`라고 부르면 안
된다.

### Qwen smoke

이 보고서로 **Qwen smoke를 위한 result-analysis 선행조건 하나는 PASS**했다.
그러나 canonical preflight/plan은 Qwen 제출 전에 다음을 별도로 요구한다.

1. 실행 담당과 분리된 post-run red audit 승인
2. Qwen용 새 execution audit
3. 제출 직전 resource-cap/preflight
4. 새 run ID와 exact one-shot envelope
5. Llama와 동시 제출 금지

허용된 증거에는 독립 post-run red audit와 Qwen용 새 execution audit 결과가
없다. 따라서 **Qwen smoke의 준비·감사 단계로 진행하는 것은 허용하지만,
현재 상태에서 실제 Slurm 제출은 HOLD**다. 위 gate가 모두 PASS하면 그때
Qwen one-case smoke를 별도 승인할 수 있다.

MV-0 calibration case 확장이나 MV-1 진입은 계속 HOLD다.

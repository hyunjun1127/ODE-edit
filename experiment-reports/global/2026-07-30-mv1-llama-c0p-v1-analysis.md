# MV-1 Llama C0 pilot 독립 post-run 분석

- 작성일: 2026-07-30
- 분석 대상: `mv1_llama_c0p_v1`
- model: `llama3-8b-inst`
- Slurm child: job `15536.1`
- claim 상태: **3-case descriptive pilot only**
- 기술 판정: **PASS — C0 calibration 확장 가능**
- 연구 판정: **CONTINUE C0 ONLY / expected method gain 및 MV-2 진입은 BLOCK**

## 1. 핵심 결론

세 fraction `q ∈ {1/256, 1/64, 1/16}`은 각각 18/18
case-action derivative 비교가 완전했고, sign concordance `1.0` 및 사전
기준의 relative-error gate를 모두 통과했다. 따라서 finite-difference
measurement와 실행 artifact는 다음 C0 calibration을 진행할 수준의 최소
기술 신호를 보였다. 이 3-case pilot에서 가장 큰 통과 후보는 `q=1/16`이지만,
이는 **pilot candidate**일 뿐 model별 C0 budget lock가 아니다.

반면 3 case와 3 fraction 모두에서 controller가 선택한 arm은 `uniform`이었고,
사후 계산한 best static 및 event-wise retrospective oracle arm도 모두
`uniform`이었다. 그 결과 controller-minus-uniform, controller-minus-best-static,
retrospective-oracle-minus-best-static은 모든 case에서 정확히 `0`이었다.
즉 이 pilot에는 fixed action set 안의 event-specific routing opportunity가
관측되지 않았다.

따라서 현재 허용되는 결론은 다음 둘을 분리해야 한다.

- **CONTINUE:** artifact와 derivative calibration이 정상인 상태에서 C0
  calibration case를 더 실행한다.
- **BLOCK:** 이 3 case로 ODE-Edit의 expected method gain, positive routing
  effect, C0 budget lock, confirmatory GO 또는 MV-2 진입을 주장하지 않는다.

## 2. 독립 분석 범위

사전 커밋 `f364f37`의
`project/run_scripts/ode_edit_motivation/mv1_pilot_analysis.py`를 먼저
실행했다. 해당 commit의 analyzer SHA-256과 실행 파일 SHA-256은 모두
`3cd3c093226568ff3fa148eb12a4140c9365ee1376f6d2f4ed47c11f9619a7b8`로
일치했다.

Exclusive 분석 산출물은 다음과 같다.

```text
local/results/analysis/session01_motivation/mv1_llama_c0p_v1.analysis.json
sha256=c91c6b9e48aef4ca444bd49c8c6928ed01127eca86759c323bf1e7e6255eb2d1
analysis_status=descriptive_complete
claim_status=pilot_descriptive_only
```

분석 agent는 허용된 sanitized
`manifest/features/actions/outcomes/events/summary/action_receipts`와 지정된
문서, job `15536.1`의 `sacct` row 및 pair log만 읽었다.
`direct_z/*.pt`는 열지 않았다. Qwen local run과 raw proposal payload도 읽지
않았다.

## 3. 네 범주

### Proposal에서 온 내용

아래는 raw proposal을 다시 열지 않고, 지정된 구현 명세에 canonical하게
재기록된 범위만 사용한 것이다.

- fixed direct-z 아래 same-snapshot layer proposal의 local actuator를 비교한다.
- outcome-blind utility와 editor-native geometry만으로 event별 allocation을
  선택할 수 있는지 본다.
- static allocation으로 충분하거나 analytic utility가 actual progress를
  예측하지 못하면 routing/ODE 방향을 kill 또는 pivot한다.

### Repo/protocol에서 확인한 사실

- Llama 고정 snapshot은
  `meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`이다.
- action set은 `layer_4`부터 `layer_8`까지의 five single-layer arm과
  `uniform`으로 정확히 고정됐다.
- probe ratio는 `1/4`, utility는
  `temperature=1 mean-context-token smooth target margin`이다.
- 세 case는 calibration split의 선두
  `18447`, `3176`, `15669`이며, request identity는 모두 full SHA-256이다.
- 기존 covariance 5개를 verified read-only로 사용했고 summary의
  `projector_files_loaded=[]`이다. Manifest는 projector를 hash/size만
  검증하고 deserialize하지 않는 정책을 기록한다.
- run은 tracked-clean ODE-Edit commit
  `5df3cbfb7c34615df2100a6535acf83ea1496017`에서 실행됐다.

### GH 추정

- derivative 오차가 `q`와 함께 증가했지만 `q=1/16`도 현재 기준을 통과했으므로,
  full C0에서 재검증할 가장 큰 budget 후보는 `1/16`이다.
- 세 case 모두 `uniform`이 event-wise winner였다는 결과는 routing
  Motivation에 불리한 조기 경고다. 다만 3-case convenience pilot에는
  uncertainty bound가 없으므로 이것만으로 routing kill을 확정할 수 없다.
- `uniform`이 `ordered_global_alpha`보다 높았던 관측은 static allocation 또는
  geometry 차이를 더 볼 이유는 주지만, event-specific controller나 ODE
  refresh의 효과를 입증하지 않는다.

### 사용자 확인 필요

- 현재 없음. Canonical 절차에 따라 C0를 확장하되 이 pilot을 성능 claim에
  쓰지 않는 것이 직접적인 다음 단계다.
- GPU 시간과 utility progress를 한 scalar로 합치는 의사결정이 필요해질
  경우에만 별도의 사용자 교환가치 확인이 필요하다.

## 4. Artifact, denominator, receipt 독립 검증

사전 analyzer의 검증 결과를 받은 뒤 raw tensor를 열지 않는 별도 hash/count
join으로 다시 확인했다.

| 항목 | 기대 | 관측 | 판정 |
| --- | ---: | ---: | --- |
| planned / attempted / pass case | 3 / 3 / 3 | 3 / 3 / 3 | PASS |
| failure / abort로 미실행 | 0 / 0 | 0 / 0 | PASS |
| feature / commitment | 9 / 9 | 9 / 9 | PASS |
| operational outcome | 54 | 54 | PASS |
| `ordered_global_alpha` outcome | 9 | 9 | PASS |
| native reference / replay | 3 / 3 | 3 / 3 | PASS |
| total outcome / event | 69 / 3 | 69 / 3 | PASS |
| exclusive action receipt | 9 | 9 | PASS |

`manifest.json`, `features.jsonl`, `actions.jsonl`, `outcomes.jsonl`,
`events.jsonl`의 실제 SHA-256은 summary의 hash map과 모두 일치했다.
9개 receipt 파일도 summary 및 event reference의 파일명/hash와 모두
일치했다. 각 `(case, fraction)`의 receipt entry는 feature hash,
commitment hash, chosen action, request hash와 일치했고, 각 commitment
hash는 six operational outcome과 one `ordered_global_alpha` outcome,
즉 7개 outcome에 연결됐다. 모든 JSONL sequence는 `0`부터 연속이었다.

Receipt chain은 runner contract상
`feature/action write → flush+fsync → exclusive receipt → outcome` 선행관계를
검증한다. 다만 receipt 자체에는 외부 공증 timestamp가 없으므로 이 검증을
독립 wall-clock notarization으로 과장하지 않는다.

Structured JSON key 검사에서 raw prompt, answer, target, token IDs, logits,
raw generation 필드는 발견되지 않았다. Summary에 기록된 direct-z artifact
count는 3이지만 tensor 파일은 열지 않았으며, firewall 위반은 관측되지
않았다. 모든 branch의 `rollback_exact=true`, run summary의
`all_rollbacks_exact=true`였다.

주요 structured artifact hash는 다음과 같다.

| artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `587ca344ff1bbeb681dc71cf785ac6bc0401d6a78493ee24b90f0efb3753242a` |
| `features.jsonl` | `d27a4a4471cd5dadbd1496ec90f8511e83e823fd488ab651fd1f66488bf15817` |
| `actions.jsonl` | `a97305aeea867d5772b238ce38dae4fa61e3616f08a59db8baf7c4f6d11829b0` |
| `outcomes.jsonl` | `784b9cb630b8c6140fc399a705a9e05e48459858732e28e584ce3c3a54554a59` |
| `events.jsonl` | `e53c3578f1dd001ddde854e3d4b7cbdd9bb791cfa70ca7cc32d9c7eac99ea775` |
| `summary.json` | `1e44a165ffc5f660c17d536149239eef6a5cdd6cdf5d49d3059af206f8a26e7a` |

## 5. Fraction별 derivative calibration

단위는 case-action이며 각 fraction마다 `3 case × 6 action = 18`이다.
Sign concordance는 실패 또는 누락을 분모에서 제거하지 않은 ITD 값이다.

| fraction | finite / planned | sign concordance | median relative error | p90 relative error | locked criterion |
| --- | ---: | ---: | ---: | ---: | --- |
| `q=1/256` | 18 / 18 | 1.000 | 0.039811 | 0.084677 | PASS |
| `q=1/64` | 18 / 18 | 1.000 | 0.078260 | 0.172626 | PASS |
| `q=1/16` | 18 / 18 | 1.000 | 0.180300 | 0.350578 | PASS |

Locked 기준은 sign concordance `>=0.80`, median relative error `<=0.25`,
p90 `<=0.75`, complete panel이다. 세 fraction 모두 통과했지만, 3-case
pilot은 C0 calibration lock를 승인하지 않는다.

## 6. Controller, static, ordered 및 oracle

각 fraction에서 controller와 retrospective best static은 모두
`uniform`이었다. Replay near-zero envelope는 세 case 모두 progress `0`,
따라서 envelope 값도 `0`이다.

| fraction | selected/static | controller progress: mean [observed range] | controller − uniform/static: mean [range] | oracle − static: mean [range] | controller − ordered: mean [range] |
| --- | --- | ---: | ---: | ---: | ---: |
| `q=1/256` | `uniform` | 0.603474 [0.129612, 0.979362] | 0 [0, 0] | 0 [0, 0] | 0.263246 [0.041879, 0.497778] |
| `q=1/64` | `uniform` | 1.358364 [0.277883, 2.266326] | 0 [0, 0] | 0 [0, 0] | 0.624150 [0.092301, 1.213776] |
| `q=1/16` | `uniform` | 3.437580 [0.651118, 5.787348] | 0 [0, 0] | 0 [0, 0] | 1.681442 [0.231182, 3.203255] |

`retrospective oracle − static=0`은 단순히 controller가 static을 골랐기
때문만이 아니라, 세 case 각각에서 six-arm panel의 최대가 `uniform`이었음을
뜻한다. 따라서 이 pilot에서는 oracle이 회수할 routing 여지조차 없었다.

반대로 controller-minus-`ordered_global_alpha`는 모든 case에서 양수였다.
그러나 `ordered_global_alpha`는 implementation spec상 contextual arm이며
oracle candidate가 아니다. 이 대조는 3-case 관측 범위일 뿐이며,
`uniform`의 static 효과, outcome-blind controller 효과, refresh 효과를
분리하지 못한다. 위 차이를 ODE-Edit expected gain으로 쓰거나 fraction 간
더하는 것은 금지한다.

Selected controller arm의 exact satisfaction은 9개
case-fraction observation에서 0/9였고, native MEMIT full reference는 2/3였다.
작은 fraction probe의 primary가 smooth progress인 점을 고려하면 이는
technical failure가 아니지만, 관측 progress를 full-edit success rate로
번역해서는 안 된다.

## 7. Resource 및 Slurm 검증

지정된 `sacct` child row는 다음과 같다.

```text
JobIDRaw=15536.1
State=COMPLETED
ExitCode=0:0
Elapsed=00:34:50
AllocTRES=cpu=8,gres/gpu=1,mem=65000M,node=1
MaxRSS=9368400K
```

Run summary의 wall time은 `2087.342 s`로 `sacct` elapsed와 약 `2.66 s`
차이다. Summary process peak와 Slurm sampling 값은 다음처럼 둘 다 cap
아래였다.

| telemetry | 관측 |
| --- | ---: |
| GPU peak allocated | 37.916 GiB / 47.404 GiB (79.99%) |
| GPU peak reserved | 40.525 GiB / 47.404 GiB (85.49%) |
| process `ru_maxrss` | 10.250 GiB |
| Slurm step `MaxRSS` | 8.934 GiB |
| requested step memory | 65,000 MiB |

두 host-memory peak의 차이는 sampling/measurement source 차이로 남기며,
더 큰 process peak를 사용해도 요청량의 약 16.15%다. Pair stdout에는 Llama
summary 한 건이 기록됐고, 지정 child가 `COMPLETED 0:0`임을 확인했다.
Pair stderr의 관측 메시지는 `torch_dtype` deprecation과 checkpoint loading
progress였으며 Llama run failure는 없었다. 이 보고서는 job `15536.1`만
판정하며 pair 전체 또는 Qwen 결과를 대신 판정하지 않는다.

## 8. Preliminary continue / block 결정

### Continue

- 동일 precommitted measurement contract로 Llama C0 calibration을 확장한다.
- `q=1/16`을 largest-passing **candidate**로 유지하되 full C0 결과 전에는
  lock하지 않는다.
- 다음 분석에서도 event를 통계 단위로 유지하고 누락/실패를 ITD 분모에
  포함한다.
- Qwen의 독립 분석과 GH pair-level red 종합 전에는 cross-model 판정을
  만들지 않는다.

### Block

- 3-case 값에 CI, population expectation 또는 expected method gain이라는
  명칭을 붙이지 않는다.
- controller/static/oracle의 세 `0` contrast를 확정 kill로 과장하지 않지만,
  반대로 `controller − ordered_global_alpha`의 양수 값을 routing/ODE gain으로
  부르지 않는다.
- Full C0에서 budget, static policy, controller policy를 lock하기 전
  confirmatory claim을 열지 않는다.
- Held-out achievable allocation gain이 양 model에서 생존하기 전 MV-2
  refresh experiment로 넘어가지 않는다.

현재 Llama-only pilot의 가장 정확한 기대효과 표현은 다음이다.

> 세 calibration case의 locked six-arm panel에서는 uniform static arm이
> controller와 retrospective oracle을 모두 포화해 관측 routing ceiling과
> controller gain이 0이었다. 이는 routing Motivation에 불리한 조기 신호지만,
> 3-case descriptive pilot이므로 expected ODE-Edit gain 또는 최종 kill을
> 식별하지 않는다.

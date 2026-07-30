# MV-1 Qwen C0 pilot 독립 post-run 분석

- 작성일: 2026-07-30
- 분석 대상: `mv1_qwen_c0p_v1`
- model: `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`
- Slurm: `15536.0`
- 분석 성격: **3-case descriptive pilot**
- 최종 판정:
  - 실행·artifact·rollback 무결성: **PASS**
  - C0 fraction lock: **BLOCK — 세 fraction 모두 locked derivative criterion FAIL**
  - Qwen pilot의 routing motivation: **positive signal 미관측**
  - 다음 실행: **runner readiness는 CONTINUE, method-gain claim과 fraction lock은 BLOCK**

## 1. 독립 분석 범위와 재현

읽은 범위는 다음으로 제한했다.

- `local/results/raw/session01_motivation/mv1_qwen_c0p_v1/`의 sanitized
  `manifest/features/actions/outcomes/events/summary/action_receipts`
- `project/run_scripts/ode_edit_motivation/mv1_pilot_analysis.py`
- `plans/global/2026-07-30-session01-mv1-implementation-spec.md`
- `audits/global/2026-07-30-motivation-expected-effect-forecast-audit.md`
- Slurm `15536.0`의 `sacct` row와 paired job log

`direct_z/*.pt`는 열지 않았고, Llama local run/report도 분석에 사용하지
않았다.

사전 고정 analyzer는 commit `f364f37`의 파일과 현재 실행 파일이 동일한
SHA-256
`3cd3c093226568ff3fa148eb12a4140c9365ee1376f6d2f4ed47c11f9619a7b8`
임을 확인한 뒤 다음과 같이 실행했다.

```bash
/usr/bin/python3 \
  project/run_scripts/ode_edit_motivation/mv1_pilot_analysis.py \
  local/results/raw/session01_motivation/mv1_qwen_c0p_v1 \
  --output \
  local/results/analysis/session01_motivation/mv1_qwen_c0p_v1.analysis.json
```

exclusive 분석 산출물의 SHA-256은
`177d9d8fb3a881d54974c850a59c5fe396a8367b366624150d27e82eb5973fec`이며,
`analysis_status=descriptive_complete`,
`claim_status=pilot_descriptive_only`다.

## 2. 네 범주

### Proposal에서 온 내용

- same-snapshot layer proposal을 local actuator로 비교하고, actual outcome을
  보지 않는 utility-based controller가 event별 routing opportunity를 회수할
  수 있는지가 Motivation의 핵심 가설이다.
- static allocation으로 충분하거나 analytic utility가 actual progress를
  예측하지 못하면 routing/ODE 방향을 kill 또는 pivot해야 한다.

### Repo/protocol에서 확인한 사실

- 이 pilot의 locked action set은
  `{layer_4, layer_5, layer_6, layer_7, layer_8, uniform}`이고,
  fraction은 `{1/256, 1/64, 1/16}`, probe ratio는 `1/4`다.
- controller는 outcome 전에 `score-before-outcome finite-action argmax`로
  action을 commit했다.
- utility는 `temperature=1 mean-context-token smooth target margin`이다.
- covariance는 기존 Qwen layer 4–8 moment 다섯 개만 read-only로 load했고,
  projector는 deserialize하지 않았다.
- Qwen model/tokenizer snapshot과 offline mode가 manifest에서 고정됐다.

### GH 추정

- sign은 안정적이지만 derivative 크기의 tail calibration이 불안정하다.
  특히 case `15669`가 세 fraction의 큰 relative-error tail을 주도한다.
- 세 case 모두에서 `uniform`이 controller, retrospective best static,
  event-wise oracle이 된 관측은 현재 six-arm set에서 Qwen routing
  heterogeneity가 약할 가능성을 시사한다.
- 다만 3-case pilot은 C0 20-case calibration을 대체하지 않는다. 따라서
  이것만으로 Qwen 전체 routing 가설을 kill하거나 ODE-Edit 기대효과를
  수치화할 수 없다.

### 사용자 확인 필요

- 없음. 현재 lock대로 C0를 진행하되, 이 pilot 수치로 fraction이나
  method-gain을 확정해서는 안 된다.

## 3. Artifact, 분모, receipt 검증

사전 고정 analyzer와 별도 대조에서 다음을 확인했다.

| 항목 | 기대 | 관측 | 판정 |
| --- | ---: | ---: | --- |
| planned / attempted / pass case | 3 / 3 / 3 | 3 / 3 / 3 | PASS |
| feature | 9 | 9 | PASS |
| action commitment | 9 | 9 | PASS |
| operational outcome | 54 | 54 | PASS |
| ordered global-alpha | 9 | 9 | PASS |
| native reference | 3 | 3 | PASS |
| no-op replay | 3 | 3 | PASS |
| total outcome | 69 | 69 | PASS |
| event | 3 | 3 | PASS |
| action receipt | 9 | 9 | PASS |

- `failure_count=0`, `not_run_due_to_abort_count=0`,
  `all_pass=true`, `all_rollbacks_exact=true`다.
- replay progress는 세 case 모두 정확히 `0`; near-zero envelope도 `0`이다.
- 실패 또는 missing row를 제외한 finite-only 분모로 바꾸지 않았다.
- 9개 case×fraction 모두 feature/action hash가 먼저 기록되고
  flush+fsync 뒤 exclusive receipt가 생성됐으며, outcome의
  `commitment_hash`까지 연결됐다.
- receipt 이름·내용·summary hash map과 실제 파일 SHA-256이 모두 일치했다.
  다만 분리 파일에 wall-clock timestamp가 없으므로 receipt chain은 runner가
  요구한 선행 commit을 증명하지만 외부 시각 공증은 아니다.
- sanitizer의 forbidden raw field 검사를 통과했고, 세 request ID는 모두
  full lowercase SHA-256이다.
- 주요 observed SHA-256:
  - `manifest.json`:
    `16ab083f90fa2d4ebf19d609c66508b08810342dd12e881190e750d49751701c`
  - `features.jsonl`:
    `70833a03d336f489b1e45fe29ca1e088c8fe64b9c74ec68cedfa23e70db02dd5`
  - `actions.jsonl`:
    `08c05c858e25c8986676dbc8395712343eff366ba2084dd189e7e16db12ddc95`
  - `outcomes.jsonl`:
    `8158554327c36029a0f5cf72a3a4d3ac49cf370de9653de745d75ad3a8e6ee88`
  - `events.jsonl`:
    `7d7f279a5df2de6b7091fab0486b1d676d7d8380806a7f99bf13feb4a2a900b1`
  - `summary.json`:
    `8f560b7a16f2b11d65fec90e75e24a7686496ee63e0932f2b1a3617b61591da6`

## 4. Fraction별 derivative diagnostic

Locked criterion은 complete 18/18 panel, sign concordance `>=0.80`,
median relative error `<=0.25`, p90 relative error `<=0.75`를 모두 요구한다.

| fraction | finite / expected | sign concordance | median relative error | p90 relative error | locked result |
| --- | ---: | ---: | ---: | ---: | --- |
| `1/256` | 18 / 18 | 1.0000 | 0.0931 | 1.1391 | **FAIL** |
| `1/64` | 18 / 18 | 1.0000 | 0.1962 | 1.2712 | **FAIL** |
| `1/16` | 18 / 18 | 1.0000 | 0.3753 | 1.1600 | **FAIL** |

- 모든 case-action의 derivative와 actual slope 부호는 일치했다.
- `1/256`, `1/64`는 median gate는 통과했지만 p90 gate를 통과하지 못했다.
- `1/16`은 median과 p90 gate를 모두 통과하지 못했다.
- 따라서 `passing_fractions=[]`,
  `largest_passing_fraction=null`이며, 이 pilot에서 fraction을 lock할 수 없다.
- tail error는 주로 case `15669`에서 발생했다. 이는 sign-only routing은
  가능할 수 있어도, finite-step progress 크기를 calibrated analytic
  derivative로 해석하는 현재 rule은 아직 신뢰할 수 없다는 뜻이다.

## 5. Controller, static, ordered, oracle

수치는 모두 smooth utility의 raw difference이며 percentage가 아니다.
각 contrast의 분모는 3/3 case이고 missing/failure는 0이다.

| fraction | controller − uniform | controller − retrospective static | controller − ordered global-alpha | retrospective oracle − static |
| --- | --- | --- | --- | --- |
| `1/256` | mean 0; range `[0, 0]` | mean 0; range `[0, 0]` | mean 1.6183; range `[0.6190, 3.3995]` | mean 0; range `[0, 0]` |
| `1/64` | mean 0; range `[0, 0]` | mean 0; range `[0, 0]` | mean 3.9541; range `[1.7766, 7.9526]` | mean 0; range `[0, 0]` |
| `1/16` | mean 0; range `[0, 0]` | mean 0; range `[0, 0]` | mean 4.2556; range `[1.9217, 5.7140]` | mean 0; range `[0, 0]` |

세 fraction 각각에서:

- controller는 3/3 case 모두 `uniform`을 outcome 전에 선택했다.
- calibration-retrospective best static도 `uniform`이었다.
- `uniform`은 각 case의 six-arm event-wise maximum이어서 raw 및
  near-tie-adjusted oracle gap이 모두 정확히 `0`이었다.
- controller가 ordered global-alpha보다 높았던 case fraction은 3/3이다.
  반대 contrast인 `ordered global-alpha − static`의 mean은 각각
  `-1.6183`, `-3.9541`, `-4.2556`이다.

`ordered global-alpha`는 locked oracle candidate가 아닌 contextual arm이다.
따라서 위 positive contrast는 이 pilot budget에서 `uniform` endpoint가
ordered contextual endpoint보다 높았다는 관측일 뿐, ODE-Edit method gain이나
MEMIT 대비 일반 성능 향상으로 해석할 수 없다.

## 6. 기대효과와 Motivation 판정

이 3-case Qwen pilot에서 방어 가능한 기대효과 기술은 다음뿐이다.

- candidate-set retrospective oracle ceiling relative to pilot static:
  세 fraction 모두 observed mean/range `0 / [0, 0]`
- outcome-blind controller relative to pilot static:
  세 fraction 모두 observed mean/range `0 / [0, 0]`
- controller relative to ordered contextual arm:
  observed mean `1.6183–4.2556`, 단 비확증적 contextual contrast

따라서 현재 Qwen 결과만으로 예측 가능한 ODE-Edit 개선폭은 **0이라고
확정할 수도, positive라고 주장할 수도 없다**. 관측된 3-case 범위에서는
routing이 static `uniform`을 개선할 여지가 전혀 나타나지 않았다는
negative descriptive signal만 보고할 수 있다.

다음 gate를 제안한다.

1. **Technical CONTINUE**: complete denominator, exact rollback, receipt chain,
   resource cap을 모두 통과했으므로 동일 runner로 preregistered C0 calibration
   분석을 계속할 수 있다.
2. **Scientific BLOCK**: 이 pilot으로 fraction lock, controller gain,
   expected method gain, MV-2 진입을 허용하지 않는다.
3. C0에서도 양상이 반복되어 어떤 fraction도 derivative criterion을
   통과하지 못하면 `analytic_utility_calibration_failure`다.
4. C0에서도 near-tie-adjusted oracle upper bound가 null이면 Qwen의 현재
   six-arm routing 방향을 kill하거나 action/controller를 pivot해야 한다.

## 7. Resource 검증

Slurm `15536.0` row:

| State | ExitCode | Elapsed | allocation | MaxRSS |
| --- | --- | --- | --- | ---: |
| `COMPLETED` | `0:0` | `00:43:16` | 1 GPU, 8 CPU, 65,000M | 15,046.80M |

Qwen summary의 runner-side resource:

- wall: `2593.10 s` (`43.22 min`)
- visible GPU: 1
- GPU peak allocated: `43,798,774,272 B` (`41,769.77 MiB`)
- GPU peak reserved: `47,248,834,560 B` (`45,060.00 MiB`)
- observed GPU total: `50,899,386,368 B`
- reserved peak: observed total의 `92.83%`; headroom 약 `3.40 GiB`
- process-side host max RSS: `17,341,384 KiB` (`16,934.95 MiB`)

`sacct MaxRSS`와 process-side RSS는 sampler/집계 경계가 달라 값이 같지는
않지만 둘 다 65,000M child allocation 이내다. OOM, non-finite,
rollback failure는 없었다. GPU reserved headroom이 약 7.17%뿐이므로 이후
Qwen lane에서 backward graph나 동시 추가 GPU-resident tensor를 여는 것은
별도 headroom 검증 없이는 금지하는 것이 타당하다.

## 8. Claim boundary

- 이 결과는 3-case descriptive pilot이며 confidence interval이나 population
  inference가 아니다.
- retrospective oracle은 candidate-set opportunity ceiling이지 achievable
  method gain이 아니다.
- controller/static/oracle contrast를 ordered contextual contrast와 더해
  “총 기대 향상”을 만들 수 없다.
- refresh gain, displacement 감소, retention 개선, 두 pinned model 밖의
  일반화는 이 결과에서 주장할 수 없다.

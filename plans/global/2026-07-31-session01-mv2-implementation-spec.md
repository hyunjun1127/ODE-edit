# Session 01 Motivation — MV-2 refresh diagnostic 구현 사전등록

- 작성일: 2026-07-31
- 소유자: `head-server1-gh` (`global-head`)
- 상태: `outcome-blind precommit`
- method name: `ODE-Edit`
- 선행 gate: MV-1 untouched pair red audit의 exact `MV2 PREPARE`
- raw artifact root:
  `local/results/raw/session01_motivation/`

이 문서는 MV-1 untouched outcome을 열기 전에 고정한 최소 MV-2 실행 계약이다.
MV-1이 선행 gate를 통과하지 못하면 아래 구현이 존재하더라도 실행하지 않는다.

## 1. 네 범주

### Proposal에서 온 내용

- fixed direct-z guide 아래 partial update 뒤의 state-dependent
  non-stationarity를 검증한다.
- initial-state decision을 그대로 재사용하는 arm과 state를 다시 측정하는 arm을
  비교해, ODE/relinearization이 필요한 최소 신호를 찾는다.
- paraphrase, neighborhood/locality, downstream evaluation prompt는 edit-time
  decision에서 금지한다.

### Repo/protocol에서 확인한 사실

- `PROTOCOL.md`가 canonical 운영 source다.
- Motivation의 고정 backbone은 `llama3-8b-inst`와
  `qwen2.5-7b-inst`이고, MEMIT layer는 정확히 `4,5,6,7,8`이다.
- canonical CounterFact selection은 salted rank의 처음 100개를
  calibration/confirmatory/untouched에 이미 고정했다.
- EasyEdit, model cache, dataset, covariance cache는 read-only이고 projector는
  identity preflight만 허용하며 deserialize하지 않는다.
- 현재 server1 GPU cap은 3, host-memory cap은 requested GPU당
  198117 MiB다.

### GH 추정

- 가장 작은 ODE 필요성 검정은 동일 W1에서
  `refreshed direction + refreshed coefficient`와
  `fixed W0 direction + refreshed coefficient`를 equal-C로 비교하는 것이다.
- coefficient refresh만 남으면 ODE/relinearization보다
  fixed-direction dynamic coefficient controller가 더 정직한 연구 방향이다.
- 아래 12-case/model은 paper-level 성능 검정이 아니라 Motivation
  mechanism kill-test로 충분한 최소 규모다.

### 사용자 확인 필요

- 없음. 사용자는 Motivation을 빠르게 실행하고 Llama/Qwen을 함께 제출하라고
  명시했다. 다만 실제 제출은 선행 MV-1 pair gate와 별도 red preflight를
  통과한 경우에만 허용된다.

## 2. 고정 실행 identity

| 항목 | 고정값 |
| --- | --- |
| Slurm job name | `odeedit_mv2refresh_pair_v1` |
| Llama run ID | `mv2refresh_llama_e0_v1` |
| Qwen run ID | `mv2refresh_qwen_e0_v1` |
| model | canonical fixed Llama/Qwen snapshot |
| layer | `4,5,6,7,8` |
| selection | canonical salted ranks `[100:112]` |
| case 수 | model당 12, 두 model 총 24 |
| seed | `17` |
| `q` | `1/256` |
| partial fraction `h` | `1/2` |
| bootstrap | seed `20260801`, 4,000 paired resamples |
| matched-C tolerance | `rtol=3e-5`, `atol=3e-5` |

salted ranks `[100:112]`는 기존 first 100과 exact disjoint여야 한다. dataset
source hash, first-100 order hash, next-12 order hash가 모두 manifest에 남고,
CLI로 case, seed, `q`, `h`, layer, arm, bootstrap을 바꿀 수 없다.

## 3. Frozen-target trajectory

각 case에서 다음 순서를 한 번만 수행한다.

1. W0에서 teacher-forced allowed rewrite utility와 target token identity를
   고정한다.
2. W0에서 direct-z를 정확히 한 번 계산하고 tensor/artifact/context/request/
   target-token hash를 lineage에 고정한다.
3. pinned covariance moment만 한 번 load한다. cache miss, recompute, download,
   projector load는 즉시 중단한다.
4. W0의 synchronous/ordered MEMIT factor와 layer별 central probe를 한 번
   구성한다.
   `h=0` sham은 같은 W0 proposal과 unit-action tensor/C-energy hash가 exact
   동일한지만 검증하고 중복 central probe 12회를 실행하지 않는다.
5. outcome을 보지 않는 W0 `score_mix` action을 `h=1/2`만 적용해 W1을 만든다.
6. 실제 적용 receipt와 W0/W1 parameter hash로 W1 descendant lineage를
   검증한다.
7. 같은 frozen W0 direct-z bytes를 exact lineage 아래 W1에서만 재사용해
   refreshed direction을 만든다. primary에서 direct-z를 다시 계산하지 않는다.
8. 모든 feature, six-arm action hash, C-budget을 flush/fsync하고 exclusive
   receipt를 만든 뒤에만 operational outcome을 평가한다.
9. 각 branch 뒤 W0 parameter hash와 CPU/CUDA RNG를 exact rollback한다.

`B0 = q * E_native`, 첫 partial step 거리는 `h*sqrt(B0)`, continuation
거리는 `(1-h)*sqrt(B0)`다. 따라서 `h=1/2`에서 partial과 각 continuation의
C-energy는 모두 `B0/4`이며, 세 continuation은 같은 immutable covariance로
재측정해 equal-C를 통과해야 한다.

## 4. 정확한 six-arm 대비

arm order는 다음과 같고 변경할 수 없다.

1. `no_op_replay`
2. `h0_sham`
3. `partial_joint`
4. `refreshed_direction_refreshed_coefficient` (`A`)
5. `fixed_direction_refreshed_coefficient` (`B`)
6. `fixed_direction_fixed_coefficient` (`C`)

`A`, `B`, `C`는 동일한 W1에서 시작하고 동일한 positive second-step C-energy를
사용한다. `B`와 `C`의 direction tensor bytes는 W0 factor basis와 exact
동일해야 한다. `A`만 W1에서 direction을 다시 선형화하며, `B`만 W1 probe로
coefficient를 다시 정한다.

## 5. 고정 estimand와 기대효과 보고

allowed teacher-forced rewrite utility의 W0 대비 progress를 `P(arm)`이라 한다.

```text
Delta_direction = P(A) - P(B)
Delta_coefficient = P(B) - P(C)
Delta_total_refresh = P(A) - P(C)
Oracle_refresh_opportunity = max(P(A), P(B), P(C)) - P(C)
Generic_continuation_gain = max(P(A), P(B), P(C)) - P(partial_joint)
```

`Delta_direction`이 유일한 primary다. `Delta_coefficient`는 direction이
clear하지 않을 때만 여는 conditional pivot estimand이고,
`Delta_total_refresh`는 secondary current-method 기대효과다. 따라서 이
hierarchy에서는 multiple-primary claim을 만들지 않으며 ordinary paired
bootstrap CI는 descriptive uncertainty로만 보고하고 gate를 뒤집지 않는다.

사용자가 요청한 method 기대효과는 secondary
`Delta_total_refresh = Delta_direction + Delta_coefficient`로 model별 point
estimate, 10% trimmed mean, median, sign, paired-bootstrap mean 95% CI를
보고한다. 이는 한 번의 equal-C continuation에서 stale controller 대비
refresh의 예상 absolute utility 개선이며, downstream accuracy나 최종 paper
성능으로 해석하지 않는다.

MV-1 untouched의 adaptive-vs-static allocation gain과
`Delta_total_refresh`는 서로 다른 case/support의 component forecast로 나란히
보고하되 단순 합산하지 않는다. full ODE-Edit 성능의 단일 숫자 예측은 MV-3의
동일 event/matched-progress baseline이 생기기 전에는 산출하지 않는다.

수치 outcome을 보기 전에는 refresh magnitude를 식별할 근거가 없으므로
MV-2의 사전 numeric point forecast를 만들지 않는다. 이것은 사후 threshold
조정을 막기 위한 의도적 제한이다.

`Oracle_refresh_opportunity`는 outcome-selected upper bound이며 실현 가능한
method 성능으로 쓰지 않는다. `Generic_continuation_gain`은 refresh와 무관한
두 번째 half-step 자체의 이득이므로 refresh kill을 구제할 수 없다.

W1에서의 controller incremental cost/case도 효과와 함께 공개한다.

| policy | 새 proposal build | allowed central-probe NFE |
| --- | ---: | ---: |
| A: refreshed direction/coefficient | 1 | 12 |
| B: fixed direction/refreshed coefficient | 0 | 12 |
| C: fixed direction/fixed coefficient | 0 | 0 |

이는 W0 공통 routing과 diagnostic-only outcome forward를 제외한 operation
count다. 전체 wall time, peak GPU memory, host RSS도 model별로 함께 보고한다.

## 6. Noise envelope와 model별 판정

case별 noise는

```text
e_i = max(1e-4, abs(P(no_op_replay)),
          abs(P(h0_sham) - P(no_op_replay)))
```

`1e-4`는 outcome을 보기 전에 고정한 absolute utility SESOI이며 deterministic
replay의 float jitter보다 큰 practical floor다. model gate의 envelope
`e_m = max_i(e_i)`다. 12-case 10% trimmed mean은
양 끝에서 정확히 1개씩 제외한다. 실패 case를 제거하지 않고 effect 0으로
denominator 12에 유지하며, technical gate 실패가 하나라도 있으면 scientific
판정을 차단한다.

model별 direction clear:

- `mean(Delta_direction) > e_m`; 그리고
- 10% trimmed mean `> e_m`, median `> e_m`, positive sign `>=7/12` 중
  하나 이상.

model별 coefficient clear도 같은 balanced gate를
`Delta_coefficient`에 적용한다.

model별 total clear도 같은 balanced gate를 `Delta_total_refresh`에 적용한다.

model별 direction null:

- mean과 10% trimmed mean이 모두 `<=e_m`; 그리고
- positive sign `<=6/12`.

판정은 다음 우선순위를 따른다.

1. technical failure: `BLOCK_TECHNICAL_INVALID`
2. direction clear이고 total clear:
   `DIRECTION_REFRESH_CLEAR`
3. direction clear이지만 total clear가 아님:
   `DIRECTION_MECHANISM_ONLY_REDESIGN_COEFFICIENT`
4. direction은 clear가 아니고 coefficient clear:
   `PIVOT_FIXED_DIRECTION_DYNAMIC_COEFFICIENT`
5. direction null, coefficient null,
   `Oracle_refresh_opportunity` mean `<=e_m`:
   `PIVOT_STATIC_ROUTING`
6. 그 외: `INCONCLUSIVE_BOUNDED_CONTINUATION`

단일 모델 analyzer는 pair/cross-model 결정을 내리지 않는다.

## 7. Pair red gate

Llama와 Qwen은 같은 parent allocation으로 동시에 제출한다. 각 model 결과는
서로의 artifact를 볼 수 없는 별도 agent가 먼저 분석하고, 이후 pair red
agent는 두 compact summary만 읽는다. 아래를 위에서부터 적용하며 첫 일치
판정에서 중단한다.

1. 한 model이라도 `BLOCK_TECHNICAL_INVALID`:
   `NO MV3 / TECHNICAL BLOCK`.
2. 한 model이라도
   `DIRECTION_MECHANISM_ONLY_REDESIGN_COEFFICIENT`:
   `REDESIGN COEFFICIENT`; 다른 model이 `DIRECTION_REFRESH_CLEAR`여도 현재
   ODE-Edit policy의 MV-3 진입을 금지한다.
3. 양 model `DIRECTION_REFRESH_CLEAR`—즉 direction과 A-C total이 모두 clear:
   `MV2 REPRODUCED`; 최소 MV-3 matched-progress diagnostic만 허용.
4. 정확히 한 model만 `DIRECTION_REFRESH_CLEAR`이고 다른 model은 technical
   valid이며 `mean(Delta_direction) >= -e_m` 및
   `mean(Delta_total_refresh) >= -e_m`:
   `ARCHITECTURE CONDITIONAL`; cross-model/일반 ODE claim 금지, clear
   architecture에 한해 최소 MV-3를 검토.
5. `DIRECTION_REFRESH_CLEAR`가 없고 한 model 이상
   `PIVOT_FIXED_DIRECTION_DYNAMIC_COEFFICIENT`:
   `PIVOT FIXED-DIRECTION DYNAMIC-COEFFICIENT`; refreshed-direction ODE를
   중단한다.
6. 양 model `PIVOT_STATIC_ROUTING`:
   `KILL DYNAMIC REFRESH`; static allocation만 남긴다.
7. 나머지 혼합:
   `NO MV3`; 재튜닝, threshold 변경, 추가 fold로 구제하지 않는다.

MV-2만으로 ODE-Edit 우위, locality/retention 개선, paper claim, MV-3 성공을
선언하지 않는다.

## 8. 최소 red preflight와 중단 조건

필수 preflight는 다음만 수행한다.

- 선행 MV-1 pair audit exact `MV2 PREPARE`
- server1 GH/SH session boundary와 repo CWD/HEAD 확인
- fixed model/dataset/EasyEdit core/covariance hash 확인
- first100/next12 disjoint selection 확인
- direct-z one-compute lineage와 target-token identity 반례 test
- equal-C, receipt-before-outcome, rollback/RNG, failure-in-denominator test
- `h=0` proposal/unit-action hash equality와 duplicate-probe 제거 확인
- `Delta_direction` clear지만 `Delta_total_refresh<0`인 반례가 advance를
  막는지 확인
- information firewall와 raw artifact Git exclusion
- `sbatch --test-only`, GPU/memory cap, Llama/Qwen same-parent pair 확인

다음이면 즉시 중단한다.

- 선행 gate 불통과
- source/artifact/session/CWD/Git identity mismatch
- direct-z recompute, cache recompute/download, projector load
- target token/context/request lineage mismatch
- action receipt 전에 outcome 접근
- unequal C-budget, non-finite metric, rollback/RNG failure
- case overlap/누락/중복, resource cap 초과, OOM

## 9. 산출물과 역할 경계

- raw tensor/log/outcome/receipt:
  `local/results/raw/session01_motivation/<run_id>/`
- compact model report:
  `experiment-reports/global/<date>-mv2refresh-<model>-e0-v1-analysis.md`
- model summary JSON:
  `experiment-reports/global/<date>-mv2refresh-<model>-e0-v1.analysis.summary.json`
- pair red audit:
  `audits/global/<date>-mv2refresh-pair-v1.postrun.md`
- server completion:
  `messages/server-heads/server1/<date>-mv2refresh-pair-v1.md`

GH가 time-critical exception으로 직접 제출하는 경우 목적, exact command,
GPU/memory 범위, 영향, 후속 보고 경로를 server1 inbox/audit에 기록한다.
다른 repo의 Codex session, 다른 server path, EasyEdit source는 조작하지 않는다.

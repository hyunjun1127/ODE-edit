# ODE-Edit proposal 재감사 및 EasyEdit baseline grounding

- 작성일: 2026-07-30
- 작성자: `head-server1-gh` (`global-head`)
- 문서 성격: 실행 전 논리·구현·leakage·재현성 감사
- 현재 판정: `preflight block` — 연구 가설을 kill한 것은 아니며, 아래 MV-0을
  통과하기 전 GPU 결과를 Motivation evidence로 승격하지 않는다는 뜻
- 원 canonical handoff:
  `project/proposals/00.proposal`
- 현재 canonical 연구명: **ODE-Edit**
- 현재 연구 세션: **Session 01 — Motivation Validation**

## 1. 감사 범위와 정보 출처

### Proposal에서 온 내용

- fixed direct-z guide 아래 여러 editable layer의 low-rank write를 local
  actuator로 보고, same-snapshot utility와 누적 geometry를 이용해 write를
  재배분할 수 있다는 가설
- utility heterogeneity, partial update 뒤 ranking non-stationarity,
  long-horizon load concentration을 Motivation signal로 삼는 구상
- 평가용 paraphrase, neighborhood/locality, downstream prompt를 edit-time
  controller에서 차단하는 information firewall

### Repo/protocol에서 확인한 사실

- `PROTOCOL.md`가 이 repository의 canonical 운영 protocol이다.
- 원 proposal은 historical handoff이므로 그 안의 `BF-ODE-Edit` 표기는 보존하되,
  이후 문서·코드·run에서는 `ODE-Edit`만 사용한다.
- `main`은 `origin/main`을 추적하며 remote는
  `https://github.com/hyunjun1127/ODE-edit.git`이다.
- server1 GH clone은 `/mnt/raid5/janghj/ODE-edit`이고, 아직 별도 server-head
  Codex session은 없다.
- raw trace, generations, model weights, covariance, projector, checkpoint는
  Git이 아니라 `local/`에만 둔다.

### GH 추정

- same-snapshot comparison은 유용한 order-neutral counterfactual이지만 그
  자체가 layer 간 “공정성”이나 routing benefit을 보장하지 않는다.
- 현재 방법은 실증 전까지 ODE라기보다 `state-dependent iterative feedback
  controller`로 부르는 편이 안전하다.
- Motivation은 아직 살아 있으나 원 proposal의 H1–H3만으로는 다음 세션
  진입을 정당화할 수 없다.

### 사용자 확인 필요

- 없음. 사용자가 실행 모델을 `Meta-Llama-3-8B-Instruct`와
  `Qwen2.5-7B-Instruct`로 고정했고, Motivation kill-test와 관련 문헌 조사를
  실제로 완료하도록 지시했다.
- 단, 별도 SH가 없는 동안 GH가 직접 Slurm을 제출해야 하는 경우에는
  `PROTOCOL.md`와 사용자 prompt가 정한 emergency/time-critical 예외 기록을
  제출 전에 별도 audit에 남긴다.

## 2. 병렬 감사 구성

GH는 세 역할을 분리하여 원 proposal과 local EasyEdit 구현을 재검토했다.

1. proposal logic audit: 가설 연결, 수식, falsifiability, novelty boundary
2. EasyEdit implementation audit: MEMIT/AlphaEdit entrypoint, state transition,
   context, covariance, cache, runner semantics
3. red baseline audit: leakage, metric error, parser/runner artifact, provenance,
   resource waste, overclaim

각 subagent는 Git push나 Slurm 제출을 하지 않았고, GH가 source line을 직접
재검증했다.

## 3. 가장 중요한 결론

### 3.1 정확한 baseline 대비

MEMIT을 “모든 layer update를 최초 model state에서 계산한 완전한 open-loop
editor”로 설명하면 틀린다.

EasyEdit의 MEMIT은:

- `easyeditor/models/memit/memit_main.py:160-233`에서 layer를 순서대로 방문한다.
- 각 layer에서 current-model key를 다시 계산한다 (`:163-165`).
- 앞 layer의 임시 write가 반영된 상태에서 z-layer activation과 residual을
  다시 계산한다 (`:168-179`).
- residual을 남은 layer 수로 나눈다 (`:210`).
- 현재 layer의 full update를 임시 적용한 뒤 다음 layer로 간다 (`:220-226`).
- 마지막에 원 weight를 복원하고 factor delta를 반환한다 (`:235-239`).

따라서 올바른 대비는 다음과 같다.

| 항목 | EasyEdit MEMIT | ODE-Edit Motivation 후보 |
| --- | --- | --- |
| delta construction | ascending-order Gauss–Seidel-style | same-snapshot Jacobi-style probe |
| state refresh | layer 사이 refresh | joint partial update 뒤 all-layer refresh |
| layer revisit | 한 edit 안에서 앞 layer를 재방문하지 않음 | accepted round마다 모든 layer 재방문 |
| allocation | remaining-layer residual 분배 | measured utility/cost에 따른 coefficient |
| 핵심 검증 | canonical baseline | refresh가 static/early-stop을 넘어 actual regret를 줄이는가 |

차이는 최종 tensor add 순서가 아니라 **delta construction과 layer revisit
규칙**이다.

### 3.2 Direct-z 비유일성의 claim boundary

full parameter Jacobian의 null space가 크다는 사실은 다음을 보장하지 않는다.

- 실제 5개 layer proposal의 non-negative cone 안에 대체 endpoint가 존재함
- 복수 layer가 서로의 rewrite progress를 보충할 수 있음
- 더 낮은 `C`-weighted displacement가 retention/locality를 개선함
- fixed direct-z guide를 쓴 두 endpoint가 같은 direct-z를 실제로 구현함

ODE-Edit controller가 접근하는 집합은 보통
`{sum_l c_l B_l : c_l >= 0}`인 저차원 cone이다. 또한 proposal의 terminal
condition은 `H^L = Z`가 아니라 output-side rewrite condition이다. 따라서
`same direct-z realization` 대신 **same fixed-z guide와 matched rewrite
endpoint**라고 기록한다.

### 3.3 Same-snapshot의 claim boundary

same-snapshot은 order bias를 제거한 first-order probe이지 layer 간 완전한
공정성 보장이 아니다. layer마다 causal depth, key coordinate, covariance,
weight scale이 다르고 joint update에는 cross-layer interaction이 있다.

따라서 다음이 관측돼야 actionable signal이다.

- autograd directional utility와 actual finite difference가 일치
- 복수 layer가 positive actual progress를 제공
- stale static choice가 refresh choice보다 actual regret를 보임
- joint predicted progress와 actual progress의 additivity error가 trust
  region 안에서 작음
- 한 layer를 cap했을 때 다른 layer가 matched progress를 보충

단순 `CV(a_l) > 0`, Spearman 변화, top-k turnover만으로는 pass하지 않는다.

## 4. 원 proposal 수식·정의의 block 항목

### 4.1 Rewrite deficit

원 proposal의

```text
d = logsumexp(other logits) - z_target
```

은 “strongest competitor margin”이 아니다. 이는
`log((1-p_target)/p_target)`이므로 `d <= 0`은 target top-1이 아니라
`p_target >= 0.5`를 뜻한다. outer unnormalized smooth aggregation까지 쓰면
context/token 수에 따라 threshold가 변한다.

Motivation code는 다음을 분리해야 한다.

- exact stop metric:
  `phi_stop = max_{context,token}(max_{v != y} z_v - z_y)`
- differentiable utility surrogate:
  normalized smooth approximation 또는 target sequence NLL
- multi-token target:
  teacher-forced position, leading-space tokenization, mask와 denominator를
  manifest에 기록

`phi_stop <= 0`만 모든 허용 context/token에서 target top-1이라는 뜻으로
사용한다.

### 4.2 QP progress constraint

실제 update가 `h * v_l`이면 first-order progress constraint에도 `h`가
포함돼야 한다. 또한 slack `xi`를 허용하는 이상 “required progress를
달성하는 해”라고 주장할 수 없다. 정확한 표현은:

> actual acceptance test를 갖고 soft progress shortfall을 penalty로 허용하는
> local controller

Motivation 단계에서는 QP superiority를 먼저 가정하지 않고 per-layer/joint
finite-difference diagnostic을 우선한다.

### 4.3 Capacity 용어

원 `Psi_l`은 현 상태와 base weight 사이의 covariance-weighted displacement
energy다. 이는 아직 remaining capacity나 forgetting의 검증된 척도가 아니다.

- cross term 때문에 한 step의 `Delta Psi_l`은 음수일 수 있다.
- step increment 합은 endpoint displacement로 telescope하므로 path expenditure와
  다르다.
- denominator가 큰 layer를 구조적으로 싸게 만들 수 있다.
- frozen base `C_l`은 current locality distribution이 아니다.
- 낮은 layer-load Gini가 반드시 좋은 것은 아니다.

실행에서는 다음 이름과 값을 분리한다.

- `c_displacement_state_l`: base 대비 현재 `C`-weighted displacement
- `c_path_expenditure_l`: accepted update별 non-negative squared energy 합
- `weight_norm_ratio_l`: NAS-aligned Frobenius norm ratio
- offline-only retention/locality damage

`capacity`라는 mechanism claim은 proxy relevance가 확인된 뒤에만 사용한다.

## 5. EasyEdit context와 information firewall

### 5.1 실제 context semantics

EasyEdit MEMIT은 단순한 “고정 5-prefix”가 아니다.

- `get_context_templates()`는 base `"{}"`와
  `["The", "Therefore", "Because", "I", "You"]`에서 top-k sampling으로 생성한
  5개 template를 process-global cache한다.
- direct-z는 raw + generated 5 context를 사용하고, 별도 KL row `"{} is a"`도
  사용한다.
- `compute_ks()`는 raw key와 5-prefix 평균을 다시 1:1로 평균한다. 6개 key의
  단순 균등 평균이 아니다.
- canonical MEMIT residual용 current `h^L`는 raw rewrite prompt에서 측정한다.

ODE-Edit이 six-context residual matrix를 쓰면 canonical MEMIT execution만
바꾼 것이 아니라 proposal family도 바꾼 것이다. Motivation main lane은:

- key aggregation: EasyEdit `compute_ks()` semantics
- residual: raw rewrite prompt
- direct-z: canonical EasyEdit target construction
- controller utility: frozen allowed six rewrite contexts

로 고정한다. six-context residual은 별도 ablation으로만 허용한다.

### 5.2 구조적 firewall

EasyEdit `BaseEditor`는 edit request와 rephrase/locality evaluation field를 같은
dict와 process에 보유한다. core MEMIT이 현재 그 field를 읽지 않는다는 사실만으로
firewall이 성립하지 않는다.

반드시:

1. edit process에는 `case_id`, `prompt`, `subject`, `target_new`만 담은 immutable
   sanitized request를 전달한다.
2. evaluation request와 evaluator process를 별도로 둔다.
3. controller artifact에는 evaluation prompt/label 문자열이나 dataset row
   전체를 쓰지 않는다.
4. generated context 문자열과 hash는 model/order별 fresh process에서 먼저
   freeze한다.
5. direct-z cache는 `model revision + order + step + state fingerprint` namespace를
   사용하거나 끈다. `case_id`만으로 두 order가 cache를 공유하지 않는다.

## 6. Local EasyEdit source와 baseline readiness

### 6.1 Provenance

- EasyEdit root: `/mnt/raid5/janghj/EasyEdit`
- Git HEAD:
  `3488a66ee988d83ee7891a8abbbe6bcb24a77daf`
- 상태: dirty worktree
- `memit_main.py` SHA256:
  `32f27516d30d196ceb2f99bc02359b66a0e886246953bdedc232f64fe7ee6d0a`
- `compute_z.py` SHA256:
  `6e43c1f03bc03c87dcff70111dff1a93f4e00ea31b4bfeb5e8eab4a516ea7ae4`
- `compute_ks.py` SHA256:
  `0b039be47c548f046b71cad4efce63414f53edbaa657bde868a20cdd35a2f5c7`
- local full tracked diff SHA256 at audit time:
  `88db7c8f3291682f229562918d5b132e382f35ba9c1e9b424237c7132404a89f`

HEAD만 기록하면 current compatibility patch가 빠진다. 실제 run manifest는
import되는 모든 EasyEdit file의 path와 SHA256을 기록하고 mismatch에서
fail-closed한다. ODE-edit는 EasyEdit file을 수정하지 않는다.

### 6.2 Credential block

EasyEdit의 untracked example shell 두 곳에서 literal HF credential이
발견됐다. 값은 읽어 옮기거나 report하지 않았다.

- 해당 shell 복사·실행·rsync·Git commit 금지
- credential은 노출된 것으로 간주해 사용자 측에서 rotate/폐기 필요
- ODE-edit runner는 `HF_HUB_OFFLINE=1`과 local cache만 사용
- token/credential 환경변수를 artifact나 subprocess command에 기록 금지

### 6.3 기존 runner를 쓰지 않는 이유

- `batch_size=100`은 atomic 100-step stream이 아니라 100-request joint batch다.
- `BaseEditor.batch_edit(sequential_edit=True)`의 pre metric은 edited state에서
  계산되는 경로가 있다.
- example runner는 eval prompt를 edit request에 싣고 full model을 저장한다.
- relative `./logs`와 `./data/stats`는 CWD에 따라 artifact/stat path가 바뀐다.

Session 01은 ODE-edit의 custom atomic runner에서 EasyEdit internal function을
read-only import한다.

## 7. EasyEdit 구현 baseline 분석

| 방법 | local registry | 이 연구에서의 역할 | Session 01 판정 |
| --- | --- | --- | --- |
| MEMIT | 있음 | primary actuator·canonical Gauss–Seidel baseline | MV-0부터 필수 |
| AlphaEdit | 있음 | null-space/history-aware adjacent baseline | MEMIT Motivation 생존 뒤 |
| ROME/R-ROME | 있음 | single-layer/contextual control | later contextual |
| EMMET | 있음 | preservation–memorization objective contextual baseline | later contextual |
| PMET | 있음 | direct-z/hidden-target construction 변형 | execution claim과 분리 |
| FT | 있음 | EasyEdit FT이며 LocFT-BF와 동일하지 않음 | later end-to-end |
| WISE/GRACE/IKE | 있음 | side memory/router/retrieval 등 정보·parameter family가 다름 | mechanism-matched 아님 |
| ENCORE/NAS/BetaEdit/CrispEdit/LyapLock | 없음 | proposal의 강한 external baseline | pinned external integration 전 실행 claim 금지 |
| WilKE | 없음 | 가장 인접한 static per-edit layer-selection prior | literature + later implementation 필요 |

EasyEdit에 등록돼 있다는 사실은 현재 두 model에서 run-ready라는 뜻이 아니다.
method별 hparams, model snapshot, stats/projector, runner semantics, imported-file
hash를 별도로 확인한다.

## 8. AlphaEdit 관련 별도 block

원 proposal의 `B_MEMIT P_l` 표기는 local AlphaEdit native solve와 같지 않다.
local 구현은 대략:

```text
solve(P_l (K K^T + cache_c,l) + L2 I, P_l K R^T)
```

를 사용한다. 또한 `cache_c`는 edit마다 edited-state key의 `K K^T`를 누적한다.
따라서:

- local AlphaEdit은 이미 compressed history state를 가진다.
- `execute_AlphaEdit()`은 weight를 복원해도 global `cache_c`를 변경한다.
- branch/rollback diagnostic에서 native execute를 호출하면 이후 run을 오염시킨다.
- projector filename에는 model뿐 아니라 layer set identity도 들어가야 한다.
- precomputed projector는 checksum·shape·model revision·layer set을 검증하고
  절대 재계산하지 않는다.

AlphaEdit Motivation lane은 MEMIT kill-test가 생존하고 별도 cache-isolation
audit를 통과할 때만 연다.

## 9. 고정 모델·데이터·precomputed artifact 사실

### 모델

| alias | pinned local snapshot |
| --- | --- |
| `llama3-8b-inst` | `meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| `qwen2.5-7b-inst` | `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28` |

다른 backbone 결과는 Session 01 canonical evidence에 섞지 않는다.

### CounterFact

- local full file:
  `/mnt/raid5/janghj/EasyEdit/data/counterfact/counterfact.json`
- row count: `21,919`
- SHA256:
  `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`
- proposal의 NAS-filtered `20,877` stream manifest는 local EasyEdit에서 확인되지
  않았다.

따라서 Motivation subset은 full file에서 deterministic eligibility rule과
case-id manifest를 새로 만들고 Git에는 case ID/hash만 남긴다. evaluation
prompt text는 edit process로 보내지 않는다.

### Wikipedia covariance stats

precomputed stats는
`/mnt/raid5/janghj/EasyEdit/examples/data/stats/` 아래에 존재한다.

- Llama-3-8B-Instruct: layer 4–8, 각 약 822 MB
- Qwen2.5-7B-Instruct: layer 4–8, 각 약 1.44 GB

runner는 absolute `stats_dir`를 사용하고 missing file에서 계산을 시작하지
않고 fail-closed한다. `layer_stats()`의 download/recompute path는 Session 01에서
금지한다.

### AlphaEdit projector

precomputed projector는 EasyEdit `examples/` 아래의 large local artifact다.
AlphaEdit lane은 exact filename, byte size, SHA256, tensor shape와 layer mapping이
고정 config와 맞을 때만 로드한다. missing/mismatch이면 재계산하지 않고
`blocked_missing_precomputed_projector`로 종료한다.

## 10. 수정된 Motivation kill-test ladder

### MV-0 — Baseline fidelity와 instrumentation

- 1–3 deterministic edit
- custom atomic runner가 native singleton MEMIT delta와 endpoint를 tolerance
  안에서 재현
- prefix, direct-z, key aggregation, residual, layer order, covariance, weight
  orientation, tokenizer target IDs를 manifest로 기록
- rollback 전후 editable-weight hash 동일

하나라도 실패하면 이후 실험을 실행하지 않는다.

### MV-1 — Calibrated layer heterogeneity

- 같은 snapshot에서 모든 layer proposal을 구성
- `C`-normalized signed autograd derivative와 symmetric/forward finite
  difference를 비교
- 복수 layer의 positive actual progress, derivative calibration error,
  best-vs-uniform actual progress/energy ratio를 기록

variance만 존재하고 actual progress가 없거나 derivative가 calibration되지
않으면 kill한다.

### MV-2 — Actionable non-stationarity

- no-op repeated measurement로 numerical noise floor를 측정
- matched small joint step 전후 ranking, proposal cosine, actual best layer를 측정
- near-tie를 별도 표시
- stale pre-step choice와 refreshed post-step choice의 actual regret를 비교

rank가 변해도 regret가 noise 수준이면 dynamic refresh claim을 kill한다.

### MV-3 — Reroutability와 joint validity

- global first-hit scaling, static layer scaling, best single layer,
  `C`-normalized uniform allocation, refreshed joint, Gauss–Seidel small-step를
  matched actual progress에서 비교
- 한 layer의 coefficient를 cap한 뒤 다른 layer가 progress를 보충하는
  compensation frontier를 측정
- predicted additive progress와 actual joint progress 차이를 기록

early stopping이나 total scale만으로 설명되거나 compensation이 없으면 dynamic
cross-layer routing claim을 kill한다.

### MV-4 — Short sequential proxy relevance

- MV-0–3 생존 뒤에만 100-edit canonical subset과 두 fixed order를 실행
- `C`-weighted state displacement, path expenditure, norm ratio, layer share를
  기록
- evaluation-only process에서 retention/locality damage와 proxy의 checkpoint
  association을 계산
- order/subject/relation/difficulty confound를 stratify

Gini/concentration만 있고 actual damage와 관계가 없으면 capacity-routing
mechanism claim을 kill한다.

### Claim-closing 추가 실험

ODE 용어를 유지하려면 다음 중 최소한 step refinement와 refresh benefit을
확인해야 한다.

- stale-choice regret
- layer-cap compensation frontier
- fixed initial direction 대 refreshed direction
- step-size refinement와 endpoint/trajectory stability
- Euler 대 Heun 또는 local truncation proxy
- proposal curvature/cosine change

평균 1–2 discrete round에서 fixed-direction/static choice와 차이가 없다면
`ODE`를 제거하고 static routing 또는 early-stop method로 pivot한다.

## 11. 최종 kill / go / pivot

### 즉시 실행 block

- EasyEdit imported-file hash mismatch
- precomputed covariance/projector missing 또는 shape mismatch
- stats/projector recompute/download 시도
- eval prompt/label이 edit process나 controller artifact에 존재
- rollback hash 불일치
- unknown/exceeded GPU·host-memory cap
- credential/raw output/weight/checkpoint의 Git 유입
- native fidelity 실패

### ODE/dynamic routing kill

다음 중 하나가 두 model에서 일관되면 kill한다.

- calibrated actual utility를 제공하는 layer가 하나뿐이거나 없음
- stale-choice regret가 no-op/noise floor와 구분되지 않음
- matched-progress compensation/reroutability가 없음
- refreshed direction이 fixed direction보다 유의한 이득이 없음
- step refinement에서 안정된 state-dependent trajectory가 없음

### Capacity mechanism kill

- `C`-weighted proxy가 retention/locality damage와 관계가 없거나 total norm보다
  약함
- lower Gini가 더 나은 outcome과 연결되지 않음
- order/relation/difficulty를 통제하면 concentration association이 사라짐

### Pivot

- dynamic signal 없음 + static layer effect 있음:
  static leakage-free layer routing
- routing 없음 + global scaling/first-hit 효과 있음:
  norm/early-stop execution control
- displacement proxy 실패 + refresh benefit 있음:
  capacity claim을 제거한 state-adaptive scheduler

### Motivation close / 다음 세션 진입

두 fixed model 모두에서:

1. MV-0 pass
2. finite-difference-calibrated multi-layer utility
3. noise를 넘는 stale-choice regret
4. matched-progress rerouting opportunity
5. acceptable joint prediction error

가 재현되고, MV-4에서 최소 한 proxy가 actual damage와 연결돼야 Motivation을
`supported diagnostic`으로 닫는다. 이는 long-horizon superiority, causal
mechanism, novelty, paper-ready claim이 아니다.

## 12. Red-team checklist

- [ ] imported EasyEdit runtime file과 hparams hash가 manifest와 일치
- [ ] model snapshot, tokenizer, stats path/hash, dataset hash가 pinned
- [ ] prefix generation seed와 frozen templates/hash 기록
- [ ] edit-only request schema에 eval field가 존재하지 않음
- [ ] evaluator가 별도 process/artifact namespace 사용
- [ ] direct-z cache가 model/order/step/state identity를 포함하거나 disabled
- [ ] target leading space와 token IDs 기록
- [ ] raw key vs six-context residual 정의가 lane별로 명시
- [ ] native MEMIT delta/endpoint parity
- [ ] transpose/orientation parity
- [ ] no-op noise, near-tie, failed edit, zero utility를 denominator에 포함
- [ ] early-stop/global scale/static layer control 포함
- [ ] bootstrap unit이 edit이며 order 수를 sample 수로 부풀리지 않음
- [ ] rejected step, rollback, infeasibility를 사후 제거하지 않음
- [ ] raw artifact는 `local/`, Git에는 compact summary/checksum만 존재
- [ ] experiment별 결과 해석을 실행 담당과 다른 agent가 작성

## 13. 현재 결정

이 감사 시점에는 **Motivation 가설을 kill하지 않는다.** 다만 원 proposal의
H1–H3 threshold, MEMIT 설명, rewrite deficit, capacity 용어를 그대로 사용한
실험은 block한다.

다음 허용 작업은:

1. ODE-edit 내부 reusable read-only EasyEdit adapter 구현
2. two-model immutable preflight
3. MV-0 1–3 edit fidelity

까지다. MV-0 결과를 별도 analysis agent가 검토해 `pass`를 낸 뒤에만 MV-1
이상으로 진행한다.

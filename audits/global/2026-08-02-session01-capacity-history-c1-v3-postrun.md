# Session 01 capacity/history c1_v3 — 최소 post-run audit

- 날짜: 2026-08-02
- job: `15855` / `odeedit_capacity_history_pair_c1_v3`
- source commit: `77aab8d8f8303e998d4063585a376a628150b6d3`
- technical verdict: **pass**
- scientific verdict: **`CAPACITY_HISTORY_HARM_SIGNAL`**

## 실행·자원

- Slurm job `15855`: `COMPLETED 0:0`, elapsed `01:29:41`.
- allocation: server1 `4 GPU`, `32 CPU`, `260000M`, node `devbox`.
- controller/evaluator child step 8개가 모두 `COMPLETED 0:0`이다.
- 네 1-GPU worker가 Llama/Qwen × MEMIT/Alpha로 동시에 시작했다.
- controller summary 기준 최대 host RSS는 약 `16.54 GiB`, 최대 GPU reserved는 약
  `46.59 GiB`; worker request `65000M`과 A6000 allocation 안이다.
- GH direct-submit 예외 사유: server1 SH 부재, 사용자의 time-critical 재구현·동시제출
  명령. 명령은
  `project/run_scripts/submit_session01_capacity_history_pair_server1.sh`였고 영향은
  job `15855`와 ignored `local/`뿐이다.

## 선행 technical recovery

- job `15842`/c1_v1은 shared lineage가 round 4 label을 허용하지 않아 첫 QP action 전
  fail-closed했다.
- job `15843`/c1_v2는 QP trust numerical residual과 lineage exact scale check가
  충돌해 첫 QP action 전 fail-closed했다.
- label 1--4 허용과 `1e-5` identity tolerance를 test와 함께 source에 고정했다.
  적용 tensor를 round/rescale하지 않았고 exact hashes는 유지했다.
- 두 실패 run은 evaluator 이전이므로 scientific evidence에 포함하지 않는다.

## Controller 계약

- controller 8개 모두 4 features/actions/events, terminal, `all_pass=true`다.
- native branch는 ordered native endpoint와 exact state identity가 일치한다.
- QP 16 edit의 accepted diagnostic마다
  `requested_gain == remaining_reference_gain_before`다.
- unmatched QP는 모두 accepted round 4회와 `budget_exhausted=true`다.
- Qwen MEMIT/Alpha 각 1 edit만 native reference에 도달했고 나머지 14 edit은
  budget exhaustion이다.
- exact-top1은 diagnostic-only였고 native utility match 전 terminal로 쓰이지 않았다.
- reject/retry는 0회, barrier violation은 numerical zero다.

## Evaluator·leakage·precomputed boundary

- evaluator 8개 모두 checkpoint 4, terminal, `all_pass=true`; 총 32 checkpoints다.
- checkpoint technical false는 0개다.
- evaluation fields는 네 controller barrier 뒤에만 load됐다.
- raw evaluation text/logit/token persistence는 false다.
- compact artifact와 analysis에서 NFE key는 0개다.
- covariance 5/5와 Alpha projector는 pinned precomputed artifact를 read-only로
  사용했고 재계산하지 않았다.
- EasyEdit source/global cache를 수정하지 않았다.

## Artifact integrity와 재현성

- controller/evaluator summary가 선언한 artifact SHA-256과 실제 file hash:
  `48/48` 일치.
- model checkpoint count `16+16`, pair technical pass true.
- Llama analysis SHA-256:
  `42b63463198c53c39880f7bef1d11d3cd93c9f1fed1a2ea6404b469c7469b78b`.
- Qwen analysis SHA-256:
  `58b25722f69d1e11de3c8360365632a9b5321cd7e805c6b42184934234ee400f`.
- Pair analysis SHA-256:
  `9f0d6cc5d4073aae1c8f103c2c669322be57c6cdd7a40b146a94b4def7bbd0e0`.
- evaluator/combined/pair compact 29 files aggregate SHA-256:
  `745637bfb654abdcf2fe8c075d2130b126739854f75eaa51b04b83c3b7b95113`.
- fresh local analyzer output은 canonical model/pair JSON과 `3/3` byte-identical하다.
- 실행 직후 HEAD와 `origin/main`은 모두 `77aab8d`, tracked worktree는 clean이었다.
- capacity/QP/controller/evaluator/analysis/lineage unit test `35/35`, worker shell
  syntax, `git diff --check`, GH changed-path access check가 통과했다.

Raw proposal tensors, action receipts, checkpoints와 logs는 ignored `local/`에만 두고
Git에 넣지 않는다.

## Agent gate

Llama, Qwen, pair/red 분석 agent를 각각 관련 c1 JSON/feature에만 격리하고 Terra Ultra
runtime metadata를 먼저 확인시켰다. 세 agent 모두 metadata를 검증할 수 없어 파일을
읽지 않고 즉시 `BLOCK` 종료했다. 독립 analysis로 세지 않으며 GH fallback만 쓴다.

Protocol의 pre-push-sensitive red gate는 원래 block이다. GH는 다음 좁은 범위만
waive한다.

- positive claim이나 Method 진입으로 승격하지 않는다.
- raw/evaluator/analyzer를 사후 변경하지 않는다.
- pair harm verdict와 realized-distance shortfall을 그대로 보고한다.
- Direct-z 별도 session evidence를 이 gate에 합치지 않는다.

이 waiver는 새 실험 제출, model-specific rescue 또는 larger scale에 재사용할 수 없다.

## Scientific audit

| Model | Family | current delta | mean path/native | native match | verdict |
|---|---|---:|---:|---:|---|
| Llama | MEMIT | `-4.032392` | `0.671819` | `0/4` | harm |
| Llama | Alpha-history | `-2.495123` | `0.686245` | `0/4` | harm |
| Qwen | MEMIT | `-0.397837` | `0.863935` | `1/4` | harm |
| Qwen | Alpha-history | `+0.029707` | `0.742208` | `1/4` | model signal |

Cost/KL/concentration 감소는 네 cell에 대체로 공통이지만 current non-collapse가 양
모델에 공통되지 않는다. Llama overload/reroute는 0이고 Qwen만 family별 1회라 routing
mechanism도 cross-model evidence가 아니다. Pair analyzer의 passing family는 `[]`다.

## Git/protocol/artifact broadcast

- Git에는 code/test/spec/audit/report와 compact hash만 둔다.
- credential, raw IP/username/port/key/token/private path는 추가하지 않는다.
- server1 외 peer SH/clone이 아직 없어 raw artifact broadcast는 no-peer exception으로
  미실시한다. 임의 SSH/rsync를 수행하지 않는다.
- Direct-z temporary session의 결과·artifact는 읽거나 병합하지 않았다.

## 최종 audit 판정

c1_v3은 구현·실행·artifact 측면에서 유효하다. c0 under-edit bug를 고친 뒤 결과가
크게 회복됐지만, model-common lenient gate는 여전히 실패했다. 현재 fixed-`K=4`
capacity-QP sequential controller는 closed-negative로 기록하고 추가 Motivation retune을
허용하지 않는다.

# Session 01 capacity/share exact-quarter c2_v1 — 최소 post-run audit

- 날짜: 2026-08-02
- job: `15875` / `odeedit_capacity_history_pair_c2_v1`
- source commit: `78b17d1`
- technical verdict: **pass**
- causal verdict: **Llama low-update confound confirmed; cross-model effect bundled with share change**

## 실행·자원

- parent job: `COMPLETED 0:0`, elapsed `01:28:11`.
- allocation: server1 `4 GPU`, `32 CPU`, `260000M`, node 1개.
- worker/controller step 4개와 evaluator step 4개가 모두 `COMPLETED 0:0`이다.
- 네 worker는 Llama/Qwen × MEMIT/Alpha를 동시에 실행했다.
- observed QP maximum은 host RSS 약 `16.54 GiB`, GPU reserved 약 `46.59 GiB`로
  child `65000M` 및 GPU allocation 안이다.
- GH 직접 제출 예외: server1 SH 부재와 사용자의 time-critical 동시 실행 지시. 영향은
  job `15875`와 c2 ignored `local/` namespace뿐이다.

## Technical integrity

- controller 8/8, evaluator 8/8 terminal 및 `all_pass=true`.
- evaluator checkpoint `32/32`, technical false 0.
- QP edit `16`, hop `64`; 각 edit가 정확히 4 hops와 total path `D`를 만족했다.
- max path relative error `1.62355e-16`.
- max hop relative error `1.88223e-16`.
- max applied share L2 error `2.22045e-16`.
- capacity barrier max violation 0, trust gate failure 0.
- positive overload observation 41개는 모두 hard suppression 계약을 통과했다.
- fresh analyzer output과 canonical model/pair JSON은 `3/3` byte-identical했다.

## Firewall·read-only boundary

- controller terminal/hash barrier 뒤 별도 evaluator process에서만 evaluation field를
  load했다.
- direct-z는 branch/edit당 한 번이고 raw evaluation text/logit/token을 Git에 남기지
  않았다.
- covariance 5/5와 Alpha projector/history는 pinned precomputed artifact를 read-only로
  재사용했다.
- EasyEdit source, global cache, dataset, model weight를 수정하지 않았다.

## Artifact integrity

- Llama analysis SHA-256:
  `8053ac2c753ed6bd7059709b478cf0a7e8558dcbdb3820d57b328e3cae731e18`.
- Qwen analysis SHA-256:
  `60086eeb7be8882f636e0fdf7630160ff9a3ccdfa7fdc3ccc8c7140842791a7f`.
- Pair analysis SHA-256:
  `43f4582e8f2cf6dbddcef25a2e831f5e4880dfe6cf147b3ff76d800bcbaafc59`.
- evaluator/combined/pair compact 29 files aggregate SHA-256:
  `52fc5a6629b2c0d60b26adc656d9d2df4fc4787ebd557825438bc0f47673793f`.
- raw proposal/action/checkpoint/log는 ignored `local/`에만 있다.

## Agent gate

관련 파일만 허용한 Terra Ultra Llama/Qwen/red agents 3개는 runtime metadata에서 지정
profile을 검증하지 못해 아무 파일도 읽지 않고 즉시 `BLOCK` 종료했다. 독립 review로
세지 않는다. GH fallback은 positive superiority claim을 열지 않고 식별 한계를
명시하는 좁은 범위로만 사용한다.

## Scientific audit

| Model | Family | c1 current delta | c2 current delta | c2-c1 |
|---|---|---:|---:|---:|
| Llama | MEMIT | `-4.032392` | `-1.369350` | `+2.663042` |
| Llama | Alpha-history | `-2.495123` | `-1.137219` | `+1.357904` |
| Qwen | MEMIT | `-0.397837` | `-0.797188` | `-0.399352` |
| Qwen | Alpha-history | `+0.029707` | `-0.400738` | `-0.430445` |

c2는 magnitude와 cap/share definition을 함께 변경했다. 그러므로 Llama에서는 c1
under-update confound를 확인하지만 Qwen worsening을 magnitude의 scientific harm으로
귀속하지 않는다. 동일 BF share를 고정한 magnitude-only control 외의 추가 tuning은
허용하지 않는다.

## Git/protocol/artifact broadcast

- Git에는 code/test/spec/audit/report/compact hash만 둔다.
- credential, raw IP/username/port/key/token/private artifact는 추가하지 않는다.
- peer SH/clone이 없어 raw artifact broadcast는 no-peer exception으로 미실시한다.
- direct-z 임시 session의 결과·artifact는 이 audit에 합치거나 모니터링하지 않았다.

## 최종 audit 판정

c2 실행은 기술적으로 유효하다. Llama에서 low-update 구현 confound는 확인됐지만,
Qwen causal attribution은 layer-share 변경 때문에 열려 있다. Motivation은 한 번의
BF-share-preserving magnitude-only control 뒤 닫는다.

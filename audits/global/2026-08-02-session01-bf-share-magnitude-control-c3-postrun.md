# Session 01 BF-share magnitude-only c3_v1 — 최소 post-run audit

- 날짜: 2026-08-02
- job: `15891` / `odeedit_capacity_history_pair_c3_v1`
- source commit: `bbd3d0ec5b8c41df96f367a0503d29823d7d7b89`
- technical verdict: **pass**
- causal verdict: **cross-model low-update implementation cause confirmed**

## 실행·자원

- parent: `COMPLETED 0:0`, elapsed `01:29:01`.
- allocation: server1 GPU 4 / CPU 32 / `260000M`.
- controller child 4개와 evaluator child 4개가 모두 `COMPLETED 0:0`.
- 네 model×family worker를 동시에 시작했다.
- observed maximum: host RSS 약 `16.06 GiB`, GPU reserved 약 `46.59 GiB`; child
  `65000M`과 A6000 allocation 안이다.
- GH direct submit 예외는 server1 SH 부재와 사용자 time-critical 지시에 한정했고,
  영향은 job `15891`과 c3 ignored local namespace뿐이다.

## Controller/evaluator integrity

- controller 8/8 `all_pass=true`, expected count exact.
- evaluator 8/8 `all_pass=true`, checkpoint `32/32`, technical false 0.
- QP 16 edits/64 hops; exact four-hop/total-D contract 통과.
- max path/hop/share/radial error:
  `1.802e-16 / 2.715e-16 / 2.220e-16 / 1.388e-17`.
- allocation cap violation 0; positive overload `7/7` zero-suppressed.
- first edit/round c1 coefficient와 c3 allocation coefficient는 네 cell 모두 max abs
  diff 0이고 native distance 동일.
- trust diagnostic false는 Qwen MEMIT 1회이나 trust는 fixed-distance c3에서
  diagnostic-only이며 evaluator technical contract를 바꾸지 않는다.

## Firewall·read-only

- 네 controller terminal/hash barrier 뒤 별도 evaluator process만 outcome field를
  load했다.
- direct-z branch/edit당 1회, raw evaluation text/logit/token persistence false.
- compact artifact의 NFE key 0.
- covariance 5/5, Alpha projector/history, Wikipedia artifact는 pinned precomputed
  read-only로 재사용했다.
- EasyEdit source/global cache/dataset/model weight를 수정하지 않았다.

## Artifact integrity

- controller/evaluator summary declared SHA-256과 실제 file hash `48/48` 일치.
- fresh model/pair analyzer output은 canonical JSON과 `3/3` byte-identical.
- Llama analysis:
  `fdf52b7c5f1db78d1a12e8092988816bccbadffdab64d9842f401673ab43765f`.
- Qwen analysis:
  `45b2e7b807d7a7622aea7c4843259b75d109124d01f85bd1cc8073bb5785637a`.
- Pair analysis:
  `cf9c320fff1492a2d9854de0c57811cdc5788fda152a4b80d3b21b04fb1e00d6`.
- compact 29-file aggregate:
  `bc2d3da7fe3cb7ae0859e72e6bcdb144c53f0307f0d013be5bacc511d4203127`.
- raw factors/receipts/checkpoints/logs는 ignored `local/`에만 있다.

## Scientific audit

| Model | Family | c1 current | c3 current | recovery | strong floor |
|---|---|---:|---:|---:|---|
| Llama | MEMIT | `-4.032392` | `-0.361365` | `+3.671027` | fail |
| Llama | Alpha | `-2.495123` | `-0.563717` | `+1.931406` | fail |
| Qwen | MEMIT | `-0.397837` | `-0.251563` | `+0.146274` | fail |
| Qwen | Alpha | `+0.029707` | `+0.555242` | `+0.525536` | pass |

Pre-registered lenient causal/directional gate는 pass하고 strict analyzer/native gate는
fail한다. 이 차이를 숨기지 않으며 positive claim은 implementation cause와
Motivation direction에만 한정한다.

## Agent gate

관련 파일만 허용한 Terra Ultra Llama/Qwen/pair-red agents 3개는 runtime metadata를
검증하지 못해 무열람 `BLOCK` 종료했다. 독립 analysis/review로 세지 않고 GH fallback은
위 claim boundary 안에서만 사용했다.

## Git/protocol/artifact broadcast

- 실행 manifest의 commit은 `bbd3d0e`, tracked worktree clean true다.
- Git에는 code/test/spec/report/audit/hash만 두고 credential/private connection/raw
  artifact를 추가하지 않는다.
- active peer SH/clone이 없어 artifact broadcast는 no-peer exception으로 미실시한다.
- direct-z temporary session은 모니터링·artifact 결합·job 조정하지 않았다.

## 최종 audit 판정

c3는 기술적으로 유효하며 c1 under-update가 cross-model implementation cause였음을
확인했다. Motivation은 directional-positive로 종료한다. Strong method gate가 남았으므로
현재 결과를 deployable/lifelong superiority로 사용하지 않는다.

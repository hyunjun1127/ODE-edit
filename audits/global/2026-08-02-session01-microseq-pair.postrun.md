# Session 01 Motivation — microseq pair post-run audit

날짜: 2026-08-02

판정: **technical PASS / scientific `MICROSEQ_HARM_SIGNAL`**

## 실행 lineage

- job `15798`: `FAILED`, pre-action contract naming collision; Llama actions 0,
  receipts 0, evaluator load false; Qwen fail-fast. ignored local failure archive에
  보존하고 과학 분석에서 제외.
- repair commit: `2fe2bfe`; exact boolean attestation 2개만 value-locked 허용,
  targeted/compile/full `290 tests OK`.
- job `15799`: `COMPLETED 0:0`, 2026-08-02 03:44:13--05:17:00 KST,
  2 GPU / 16 CPU / 130000M.

## Exact artifact gate

| Artifact | SHA-256 |
|---|---|
| native Llama summary | `4c98e6672f6904c77cfe0b02eac70c23ec7118d52891df2f9a259cc050daf4f4` |
| native Qwen summary | `82b0c5beee4473ac9765e0956799e27457f269cf6a3e825d06902a2514df2476` |
| ODE Llama summary | `feb1af5ad6248e8b71fb50e1f5bba317f027b1bfb4aebcd11ad07b3d99fd3de2` |
| ODE Qwen summary | `1c5ffa8a5e83079264af6cfa38ac78150aad09b81425c8ccb84bdb3d8903183f` |
| combined Llama analysis | `92abde172ee457124b35a8437fa668e55b86f4f529b934a82d98eb00d6ce46b3` |
| combined Qwen analysis | `c1dca51e6ffbbae274cbbad78284072ba170f9eef7249764a8a29f0735b5f579` |
| pair analysis | `c3e21c22b0447c084dcfb72186629ac31824d9c314b2b0841658399d08e80d7c` |

- controller별 exact features/actions/events 4, receipts 4, target artifacts 4.
- native proposal artifacts 4/model, ODE proposal artifacts 16/model.
- evaluator별 checkpoints 4, combined 8/model.
- all controller/evaluator summaries `completed/all_pass`, Git write false.
- raw evaluation text/logits/token IDs persistence false.
- fixed contract와 case order/policy/evaluator hash는 양 model/branch byte-identical.
- deterministic analyzer replay: 세 JSON 모두 byte-identical `PASS`.
- stderr 4424 B는 checkpoint shard load/deprecation 출력뿐이며 traceback/OOM 없음.

## Resource

- controller peak GPU allocated: Llama native 40.71 GB, ODE 43.07 GB;
  Qwen native 43.81 GB, ODE 47.51 GB.
- evaluator peak GPU allocated: 약 31.96--33.10 GB.
- host max RSS는 child별 약 7.9--17.4 GB로 65000M child cap 안이다.
- pair wall 1:32:47, Slurm resource cap 위반 없음.

## Red/blue 결과

- blue: 두 model 공통 capacity와 neighborhood KL reduction signal 존재.
- red: current-edit noncollapse와 retention이 양 model 모두 실패하므로 pass로
  평균내거나 lenient하게 승격할 수 없다.
- requested Terra Ultra model agents는 runtime metadata를 검증할 수 없어
  무열람 종료했다. GH provisional report가 이를 대체하며 독립 agent review로
  주장하지 않는다.

## Artifact broadcast

server1에는 SH가 없고 server4는 `registered-pending-clone`이며 전용 Codex SH
session/clone이 없다. 따라서 raw artifact rsync 대상이 없어 no-peer exception을
적용한다. 임의 SSH/rsync는 실행하지 않았다.

- 최종 판정: `PASS` — technical evidence는 Motivation closure에 사용 가능;
  scientific verdict는 `MICROSEQ_HARM_SIGNAL` 그대로 보존

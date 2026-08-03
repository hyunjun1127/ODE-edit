# Session 02 P0 technical pair — GH post-run audit

- 작성 시각: **2026-08-03T18:03:44+09:00**
- 대상 jobs: Llama `16025`, Qwen `16026`
- 대상 instruction: `ODEEDIT-S02-P0-TECH-PAIR-V1`
- audit 판정: **`block`**
- 허용되는 다음 상태: **CPU/synthetic RCA와 repair만 가능; GPU retry/P1/main table 불가**

## 필수 post-run gate

| 항목 | 판정 | 근거 |
|---|---|---|
| source/lock identity | pass | SH1이 execution head와 proposal ID 일치를 preflight에서 확인 |
| paired resource cap | pass | 제출 전 `0+0+2 <= 4`, 각 job 1 GPU |
| offline load/manifest | pass | 두 모델 모두 shards 4/4 및 manifest 완료 보고 |
| combined event identity | pass | 두 모델 모두 locked tolerance 통과 보고 |
| hook/reference identity | block | Llama Full warm-up에서 명시적 contract error |
| peak-memory feasibility | block | Qwen Static field build CUDA OOM |
| terminal artifact schema | block | terminal manifest와 compute/controller JSONL 미완성 |
| scientific metric validity | not applicable | evaluation/generation 0건 |
| model-common success | block | 두 모델이 서로 다른 technical blocker로 terminal failure |
| automatic retry discipline | pass | retry/source/parameter 변경 0건 |
| artifact preservation | provisional pass | SH1이 cleanup 없이 보존했다고 보고; hash report pending |
| artifact broadcast | pending | active peer readiness와 exact path/hash 확인 뒤 broadcast 또는 예외 기록 필요 |

## Red-team 공격 점검

- **Outcome leakage:** scientific evaluation이 열리기 전에 실패했으므로 outcome을 보고 repair
  rule을 고를 경로가 없다. repair는 numerical identity와 memory만 사용해야 한다.
- **Gate laundering:** hook mismatch를 tolerance 완화로 통과시키거나 OOM을 모델별 layer/config
  축소로 피하면 common-method contract를 훼손한다. 둘 다 금지한다.
- **Parser/partial artifact misuse:** non-terminal direct-z 및 arm entry artifact는 완료 metric이
  아니다. 빈 JSONL을 0 성능이나 0 비용으로 해석하지 않는다.
- **Resource waste:** 두 job 모두 초기 약 5분 내 fail-closed해 큰 GPU 낭비는 막았다. 그러나
  동일 원인으로 재제출하면 resource-waste gate 실패다.
- **Hidden compute:** 새 scalar-gate reference를 warm-up 비용에서 숨기면 안 된다.
  `N_reference_gate_fwd`, `N_reference_gate_bw`, wall/GPU time을 분리 기록해야 한다.
- **Model rescue:** Qwen에만 다른 solver/dtype/layer set을 쓰거나 Llama에만 다른 tolerance를
  쓰는 순간 method-common gate가 실패한다.
- **Scientific overclaim:** 이 pair는 ODE vector field나 allocation 가설의 반증도 지지도 아니다.

## Agent/runtime audit

GH가 세 개의 독립 review subagent를 runtime-gate-only로 시작했으나 모두
`gpt-5.6-terra/ultra` 불일치로 `BLOCKED_RUNTIME_MISMATCH`를 반환했다. 세 agent는 repository와
report를 읽지 않고 즉시 종료됐으며, Sol 결과로 대체하지 않았다. 따라서 별도 Terra Ultra
model-result interpretation은 여전히 pending이고, 이번 zero-outcome technical 판정에는 agent
해석을 사용하지 않았다.

## Artifact/Git boundary

- raw stdout/stderr, model outputs와 partial artifacts는 Git에 넣지 않는다.
- 기존 failed output root는 수정·삭제하지 않고 retry는 별도 root를 사용한다.
- EasyEdit source, covariance/null-space/Wikipedia artifacts는 read-only이며 재계산·download를
  허용하지 않는다.
- ordinary project artifact는 protocol helper로 active peer에 broadcast해야 한다. 현재는 SH1의
  exact manifest/hash와 peer readiness 확인이 끝나지 않아 `pending`이다. 불가능하면 SH1이
  no-broadcast exception과 이유를 completion report에 기록해야 한다.

## 최종 red 판정

`block`은 연구 kill이 아니라 현재 implementation과 execution progression에 대한 block이다.
두 독립 root cause가 outcome-free CPU/synthetic evidence로 닫히고 GH가 diff와 revised hard gate를
승인하기 전까지 P0 retry, P1, scientific arm과 main table 제출을 허용하지 않는다.

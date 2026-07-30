# Qwen MV-0 c3 v3 독립 분석

- 분석일: 2026-07-30
- 대상 run: `mv0_qwen_c3_v3`
- Slurm parent job: `15517` (`odeedit_mv0_pair_c3v3`)
- 판정: **`PASS` — Qwen 3-case MV-0 implementation fidelity에만 유효**

## 확인 범위

실행 담당과 분리하여 다음 자료만 읽었다.

- `local/results/raw/session01_motivation/mv0_qwen_c3_v3/manifest.json`
- `local/results/raw/session01_motivation/mv0_qwen_c3_v3/events.jsonl`
- `local/results/raw/session01_motivation/mv0_qwen_c3_v3/summary.json`
- paired c3 v3 execution preflight
- Motivation Validation plan의 MV-0 section
- `sacct -j 15517`

raw prompt, raw target, raw generation은 읽지 않았고 artifact에도 저장되지
않았다.

## repo/artifact에서 확인한 사실

### 실행 완결성과 case accounting

| 항목 | 관측값 | 판정 |
| --- | ---: | --- |
| planned / attempted / pass | `3 / 3 / 3` | 통과 |
| failure / abort 후 미실행 | `0 / 0` | 통과 |
| run status | `completed`, `all_pass=true` | 통과 |
| event 수와 sequence | 3개, `0,1,2` | 통과 |
| case 순서 | `18447, 3176, 15669` | manifest와 일치 |
| direct-z artifact 수 | 3 | case 수와 일치 |

세 event 모두 동일한 base-state hash
`b32ded1ea35e556ecece098b242e863c4e0db02aed83e20796a0d577fad2f66e`
에서 시작했고 `rollback_exact=true`였다. Summary의
`all_rollbacks_exact=true`와 일치한다.

### native–adapter fidelity와 neutrality

세 case 모두 다음 조건을 만족했다.

- 5개 layer 각각 factor의 `c_cosine=1.0`,
  `relative_c_norm_error=0.0`
- materialized delta의 `exact_bytes=true`,
  `max_abs_error=0.0`, `relative_l2_error=0.0`
- final delta의 모든 layer hash가 같고 relative/max error가 0
- teacher-forced logits hash가 같고 logits relative/max error 및
  NLL/context-NLL 차이가 0
- native와 bridge의 rewrite NLL reduction 차이가 0
- trace-only 경로가 모든 layer를 호출하면서 weight를 바꾸지 않았고,
  trace-off와 logits/NLL가 exact-equal
- self-replay도 layer delta와 teacher-forced output에서 exact-equal

calibration bound는 `torch.float32`에서 absolute/relative 모두
`3.814697265625e-06`이다. 관측된 primary error의 case별 최대값은
모두 0이므로 bound 안이다. MV-0 plan이 요구한 paired difference의
세 표본 자체가 모두 0이라는 점에서는 Qwen-side equivalence gate를
통과한다.

다만 persisted schema에는 aggregate equivalence CI의 양 끝점이 별도
필드로 남아 있지 않다. 따라서 이 보고서는 CI 계산 구현 자체를
재검증하지 않고, 저장된 paired scalar가 전부 exact zero임만 검증한다.

### provenance와 artifact binding

- model: `Qwen/Qwen2.5-7B-Instruct`
- model/tokenizer revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`
- provenance ID:
  `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16`
- selection manifest ID:
  `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce`
- context manifest ID:
  `e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd`
- ODE-Edit commit:
  `b4fe15d1891952bed1968ef1190b9363335b19cb`
  (`tracked_worktree_clean=true`)
- Slurm binding: job `15517`, name `odeedit_mv0_pair_c3v3`,
  node `devbox`

Manifest와 summary의 run/model/provenance/selection/context/Slurm identity가
서로 일치했다. 실제 file hash도 summary가 선언한
`manifest_sha256=f10fb97a...17f652`,
`events_sha256=5bc927fa...055a4`와 일치했다.

5개 covariance file(layer 4–8)은 `verified-read-only` 정책 아래
로드됐고 case마다 solve count가 5였다. Projector는 로드되지 않았으며
manifest 정책도 hash/size 검증만 하고 deserialize하지 않는 것으로
기록한다. Direct-z는 case마다 서로 다른 hash의 local frozen artifact로
1회 고정되었다. 실행은 offline이고 Git output은 쓰지 않았다.

### Slurm 및 resource

`sacct`에서 parent job과 두 child step은 모두 `COMPLETED 0:0`이다.
Preflight의 child mapping상 Qwen은 step `15517.1`이며,
`2026-07-30T21:56:36`부터 `22:06:18`까지 GPU 1, CPU 8,
memory `65000M`으로 실행됐다. Llama step `15517.0`도 같은 초에
시작했으므로 두 model의 실제 실행 구간은 중첩됐다. Parent allocation은
GPU 2, CPU 16, memory `130000M`이고 총 elapsed는 `00:09:43`이다.

Qwen summary 기준:

- wall time: `578.409 s` (약 9분 38초)
- visible GPU: A6000 1개, total 약 `47.40 GiB`
- peak allocated: 약 `40.80 GiB` (total의 약 86.1%)
- peak reserved: 약 `43.59 GiB` (total의 약 92.0%)
- process-side host max RSS: 약 `16.53 GiB`
- Slurm step MaxRSS: 약 `14.76 GiB`

host RSS 두 값은 수집 범위가 다른 계측치이므로 서로 대체하지 않고 함께
기록한다. GPU peak reserved는 cap 안이지만 여유가 약 8%라서 동일
Qwen 실행의 GPU당 추가 상주 tensor를 크게 늘리면 OOM 위험이 있다.

## 비차단 metric 주석

일부 layer의 `materialized_delta.cosine`은 exact-byte equality인데도
`1.0055–1.0173`으로 1을 넘는다. 이는 수학적으로 정규화 cosine의
범위를 벗어나므로 해당 cosine scalar는 독립적인 정량 근거로 사용하면
안 된다. Exact-byte flag, hash equality, zero max/relative error가
동일성을 직접 입증하므로 이번 fidelity 판정을 뒤집지는 않는다.
후속 단계에서 cosine을 비교 metric으로 쓸 경우 float64 재계산 또는
명시적 수치 검증이 필요하다.

## 독립 판정과 해석 한계

Qwen c3 v3는 native singleton `execute_memit`과 adapter canonical
mode 사이의 3-case 구현 fidelity, trace neutrality, exact rollback을
통과했다. 따라서 **Qwen MV-0 implementation fidelity는 `PASS`**다.

이 판정은 다음을 의미하지 않는다.

- Llama까지 포함한 pair 전체 gate의 독립 통과
- sequential editing의 안정성 또는 retention
- MV-1 이후의 predictive heterogeneity/nonstationarity
- ODE-Edit의 motivation 성립
- baseline 대비 method gain 또는 기대 개선폭

표본은 deterministic singleton edit 3건이며 target token count도 모두
1이다. Motivation 신호와 기대효과는 held-out MV-1 이후 diagnostic에서
별도로 추정해야 한다.

# Session 01 MV-0 Qwen exact one-shot 실행 gate

- 작성일: 2026-07-30
- 작성자: `head-server1-gh` (`global-head`)
- 선행 run: `mv0_llama_smoke_v1`, Slurm job `15500`
- 최종 판정: `PASS` — 아래 exact MV-0 smoke 한 건에만 유효
- 허용 run: `mv0_qwen_smoke_v1`
- 허용 job: `odeedit_mv0_qwen_smoke`

## 1. 네 범주

### Proposal에서 온 내용

- 두 고정 model에서 native EasyEdit MEMIT과 repo-local exact adapter의
  fidelity/trace neutrality를 먼저 확인해야 한다.

### Repo/protocol에서 확인한 사실

- Llama 1-case smoke는 job `15500`, `COMPLETED`, exit `0:0`,
  `all_pass=true`, planned/attempted/pass `1/1/1`이었다.
- 독립 result analysis와 post-run red audit가 그 Llama smoke를 PASS했고,
  Qwen 전용 audit 작성 단계로 진행을 허용했다.
- Qwen fixed provenance ID는
  `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16`,
  selection manifest ID는
  `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce`다.
- server1 cap은 GPU 3, requested GPU당 host memory `198117 MiB`다.

### GH 추정

- Llama의 peak GPU reserved 약 42.69 GB와 host RSS 약 11.2 GiB는 Qwen
  resource 보장이 아니지만, 같은 1 GPU/65000M envelope로 Qwen one-case
  smoke를 시도할 합리적 사전 근거다.

### 사용자 확인 필요

- 없음. 사용자는 최소 필수 감사 후 실험을 빠르게 제출하라고 명시했다.

## 2. 최소 선행 gate

- `experiment-reports/global/2026-07-30-mv0-llama-smoke-analysis.md`: PASS
- `audits/global/2026-07-30-mv0-llama-smoke.postrun.md`: PASS
- canonical CPU suite: `Ran 39 tests`, `OK`
- model-free Qwen artifact preflight: PASS
- shell syntax: PASS
- EasyEdit source/stats/projector는 read-only fixed hash/size 검증만 허용
- existing Qwen Wikipedia moments 5개만 사용; miss/recompute/download는 fatal
- AlphaEdit projector는 hash/size만 검사하고 deserialize하지 않음

Runner는 Slurm 실행 시 `SLURM_JOB_ID`, exact job name, node를 manifest와
summary에 기록한다. partial/mismatched Slurm identity면 model load 전에
중단한다.

## 3. Exact envelope

| 항목 | 값 |
| --- | --- |
| target | `server1` / `devbox` |
| allocation | GPU A6000 1, CPU 8, host memory `65000M` |
| time | `06:00:00` |
| model | `qwen2.5-7b-inst` |
| model revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| cases | 1 |
| seed | 17 |
| run ID | `mv0_qwen_smoke_v1` |
| job name | `odeedit_mv0_qwen_smoke` |
| raw output | `local/results/raw/session01_motivation/mv0_qwen_smoke_v1/` |
| one-shot marker | `local/state/slurm-submissions/session01_motivation/mv0_qwen_smoke_v1.submitted/` |

허용 명령은 다음 하나다.

```bash
project/run_scripts/submit_session01_mv0_server1.sh \
  qwen2.5-7b-inst mv0_qwen_smoke_v1 1
```

공통 helper와 in-job envelope는 Llama/Qwen의 두 exact triple만 허용한다.
Llama marker가 이미 존재하므로 Llama 재시도는 차단된다. Qwen helper는 이
audit와 두 Llama 독립 post-run 문서가 tracked이고, untracked를 포함한 repo
전체가 clean하며 `HEAD == origin/main`일 때만 진행한다. 이어서 session
boundary, live GPU/memory cap, exclusive output/marker/job name을 검사한 뒤
atomic marker를 만들고 `sbatch --export=NONE`을 한 번 호출한다.

## 4. 중단 및 후속

Submit 전 Git/session/resource/marker/audit gate가 하나라도 실패하면 job을
만들지 않는다. Allocation 후에도 source/data/cache/runtime/Slurm identity
preflight가 실패하면 model을 load하지 않는다.

Runtime에서 provenance, model/tokenizer revision, one-visible-GPU, float32,
offline, context/direct-z/request/state, covariance no-recompute, rollback,
finite metric, artifact firewall 중 하나라도 실패하면 `all_pass=false`다.

성공 조건은 scheduler `COMPLETED/0:0`, planned/attempted/pass `1/1/1`,
`all_pass=true`, exact rollback, native-adapter/trace equivalence, compact
artifact firewall이다. 결과가 나오면 실행에 참여하지 않은 별도 analysis
agent와 별도 post-run red agent가 manifest/summary를 검토한다. 이 Qwen
smoke가 PASS해도 Motivation claim이나 MV-0 전체를 승인하지 않으며, case
확장에는 새 gate가 필요하다.

GH 직접 제출은 사용자의 명시적 실행 지시와
`PROTOCOL.md:506-509` 예외에 근거한다. 다른 SH/peer clone이 없으므로
artifact broadcast는
`exception=no_active_peer_clone_or_sh`로 기록한다.

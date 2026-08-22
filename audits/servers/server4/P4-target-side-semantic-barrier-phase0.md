# P4 target-side semantic barrier — server4 Phase 0 audit

## 판정

- instruction: `ODEEDIT-S05-P4-TARGET-SIDE-SMOOTH-SEMANTIC-LOGODDS-BARRIER-V1`
- server/agent: `server4` / `server4-server-head`
- observed_at: `2026-08-22T19:08:56+09:00`
- authoritative contract: `FULL_READ_PASS`
- core tensor/control-plane implementation: `PASS`
- pre-GPU readiness: `BLOCKED_READINESS`
- exact blockers:
  - `BLOCKED_HF_MAPPING`
  - `BLOCKED_EVALUATOR_MAPPING`
  - `BLOCKED_SEALED_STREAM_TRANSFER`
- model load / GPU use / Slurm submit: `0 / 0 / 0`
- scientific submit: `HOLD`

## Authoritative contract gate

`local/state/p4-target-side-semantic-barrier-v1/authoritative-contract.txt`를
독립 재검증하고 1–513행 전체를 읽었다.

- type: regular non-symlink
- mode: `0600`
- bytes: `13744`
- lines: `513`
- SHA256:
  `0e6b0ad5110ba8bc758a92ffa126ab094afff57aa18223c4f2d1d16de1170611`
- envelope conflict: 없음

## Source-backed reuse map

| P4 경계 | 재사용 source/API | 판정 |
| --- | --- | --- |
| model activation intervention 및 context/order | `scalable_batched_model.OrdinalTargetActivationOverlay`, scalable plan | reuse |
| native-compatible KL/teacher | `p1r24_atomic_strength.build_p1r24_kl_plan`, `evaluate_p1r24_kl` | reuse |
| alias별 KL/decay/clamp | `P1R24AliasTargetLock` | exact pinned values |
| Official writer | `scalable_batched_native.run_official_native_apply` | direct entrypoint |
| accepted-z bridge | `p1r52_target_official_alphaedit_writer.accepted_z_cache_template` | optional source label만 추가; legacy default 유지 |
| EasyEdit deployment | `agents/server4/alphaedit-runtime-path-seal.json` | EasyEdit mapping PASS |

기존 `p1r52_target_depth`의 PRIMARY/RESCUE/CURRENT, early stop 및 best/accept
selection은 authoritative contract §5의 금지항목과 충돌하므로 P4 solver에는
재사용하지 않았다. 하위 KL/teacher/activation/writer API만 재사용한다.

Pinned AlphaEdit `compute_z.py`와 hparams를 read-only 추적한 결과 Adam은
`torch.optim.Adam` 기본 betas/epsilon과 alias별 `v_lr`를 사용한다.

- Llama: lr `0.1`, KL `0.0625`, decay `0.5`, clamp `0.75`
- Qwen: lr `0.5`, KL `0.0625`, decay `0.001`, clamp `4.0`

## 최소 구현

- `p4_semantic_barrier.py`: V+/V± exact potential, softplus/sigma,
  new/old NLL·margin과 gradient norm/cosine telemetry
- `p4_fixed_target_solver.py`: request-independent full-FP32 Adam,
  fixed M=5/all iterations/final-only, outer moment reset, origin clamp telemetry
- `p4_waypoint_writer.py`: K=8 waypoint, current-state refresh call contract,
  Official apply adapter 및 ours/native `compute_z` call-count binding
- `p4_cache_transaction.py`: batch-entry snapshot K1–K8, append0,
  successful K8 후 append1, failure rollback/append0
- `p4_panel_analysis.py`: ZA/ZB same-entry pairing과 factual `Z±-Z+` / `A±-A+`
- numerical lock root:
  `a7e8d51b0fcc8ac7576d4d16346f415fdca5c98904753cdcc2e9133dc6ba92c6`
- focused source manifest root:
  `bda8f5b204f8edc227f68127d370b2cd8c54beba40b0a895562e5ab849f0092b`
- source manifest SHA256:
  `74d198e89c6bdb22a569e1e9109f2259efec0586003960b6a1765607b26c0817`

## Focused gates

- `py_compile`: PASS
- P4 focused unittest: 13/13 PASS
  - analytic/reference formula within FP32 `1e-6`, new/old gradient signs
  - Z+ old gradient authority 0
  - full-FP32 fail-close
  - fixed 5-step manual requestwise Adam = `torch.optim.Adam`
  - final-only/moment-reset/no early stop
  - K8 waypoint and refresh counts
  - Official writer ours/native compute-z counts
  - cache snapshot/append1/rollback negatives
  - paired panel and rooted numerical lock
- reuse regressions: 34/34 PASS
  - P1R52 target-depth 19
  - Official writer 8
  - runtime path seal 7
- dry-plan: PASS with three readiness blockers; model/GPU/submit false
- `git diff --check`: PASS

## Readiness blockers

### HF mapping

`/data/janghj/.cache/huggingface/hub`의 pinned revision과 locked required
members는 이전 S4-M1-R1 audit에서 target/size/digest MATCH했다. 그러나 current
`P0ArtifactGuard` exact snapshot set에는 포함되지 않은 extras가 있다.

- Llama: `.gitattributes`, `LICENSE`, `README.md`, `USE_POLICY.md`, `original`
- Qwen: `.gitattributes`, `LICENSE`, `README.md`

extras를 삭제·이동하지 않았고, 별도의 locked closure/loader influence receipt가
아직 없으므로 HF mapping은 fail-close 유지한다.

### Evaluator mapping

server4 bounded 후보 중 legacy lock의 세 source SHA를 모두 만족하는 canonical
root는 없다. `/data/janghj/SUIT`는 `experiments/summarize.py`만 MATCH하고 두
evaluator source는 MISMATCH다. 다른 bounded candidate도 MISMATCH이므로 mapping을
열지 않았다.

### Sealed stream

예정 destination
`local/state/p4-target-side-semantic-barrier-v1/sealed-stream`은 현재 없다.
샘플 재추출·재배열·생성은 수행하지 않았다.

## Resource snapshot

- `/data` available: `207415042048` bytes, use `98%`
- host available memory: `437642498048` bytes
- server4 Slurm: 다른 user의 2-GPU job 1개 RUNNING; SH4 project job 0
- physical GPU 0–1 occupied, 2–7 idle at observation
- `PROJECT_GPU_CAP=2` 유지

공간과 host memory 자체는 pilot의 즉시 치명적 blocker가 아니다. 다만 위 세
identity/readiness gate가 닫히기 전 model load 및 pilot submit은 금지 상태다.

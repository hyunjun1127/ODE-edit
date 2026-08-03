# Session 02 numerical-lock revision focused audit — server1/SH1

## Scope

Audit target은 local commit
`e6fa39548224e92244e7d02118a1375e47b2431e`의 ODE-Edit-side concrete backend,
fair-compute instrumentation, numerical-lock v3, executable P0 runner와 no-submit sbatch
template다. GPU/model load, Slurm, scientific output, push, EasyEdit write와 direct-z temporary
track은 scope 밖이며 실행하지 않았다.

## Gate 결과

| Gate | 결과 | 근거 |
| --- | --- | --- |
| canonical SH1/session/repo boundary | PASS | session `019fc63e-5217-7250-9c22-c5b2ec4248f0`, dedicated branch/worktree, effective server-head identity |
| allowed staged paths | PASS | `scripts/check-agent-access.sh --staged` |
| worktree after implementation commit | CLEAN | `e6fa395` 직후 porcelain 0 |
| EasyEdit write/download/recompute | PASS by boundary | read-only calls만 사용; foreign dirty checkout을 stage/modify하지 않음 |
| fixed-artifact CPU preflight | PASS | 양 모델 각 25 files full hash/size, selected request identity 확인 |
| common concrete backend | PASS (implementation) | 두 model alias가 동일 `EasyEditMemitBackend` class/schema 사용 |
| loaded model/tokenizer/context preflight | NOT RUN | GPU=0 revision envelope; P0 technical gate로 유지 |
| direct-z once/freeze | PASS (CPU contract) | lazy once guard, byte/artifact identity와 first-hit tests |
| Native entry-hit freeze | PASS | entry-hit에서 z/proposal/write 0 regression |
| same-snapshot Full/One-refresh | PASS | current-state proposal identity와 accept 뒤 rebuild tests |
| Static/Ordered identity | PASS | frozen entry direction/share 및 ascending current coordinate/revisit tests |
| combined event batching | PASS (CPU fixture) | right-padded one-forward와 two-forward `1e-10` identity |
| information firewall | PASS | CounterFact rewrite-only projection; evaluation mapping 거부 |
| actuator hook / dense target grad 0 | PASS (CPU) | hook/oracle identity, one backward, `.grad is None`, pointer/version unchanged |
| functional dense weight copy 0 | PASS (implementation) | low-rank output hook; mutable dense trial은 reference-only |
| Scalar hidden dense read 0 | PASS | probe 후 exact checkpoint rehash 제거, copy-free state guard regression |
| one entry checkpoint / no per-accept backup | PASS | transaction API와 injected mid-commit rollback test |
| exact rollback and RNG cleanup | PASS (CPU) | target weights, Torch/Python/NumPy RNG와 backend state identity |
| reject field rebuild 0 | PASS | same field/direction/QP inputs, trust-only retry test |
| precomputed covariance refresh reuse | PASS (CPU contract) | full-hash setup, stat/sample guard, in-memory pointer/version cache test |
| QP coefficient exact apply | PASS | solver/applied tuple equality, no post-rescale path |
| Omega terminal net once | PASS | subdivision invariant terminal C-energy와 append ordering |
| fair counters/timers | PASS (CPU schema) | actual model hook, logical/forward separation, controller/setup/eval timers |
| Native/Scalar proposal timing | PASS (implementation) | `native_proposal`, proposal/sweep counters와 controller scope |
| post-first-hit work 0 | PASS | recorder freeze와 every-arm entry-hit regression |
| artifact schema/firewall | PASS | required fields/hash manifest; raw prompt/target/context 미저장 |
| cached graph policy | PASS | `UNSUPPORTED_FAIL_CLOSED`; common no-grad backend only |
| focused CPU suite | PASS | 46/46 tests |
| paired P0 dry plan | PASS | 2 jobs/2 GPUs forecast, submit=false |
| GPU numerical/resource identity | NOT RUN | 별도 GH P0 envelope 필요 |
| Terra Ultra audit | PENDING | `TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH` |

## SH red-team corrections

1. Scalar search가 각 functional probe 뒤 exact entry checkpoint를 다시 해시하면 target weights를
   반복적으로 GPU에서 읽고 그 비용도 trial timer 밖에 남는다. 최종 구현은 functional hook의
   storage/version/RNG guard와 backend state ID만 검사한다.
2. Verified covariance를 매 adaptive field에서 EasyEdit cache miss 경로로 다시 열면
   multi-GiB artifact decompression이 controller compute를 지배할 수 있다. 최종 구현은 setup에서
   full-hash 검증·로드한 tensor를 backend-owned cache로 재사용하며 source tensor
   pointer/version을 검사한다.
3. Backend constructor의 exact initial snapshot hash가 editor와 setup 양쪽에서 누락될 수 있었다.
   P0 runner는 이를 per-run `context_setup` component에 포함해 amortized sec/edit에 기록한다.
4. Summed component GPU events는 nested region을 이중계산할 수 있다. 최종
   `controller_gpu_seconds`는 controller start/end CUDA event 한 쌍으로 측정하고 component는
   진단 breakdown으로만 사용한다.
5. Native public bridge의 verified covariance preflight reuse는 ODE-side module global을
   일시 교체한다. Global lock과 `finally` restoration test는 PASS했지만 actual P0 process에서
   복원과 source manifest를 다시 확인해야 한다.

## Unresolved verification

- CPU fixture는 bf16/fp16 model-scale event, hook, functional/committed identity를 증명하지
  않는다.
- P0 실제 `N_model_fwd`, GPU time, peak memory와 terminal dense geometry 비율은 아직
  관측하지 않았다. Resource 수치는 forecast다.
- P0에서 `D_sync_entry`가 degenerate하거나 `h0/D_native`가 `[1/8,1/2]` 밖이면 P1 HOLD다.
- Terminal geometry가 controller wall/GPU time의 10%를 초과하면 P1 전에 accumulated
  low-rank cross-term evaluator와 dense identity gate가 필요하다.
- Entry-hit `N_z=0`은 first-hit freeze 요구와 일치하지만 canonical spec의 일반
  `N_z=1/edit` 문구와 명시적 GH reconciliation이 필요하다.
- EasyEdit checkout의 foreign dirty state 때문에 repository-wide clean을 주장하지 않는다.
  SH1 authored EasyEdit change는 0이다.
- Terra runtime mismatch가 남아 있어 실제 결과 생성 시 별도 runtime check와 analysis HOLD
  policy가 적용된다.

## 결론

Audit conclusion은
`IMPLEMENTATION_REVISION_PASS; NUMERICAL_LOCK_PENDING_GH; SCIENTIFIC_EXECUTION_NOT_AUTHORIZED`다.

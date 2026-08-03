# Session 02 implementation-prep focused audit — server1/SH1

## Scope

Audit target은 local commit `9792597f2f43d6482468ebf2d79a82fbcb253b9b`의
ODE-Edit-side implementation, outcome-free lock proposal와 dry-plan launcher다. GPU/Slurm,
scientific output, EasyEdit write와 direct-z 임시 track은 audit scope 밖이며 실제로 실행하지
않았다.

## Gate 결과

| Gate | 결과 | 근거 |
| --- | --- | --- |
| session/repo/role boundary | PASS | SH1 session, dedicated worktree, effective worktree identity 확인 |
| allowed staged paths | PASS | `scripts/check-agent-access.sh --staged` |
| EasyEdit write / artifact recompute-download | PASS (0) | read-only reference만 사용; 외부 dirty state 미변경 |
| common model-independent schema | PASS | 단일 lock/config; model-specific branch/rescue 변조 test |
| direct-z once/edit | PASS | runtime invariant 및 counter test |
| same-snapshot Full/One-refresh | PASS | accept 뒤 current snapshot field identity test |
| Static entry freeze | PASS | field build 1회, direction ID 고정 test |
| Ordered identity | PASS | ascending current coordinate, single nonzero coefficient, revisit test |
| scalar bracket / alpha cap | PASS | increasing grid, first bracket bisection, `[0,1]` lock |
| event length normalization | PASS | unequal object-token length CPU test |
| information firewall | PASS | rewrite-only request type; evaluation mapping 거부 |
| one backward / target dense grad 0 | PASS | actuator-hook/reference identity, one grad call, `.grad is None` |
| dense trial weight copy 0 | PASS (implementation) | read-only functional hook; runtime trial/commit API 분리 |
| exact rollback/failure cleanup | PASS (CPU contract) | weight/storage/RNG, accepted-write injection, pre-Omega restore tests |
| reject field rebuild 0 | PASS | 1 field, 2 trials, same state/direction test |
| QP coefficient exact apply | PASS | solver/applied tuple exact equality; no post-rescale path |
| Omega outer-edit once | PASS | terminal net-write subdivision invariant/receipt guard |
| post-first-hit work 0 | PASS | entry-hit field/trial/write 0 및 recorder freeze |
| compute accounting | PASS | 9 counters, component/controller timer, reject reuse test |
| cached graph policy | PASS | current MEMIT `UNSUPPORTED_FAIL_CLOSED`; no scientific arm |
| raw/log/weight/dataset Git exclusion | PASS | staged paths/source payload inspection |
| CPU focused suite | PASS | 35/35 tests |
| P0/P1 dry launcher | PASS | explicit `--dry-run`, paired models, aggregate 2 GPU plan only |
| GPU numerical identity | NOT RUN (authorized hold) | GPU=0, Slurm=NO |
| Terra Ultra subagent audit | PENDING | runtime mismatch; GH-approved primary-only continuation |

## Red-team observations

1. 초기 implementation checkpoint의 mutable trial API는 dense checkpoint trial로 되돌아갈
   수 있었다. 최종 commit에서는 read-only trial과 accepted commit을 protocol 수준에서
   분리했고 mutable `TorchFactorTrial`을 reference-only로 명시했다.
2. Static의 frozen field reuse는 field 수만 줄이며 retry trial 수까지 줄이지 않는다.
   Resource forecast의 trial 상한을 이 사실에 맞춰 보수적으로 수정했다.
3. Omega append가 terminal result validation보다 앞서면 failure 때 ledger만 남을 수 있었다.
   최종 ordering은 entry restore 가능한 모든 연산을 끝낸 뒤 append를 마지막으로 한다.
4. `gpu_seconds_per_edit`이 evaluation까지 합치지 않도록 controller/evaluation GPU seconds를
   분리했다.
5. Generic cached graph helper는 보존했지만 current MEMIT execution에서는 supported=false
   gate를 통과할 수 없다.

## Unresolved verification

- CPU tests는 model-scale bf16/fp16 numerical identity와 GPU peak memory를 증명하지 않는다.
- Actual MEMIT proposal/context generation은 dedicated P0 envelope에서 fixed-artifact hash,
  tokenizer/context manifest, hook/reference와 functional/committed tolerance를 다시 확인해야
  한다.
- Terra analysis runtime mismatch가 해소되지 않으면 scientific result interpretation은
  HOLD다.

Audit conclusion은 `PREP_PASS; SCIENTIFIC_EXECUTION_NOT_AUTHORIZED`다.

# JLZ v5 구현과 검증 명세

[Method 정본](method-ko.md)의 구현 계약이다. 현행 v4 실행기를 hotpatch하지 않고 새 `project/run_scripts/jlz_writer_coupled/` 모듈을 만든다. 아래 경로는 **제안하는 새 구현 경로**이며 아직 존재하는 runner나 실행 가능한 CLI를 뜻하지 않는다.

## 1 구현 단위와 책임

| 제안 모듈 | 구현 내용 | 기존 재사용과 변경 |
|---|---|---|
| profile.py, inputs.py | model/benchmark adapter, native IDs/가중치/lookup/version, 실제 B | v4 input 규약 재사용; 100/6/L4–L8 하드코딩 금지 |
| entry.py | no-grad anchor/teacher/raw context keys, 첫 수정 linear 직전 cache | v4 상층 subject-prefix cache 폐기 |
| geometry.py | full-context SPD P, exact primal/dual, 고정 P identity | allocation.py 수식을 새 context second moment로 변경 |
| physical.py | candidate별 FP32 W_eff 공유, 수정 linear 한 번, custom D/input VJP | block-output subject injection 제거 |
| oracle.py | native physical NLL/KL + past native KL/hinge, microbatch 합산 | 기존 selected-position full head와 loss packing 재사용 |
| memory.py | unique-fact reservoir, immutable KL teacher, context NLL anchor, versions | 새 모듈; loss/품질 독립 selection |
| solver.py | relative coordinates, exact group prox, global τ, bounded backtracking | Adam·quadratic V gradient 제거 |
| commit.py | accepted W_eff 동일 copy, full-context H, memory publish, RAM rollback | current-key triangular solve 제거 |
| observe.py, collect.py | actual model metrics, loss/gradient/calls/배분/retention | 기존 evaluator 정의와 분모 유지 |

A/B 모두 동일 geometry와 physical oracle를 사용하며 B만 past loss를 계산한다. Memory selection/RNG는 같은 stream으로 결정한다. A도 history/current input state가 B와 달라진 이후에는 자신의 P를 다시 만든다. Arm 간 teacher, factor, W_eff를 잘못 공유하지 않는다.

## 2 Batch 실행 순서

1. 실제 B와 native input manifest를 확정한다. 이전 memory에서 현재 fact를 제외한 reference ID를 고정한다. C0/H/W/memory/RNG entry identity를 기록한다.
2. W_t에서 현재 native 입력을 no-grad 처리한다. Canonical anchors, own-entry KL teacher, 문맥별 key, 첫 write 전 cache를 만든다. Memory 신규 admission은 metadata/RNG로 결정하고 teacher는 pending 상태다.
3. 각 층에서 full-context P를 한 번 solve한다. SPD/residual/shape가 기술 조건을 통과하지 못하면 수정 없이 원인을 보고한다. 임의 jitter·층 제거로 통과시키지 않는다.
4. D=0 physical loss/gradient를 계산한다. Zero forward가 entry와 같아야 하며 모든 D의 미분 경로는 살아 있어야 한다. B1은 empty memory라 A와 같은 목적이다.
5. Candidate budget 안에서 proximal trial을 만든다. Candidate마다 W_eff 한 세트를 materialize한다. 각 microbatch는 전체 D/P의 작용을 보고 모든 D에 gradient를 누적한다. 거절 시 trial gradient/cache를 버리고 직전 accepted state를 보존한다.
6. 마지막 accepted materialized weights를 그대로 commit한다. Last evaluated와 last accepted를 혼동하지 않는다. 별도 P solve/scale/실현률 gate가 없다.
7. Accepted candidate의 native context key/NLL을 재사용하거나 필요한 정확한 commit pass를 수행한다. H full-context second moment와 pending memory를 원자적으로 반영한다. 실패 시 W/H/memory/RNG를 함께 되돌린다.
8. 실제 committed model을 평가한다. 평가 결과는 다음 candidate·memory 입장·η·층 선택에 피드백하지 않는다.

## 3 수치 의미와 최적화 경계

기준 forward는 FP64 D@P^T를 FP32로 cast한 뒤 FP32 entry weight에 더한다. Bias·activation·norm·head·attention implementation은 profile과 동일하다. Candidate와 commit은 동일 weight tensor를 사용한다. 같은 입력의 FP32 numerical parity와 bitwise weight identity는 다른 검증 항목이다.

Direct backward는 입력 gradient G W_eff를 반드시 유지한다. 이를 빠뜨리면 하층 D가 상층 write를 통과하는 효과가 잘못된다. D gradient G^T(XP)는 큰 dense weight gradient를 생략하지만 곱셈/누적 순서가 바뀐다. 작은 dense autograd reference와 각 층·각 요청 column gradient를 비교한다. Reference가 FP32 dense G^T X 후 FP64 P를 곱하는 경로라면 재배치 오차를 기록한다. Forward identity만으로 backward qualification을 대체하지 않는다.

Baseline tokenization/target positions/weights는 exact equality를 요구한다. P solve scaled residual은 FP64 fixture에서 1e-10, 실제 큰 SPD geometry에서는 조건수와 검증된 reference를 함께 기록하고 허용오차를 고정한다. Initial GPU proposal은 loss abs 5e-5, D gradient relative L2 2e-3이며 각 zero-near block에는 absolute error도 보고한다. 이는 통과 실측이 아닌 qualification 시작 기준이다. 결정 경계나 tolerance 실패에서는 dense reference 경로로 재확인하고 해당 경로의 비용을 포함한다.

Majorization ε_num은 동일 후보의 full/microbatch numerical 차이로 결정하고 config에 저장한다. 실행 중 RS/NS를 보고 바꾸지 않는다. Backtracking은 목적의 수락 규칙이며 성능 gate가 아니다. 25후보에서 24수락을 강제하지 않는다.

## 4 필요한 검증만 수행

이미 [CPU 수식 검산](math/validation.json)은 실제 GPU 구현을 검증하지 않는다. 후속 구현자는 아래 작은 검사를 수행한다.

| 검사 | 판정 대상 |
|---|---|
| Native token parity | 현재·과거 행이 원 native 입력/target/readout과 동일 |
| 고정 후보 0 및 비영점 | full-context P reference, physical forward, direct D VJP, input VJP |
| 요청 간 간섭 fixture | 한 요청 loss가 다른 D column으로 전달되며 microbatch 합산 일치 |
| Fit/commit | 동일 W_eff bitwise, logits/loss parity, commit에 solve 호출0 |
| Prox/backtracking | group zero/활성화, clamp, 수락 감소, 거절 예산, 마지막 accepted 반환 |
| Memory | fresh/readmit/evict/version/current-conflict, KL freeze, contextwise hinge, empty B1 |
| Transaction | 오류 시 W/H/memory/RNG 모두 복구; history 중복 append 없음 |
| 범용 shape | B1/2/partial, 가변 context 수·target 길이·층별 dimension; primal/dual |

실제 모델에서는 2–4개 과거에 관측 가능한 개발 요청을 순서대로 사용해 두 batch를 짧게 실행한다. 미래 main stream이나 공식 P/N을 parameter 선택에 쓰지 않는다. B2에서 past loss가 실제로 gradient에 기여하는지 확인한다. 이 검사는 baseline chain을 추가 실행하는 요구가 아니다.

그 다음 목표 logical B에서 3후보 정도로 후보 시간·메모리·FP64 solve·entry cache·reference 비용을 측정한다. 품질 순위나 장기 NS를 이 시간 pilot에서 판단하지 않는다. Model/profile별 qualification 전에는 main 실행 준비 완료로 표시하지 않는다.

## 5 계산량 보고

배치당 비용을 다음으로 분리한다.

`entry native forward + reference entry/cache + A factor/full-context solve + sum(actual candidate forward/backward/materialization) + commit/history/memory + evaluation`

거절 candidate도 실제 F/B에 포함한다. Logical candidate 횟수, Transformer F/B, 계산 token 수, head position 수, FP64 solve 수, D-gradient kernel 시간, materialization, CPU transfer, peak VRAM/RAM을 따로 기록한다. A에는 B reference forward 비용을 강제로 넣지 않는다.

B100의 context dual600은 기존 pooled100보다 준비 비용이 크다. 대신 후보별 geometry 미분/재-solve가 없고 마지막 writer solve도 없다. 기존 v4 subject-prefix cache 손실과 새 reference 비용을 빼지 않은 채 가속이라고 보고하지 않는다. 143배 작은 변수 수나 D-gradient GEMM 감소를 전체 speedup으로 쓰지 않는다.

`T_total = sum_batch(T_entry+T_geometry+sum_actual_trials T_trial+T_commit+T_history+T_memory)+T_eval`로 추정한다. 미측정 항은 UNKNOWN이다. 다른 GPU의 baseline wall time과 직접 배율을 만들지 않는다.

## 6 산출물과 실험 연결

구현 산출물에는 code/config/input/runtime hashes, numerical qualification, no-checkpoint/rollback receipts, candidate ledger, actual commit/history/memory identity, per-case evaluation raw, 실제 배분·retention·계산량 표가 필요하다.

기존 2k·두 arm 목표에 연결할 수 있지만 v4 결과에서 이어서 v5를 쓰지 않는다. 새 W0에서 독립 v5 A/B chain으로 시작해야 한다. Logical B는 profile이며 BS100이면20commit, 다른 B이면 마지막 partial을 포함해 실제2000요청을 보존한다. 평가 정의·분모는 기존 기준을 유지하고 v5는 새 method로 표기한다.

이번 작업은 재설계와 수학 검산이다. 새 runner 제출, GH/SH4 수락, GPU 실행을 완료한 것으로 표기하지 않는다. GH에 인계할 때는 method/contract/TeX와 이 구현 명세를 함께 전달하고 실제 source·CLI가 마련된 뒤 실행 manifest를 결속한다.

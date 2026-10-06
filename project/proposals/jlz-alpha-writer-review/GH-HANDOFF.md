# GH → SH4: AlphaEdit writer 두 모델 × 세 arm, MEMIT 다음 실행

2026-10-07 KST. Instruction / ACK nonce: `USER-GH-SH4-JLZ-PRICE-ALPHA-WRITER-2K-20261007-R1`.
Task ID: `jlz-price-alpha-writer-2k`.

사용자 최신 원문:

> 이것도 GH에게 전달해서 sh4에게 task 진행하도록 시키자. 이전 memit writer에 이어서 pending 걸어놓으면 됨.

직전 범위 요청:

> alphaedit writer로도 실험을 해볼 가치가 있어 보인다. 3개 arm 2개 모델 올리는 실험 검토해봐. 기대효과도 간단하게 생각해보자

유지되는 사용자 자원 제한은 `gpu cap은 2이다.`이다. 발신은 연구 설계 세션 `01a10dc1-4cd1-74b0-a951-3e043882782b`이며 GH를 사칭하지 않는다. 이번 사용자 지시는 검토를 마친 추가 AlphaEdit 여섯 run의 구현·제출 및 GH→SH4 전달 권한이다. 별도 사용자 재승인은 필요하지 않다. `review-ko.md`와 기존 `artifact-manifest.json`의 미전달·미구현 상태는 검토 시점의 기록이며 새 권한을 보류하지 않는다. 과거 검토 bytes는 보존하고 이 문서 및 `authorized-run-contract.json`으로 최신 실행 범위를 고정한다.

## 소유와 전달 완료 조건

- GH 세션 `01a04939-8873-7673-8dca-4c7fc5e31af0`, CWD `/mnt/raid5/janghj/ODE-edit`가 정본 계획·task·실행 envelope 게시 및 SH4 직접 전달을 소유한다.
- SH4 세션 `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, CWD `/data/janghj/ODE-edit`가 구현·source freeze·Slurm 제출·실험·결과 게시를 소유한다. repository는 `hyunjun1127/ODE-edit`다.
- 공식 app-server direct transport로 정확한 target session/CWD, nonce 중복, accepted turn 및 명시 owner ACK를 확인한다. 기존 active turn에는 관련 작업만 exact expectedTurnId로 steer한다. Git task 게시만으로 전달 성공을 주장하지 않는다.
- GH는 isolated worktree에서 봉인 문서의 정확한 bytes를 정본에 채택한다. SH4도 전용 non-main branch/worktree와 새 frozen source를 사용한다. 기존 MEMIT job/source/archive는 수정·취소하지 않는다. 기존 사용자 model/effort 설정을 유지한다.
- 전달 결과는 owner ACK, 실제 구현 단계, task pending 또는 Slurm pending 여부, job ID·dependency를 구분한다. 최소 완료 조건은 SH4의 작업 수락과 MEMIT 후속 대기 순서의 명시적 등록이다. 구현이 끝나면 추가 승인 없이 여섯 run을 scheduler에 등록한다. GH 전달 turn 종료가 SH4 작업 중단 조건이 아니다.

## 순서·자원 계약

선행 task는 `jlz-price-cap-base-repair-2k`다. SH4의 완료 메시지에서 확인한 제출은 Llama `59931→59932→59933`, Qwen `59934→59935→59936`, CPU collector `59937`이며 게시 main은 `1535e857`이다. 이는 제출 영수증의 관측이며 현재 scheduler 상태를 추측하지 않는다. GH/SH4가 실제 receipt, owner, dependency graph 및 source freeze를 대조한다.

**추가 AlphaEdit GPU 작업은 위 MEMIT 여섯 run 이후에 배치한다.** 구현·CPU 준비는 병행할 수 있다. 두 MEMIT lane이 afterany 직렬 chain임이 확인되면 그 말단 `59933`, `59936`의 종료를 공통 시작 frontier로 사용하고, 그렇지 않으면 실제 전체 선행 frontier에 맞춘다. 두 AlphaEdit lane은 각 모델의 CAP075→CAP100→FREE100 순서로 직렬화한다. 각 cell은 1 GPU이며 기존 MEMIT와 새 AlphaEdit를 합쳐 동시 사용 cap 2다. SH4가 발견하는 다른 소유 active queue도 기존 admission 정책에 포함한다.

대기는 과학적 성능 판정이 아닌 실행 순서다. 선행 실험 실패만으로 독립 cold AlphaEdit run이 영구 미실행되지 않도록 기본 종료 의존성은 `afterany`로 두고, 기술적 공통 결함은 원인을 기록해 해당 후속 실행을 보류한다. 실제 scheduler 설정으로 cap과 순서가 강제되는지 확인한다. CPU collector는 자기 여섯 run의 종료를 기다리며 GPU를 요청하지 않는다. 기존 collector 완료나 별도 성능 승인/monitor를 불필요한 실행 gate로 만들지 않는다.

여섯 job을 제출할 준비가 되면 held 상태에서 실제 source/config/input/W&B/noCP/resource/dependency를 확인한 뒤 release하는 기존 절차를 쓴다. 아직 제출 전이면 task pending과 dependency 의도를 먼저 정본에 남기고 job ID는 비워 둔다. scheduler PENDING으로 허위 보고하지 않는다. 중복 nonce/run 제출, 무관 job 변경, 신규 recurring monitor/heartbeat는 금지한다.

## 정확히 여섯 추가 run

| 모델별 arm | beta_base | layer cap |
|---|---:|---|
| AE-CAP075 | 0.75 | 0.75 a |
| AE-CAP100 | 1.00 | 0.75 a |
| AE-FREE100 | 1.00 | 없음 |

위 세 arm을 Llama와 Qwen 각각 독립 cold W0/H0, 같은 CounterFact first2000·순서·seed20261002, BS100×20으로 실행한다. MEMIT 여섯 run에 추가되는 여섯 cell이며 writer를 run identity의 별도 필드로 넣는다. FREE075/FLAT/REVERSE는 제외한다. edited W/H·entry 가격·실현량은 cell 간 재사용하지 않는다. cold W0 평가는 model/runtime/token/row identity가 모두 일치할 때만 모델 내 재사용 가능하다.

모델은 Meta-Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, Qwen/Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`로 고정한다. 둘 다 L4–L8/anchor8이며 readout과 dimension은 model별로 구분한다. Qwen readout27/intermediate18944/hidden3584/vocab152064, Llama readout31/intermediate14336이다. 선행 repair의 실제 수정 source/model adapter를 채택하고 이전 source 0415나 이 검토의 base commit으로 구현을 되돌리지 않는다.

공통 planner는 LR.1/KL.0625/norm.5, EfficiencyAdam absolute R, FP32 model·FP64 geometry, 25 evaluations/24 updates, grace12 own updates, n_exp4, threshold .05를 유지한다. `native_c=.75`, `beta_max_native_scale=.75`, `beta_max=max(beta_base,.75*max_l pi)`와 기존 stage schedule을 유지한다. base1과 layer cap을 혼동하지 않는다. FREE100은 실제 uncapped projection이다. 모든 writer×arm에 같은 endpoint/tiny-block repair와 telemetry schema를 적용한다. 성능을 보고 추가 tuning하지 않는다.

## AlphaEdit 구현 계약

수학적 정의와 근거는 `review-ko.md`를 필수 정독한다. 고정 projector N, mean key K, 누적 H, 얇은 writer factor Q를 사용한다.

```text
A_alpha = lambda_alpha I + N (H + K K^T)
Q_alpha = solve(A_alpha, N K)
DeltaW = R Q_alpha^T
M_alpha = Q_alpha^T K
lambda_alpha = 1
```

기존 `.02`-threshold projector를 모델별 SHA·shape·layer index로 봉인하고 cold W0부터 고정한다. 저장소 기록의 경로·SHA는 authorized contract에 포함하되 SH4가 실제 자산을 다시 확인한다. MEMIT update 사후 N 투영, 15000 C0 추가 hybrid, official compute_z/remaining-layer residual, exact forcing, Qwen native clamp4/LR.5를 도입하지 않는다.

BUILD·own-entry 가격·raw-context same-layer pullback·terminal commit이 모두 같은 AlphaEdit Q 및 실제 FP32 effective response를 사용한다. fresh upper key, full off-owner pullback, own-entry 가격 고정, history final mean-key once를 유지한다. H는 arm별 독립이며 zero/미달 요청도 기존 계약대로 포함한다. dK/dQ와 full-builder reverse는 추가하지 않는다.

가격은 `X[r,j]=(a_r/a_j)*M_alpha[r,j]/(1-M_alpha[j,j])`의 off-diagonal RMS와 기존 floor/min-layer 정규화를 사용한다. MEMIT 가격을 재사용하지 않는다. LOO 잔차는 `(lambda I+NH)q_minus+N K_minus(K_minus^T q_minus)-N k_r`로 검산한다. 기존 MEMIT 검산식을 그대로 쓰지 않는다. denominator guard는 유지하고, inverse-response 보상이나 임의 clipping은 추가하지 않는다.

`A0=lambda I+NH`는 일반적으로 비대칭이므로 layer/batch당 LU factorization 한 번과 `Y=A0^-1 NK`, `Q=Y(I+K^T Y)^-1` 경로를 사용한다. candidate마다 dense factorization/SVD를 반복하지 않는다. 저장된 N을 다른 basis로 조용히 교체하지 않는다. 원 operator residual, materialized payload 및 candidate/commit 동일성을 실제 B1 안에서 확인한다. 별도 toy/pilot/GPU qualification campaign은 하지 않는다.

solve matrix와 보존 진단 covariance를 분리한다. 비대칭 A0를 에너지 행렬로 쓰거나 `Q_C0=total-Q_H`로 추론하지 않는다. 공통 `tr(DeltaW C0 DeltaW^T)`, `tr(DeltaW H DeltaW^T)`, update norm을 보고하고 C0 배율이 있으면 명시한다. 이는 진단이며 새 loss/가격 항이 아니다.

## 게시·검증·결과

- 공통 repair: breakpoint owner/ZERO/CAP endpoint의 정확한 0/cap, tied endpoint 일관성, FP64 KKT/FP32 feasibility. 임의 norm cutoff·moment reset 없이 task-gradient 재진입을 유지한다.
- 게시 항목: 가격 분포/raw kappa/floor 수/최저가 층, exact-zero/endpoint 교정/tiny 진단 구분, target/action norm과 defined ratio·분모, 종료 상태·expansion stage·cap/spend. zero-target의 cross-owner response는 leakage로 남긴다.
- Alpha 추가 항목: `||Nk||/||k||`, M_rr, 실제 realization과 누설. 낮은 actuation이 낮은 가격으로 오인될 가능성을 점검하되 실행 중 정책을 바꾸지 않는다.
- actual B1의 fit/solve/telemetry cost 및 peak RAM/VRAM을 구분한다. 고정 N만 FP32 기준 Llama 약3.83GiB/Qwen6.68GiB이며 전체 추가 VRAM이라고 가정하지 않는다. SH4가 CPU/mmap/streaming 및 reserve를 실제 자원에 맞춰 봉인한다.
- current/birth-cohort PS acquisition, W5/10/15/20 all-seen·fixed-cohort PS/NS, TF-strict/NLL, lost/gained, 비용을 대응 MEMIT arm과 비교한다. P/N 평가는 loss/controller에 사용하지 않는다. 기대효과는 PRICE의 보존을 유지하면서 PS 획득을 개선할 가능성이며 아직 결과가 아니다.
- W&B·NoCP·raw local·compact Git 게시 정책은 선행 MEMIT 계약을 따른다. source/runtime/input/config/projector/reducer SHA와 실제 Slurm dependency를 남긴다. 허용 write 범위는 새 task의 proposal/plan/audit/report/run/local 경로와 Alpha writer 연결에 필요한 backward-compatible 공통 모듈이다. 기존 frozen source와 무관 파일은 보존한다.
- 과학적 성능이 낮다는 이유로 새 승인 gate를 만들지 않는다. 비유한 수치, operator/commit 불일치, 자산 identity 불일치, cap 위반 등 기술적 결함은 명시적으로 중단·보고한다. 임의 자동 재제출로 실행 수를 늘리지 않는다.
- 권장 완료 보고 경로: `experiment-reports/servers/server4/jlz-price-alpha-writer-2k/report-ko.md`. GH는 task·owner ACK·queue receipt를 게시하고 SH4가 추가 승인 없이 구현/제출을 계속하도록 전달한다.

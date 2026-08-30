# F2 fixed-z equal-action barrier usefulness 사전등록

상태: `PREREGISTERED_BEFORE_Q_GATE_CANDIDATE_ACCESS`

이 screen은 ODE가 아니다. Euler, trajectory, refreshed barrier, ODE solver 및 연속시간 주장을 사용하지 않는다. 고정 direct-z와 실제 registered second-moment action shell 안에서 W0 factual-margin barrier가 held-out reference risk가 더 낮은 endpoint를 gate를 보지 않고 선택하는지만 검사한다.

## 불변 입력과 실행 순서

- authoritative contract: `local/state/barrier-usefulness-f2-v1/authoritative-contract.txt`, SHA-256 `042fc456a57d1c8c672636116aebdce6580209efbd74f72d9bd21d502f716595`
- source baseline: `726247a1e45a5c7f2e3fb9ac9f1334b0965b8d97`, tree `9aaf7f5e69ffb9108752a50720c6ef18628de9c3`
- stock EasyEdit: `/data/janghj/EasyEdit-stock-14cea824`, HEAD `14cea8245f06715684592ab55184939b99d70784`
- data: CounterFact SHA-256 `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`
- FULL-FP32, TF32/autocast/quantization 0, explicit pad=eos 및 left-padding semantic-position gate를 Llama/Qwen 각각 통과해야 한다.
- Phase A corrected F1b가 hard PASS인 경우에만 Phase B ctrl-only를 실행한다. Ctrl candidate/selector receipt가 seal되기 전 edited Q_gate access는 0이다. Phase C/ODE 실행은 0이다.

## Phase A corrected F1b

기존 공개 8 case를 네 model/method cell에서 재검증한다. `A*=[K_E_all,K_H,K_T_all]`이며 `K_E_all`은 stock `compute_z`가 실제 소비한 모든 native rewrite context의 subject lookup input key, `K_T_all`은 canonical+모든 native context의 모든 teacher-forced target prefix input key다. context template, token IDs, fact lookup index, key SHA를 보존한다.

각 candidate는 Official endpoint snapshot exact copy/hash에서 시작하고 평가 뒤 exact copy/hash로 복원한다. add/sub rollback은 사용하지 않는다. Candidate 0/8/16/24/final에 Official replay를 끼우며 sentinel은 forward/reverse를 모두 실행한다. 모든 native context의 absolute realized-z residual과 Official 대비 degradation을 직접 측정한다.

실제 action은 `Q_C(D)=tr(D C D^T)`와 `<D,N>_C`를 측정한다. Cross term을 0으로 기록하지 않는다. `Q_C(N)=0.05 Q_C(D)`, `Q_C(D+N)=1.05 Q_C(D)`를 measured tolerance 안에서 요구한다. AlphaEdit은 projector-feasible intersection에서만 방향을 만들고 rank(P), rank(PA*), symmetry/idempotence deterministic operator audit를 기록한다. 전체 dense `P^2`를 물질화하지 않으며, 각 과학 candidate의 `P v=v`는 직접 검사한다.

## Fresh cases와 reference bank

fresh edit case는 prior fixed-z manifest의 calibration/controller/history/screen/final-audit와 기존 8 case를 모두 제외한 CounterFact에서 metadata SHA rank로 고른다. Llama/Qwen 모두 single-token target 4개, 모두 multi-token target 4개, subject/relation/target-pair 중복 0이다. 결과/Official success는 선택에 사용하지 않으며 failure replacement는 0이다.

각 fresh case의 ctrl/gate metadata pool은 결과 독립 SHA rank로 각각 96개를 disjoint하게 고정한다. edit subject/target, 동일 relation을 제외한다. 모델별 W0에서 target_true teacher-forced 모든 token의 strict argmax와 margin이 아래 numerical floor보다 큰 row를 rank 순서로 처음 32개 선택한다. 부족하면 typed HOLD이며 replacement/pool 확장은 0이다. Q_gate W0 teacher cache 생성은 허용하지만 selector seal 전 edited endpoint 평가 0이다.

- fresh-case root: `ef670ab365978a966ea7cd1eba533048f55335eaabe48e05686001112962aa82`
- anchor-pool root: `57b914a8b5ab68a39700e3dfc04d96895f660375b441c8a7430d5ea528031a5a`
- fresh case IDs/order: `5685, 1109, 14568, 13642, 20496, 1767, 6743, 18825`

## Numerical lock

- FP32 absolute/relative algebra tolerance: `3.0517578125e-5` (=256×float32 epsilon)
- rank: `rtol=max(m,n)*float32_eps`, `atol=0`
- realized-z tolerance: model/method별 cold Official all-context replay relative maximum의 8배와 `256 eps` 중 큰 값
- W0 margin numerical floor: model별 cold replay absolute margin drift maximum의 8배와 `256 eps` 중 큰 값
- nuisance epsilon: candidate 결과 전 exact replay, interleaved Official, reverse order, rebatch, batch permutation, cold reload/process restart, commit/rollback repeat에서 측정한 CVaR/margin 단위 upper bound들의 최대
- axis seed: SHA-256(`odeedit-s06-barrier-usefulness-f2-v1|model|method|case_id|axis_id`)
- Phase A axes 4×±, Phase B axes 16×±
- rescale: preregistered 동일 방향 1회만 허용. shell 실패 시 방향 변경, tolerance/ridge 변경, case/anchor replacement 0.

## Barrier, comparator, selector

W0 strict factual token margin을 `m0`, candidate margin을 `m`이라 두고 `s=m/m0`, `B_ctrl=-mean(log s)`를 사용한다. `m<=0`은 `+inf` typed violation이며 clipping 0. KL comparator는 W0 teacher distribution에 대한 token KL의 CVaR_0.875다. 두 점수를 합치지 않는다.

Gate 전 lexical candidate ID tie-break로 다음을 seal한다.

1. barrier-safe: finite 포함 전체 candidate의 최소 `B_ctrl`
2. barrier-adverse: finite candidate 중 최대 `B_ctrl`; violation은 별도 diagnostic
3. KL-only: 최소 `D_ctrl`
4. action representative: `axis-00-minus`
5. Official endpoint

Gate oracle은 gate-open 후 diagnostic only이다. selector/hyperparameter로 사용하지 않는다.

## Phase B validity와 판정

모든 32 candidate는 direct-z 1회/case 공유, z recompute 0, exact reset, A* equality, all-context realized-z noninferiority, actual action shell, rewrite/rephrase strict nondegradation을 통과해야 한다. final audit open 0이다.

Primary gate metric은 Q_gate teacher KL CVaR_0.875다. `A_i=adverse-safe`, `Delta_B/O=Official-safe`, `Delta_B/KL=KL-safe`, barrier-selected percentile, Spearman(B_ctrl,D_gate), margin violation/lower tail을 case 구조로 보고한다. GO-reference/GO-barrier-specific 판정은 authoritative contract §9를 그대로 적용한다. 8-case p-value 확정 주장은 하지 않으며 scientific promotion은 false다.

Gate 공개 후 이 문서, case/anchor pool, seed, selector, nuisance/tolerance, action budget을 변경하지 않는다.

**다층 공동 편집의 BS10 순차 누적 진단 — 2026-09-29**

저장된 AlphaEdit BASE의 B010/B050/B090에서 각각 시작해 동일한 500개 edit를 batch size 10으로 50회 누적한다. 각 출발점에서 Native AlphaEdit, batch마다 보존 여유를 다시 주는 공동 편집, 누적 보존 상한을 고정하는 공동 편집을 비교한다. 모든 방법은 L4–L8을 사용한다.

기존 편집을 재실행하거나 단층 분기를 만들지 않는다. [다층 공동 METHOD](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-preservation-method-ko.md)를 진단용으로 구체화한 설계다. 현재는 명세와 입력표만 생성했으며 모델 편집·GPU 실행·새 weight 저장은 수행하지 않았다.

**핵심 구성.**

| 항목 | 고정값 |
|---|---|
| 모델 | 기존 Llama-3-8B-Instruct, 동일 snapshot revision |
| 출발 checkpoint | BASE B010 / B050 / B090 — 이미 1,000 / 5,000 / 9,000개 편집 |
| 편집층 | 모든 경로에서 L4, L5, L6, L7, L8 |
| Edit batch size | 10개 요청을 한 batch로 처리 |
| 경로당 추가 누적 | 50 batch = 500개 요청 |
| 경로 수 | checkpoint 3개 × 방법 3개 = 9개 |
| 주 실행량 | batch 시도 450회, 요청 노출 4,500회; 서로 다른 continuation 요청은 500개 |
| 상세 관측 | 추가 0 / 10 / 50 / 100 / 250 / 500개 요청 처리 후 |
| Weight 저장 | 추가 250 / 500개 요청 처리 후, 경로당 두 번 |

BS10은 요청 사이의 결합과 반복 다층 변경을 함께 만드는 조건이다. BS1보다 반드시 더 큰 손상을 일으킨다는 가정은 하지 않는다. 손상은 checkpoint의 기존 누적 상태와 50회의 실제 continuation을 통해 관측하며, 손상을 만들기 위해 학습률·clamp를 키우지 않는다. 정해진 범위에서 손상이 작으면 그 결과를 보고하고 자동 연장하지 않는다.

**세 방법의 역할.**

| 방법 | 공동/순차 writer | 보존 제약 | 확인할 점 |
|---|---|---|---|
| NATIVE | 원래 5층 AlphaEdit의 batch writer | 원법의 projector/history/L2 | 추가 누적에서 실제 손상이 나타나는지 |
| JOINT_STEP | 모든 층 U를 실제 모델에서 공동 최적화 | 매 batch 진입 위험에 ε를 다시 더함 | 단계별 허용 악화가 누적되는지 |
| JOINT_CUM | JOINT_STEP과 같은 공동 optimizer | Base는 경로 시작, history는 초기/수용 anchor에 상한 고정 | 누적 상한이 배분과 보존을 바꾸는지 |

STEP/CUM은 같은 원본 teacher, 유효 목표, write family, optimizer, history 처리, 계산 상한을 사용하고 **보존 상한의 시간축 갱신만 다르게** 둔다. STEP에서도 teacher를 매 batch live 모델로 교체하지 않는다. 두 arm 모두 보존 multiplier를 warm-start하므로 multiplier 초기화 차이를 별도 요인으로 섞지 않는다.

NATIVE와 공동 방법 사이에는 target·목적·solver 차이가 있다. 둘의 차이를 배분만의 인과 효과로 주장하지 않는다. STEP/CUM이 누적 상한의 역할을 확인하는 주 대조다. 순서 seed·단층·다른 배분 방법을 추가하지 않는다.

STEP은 ε를 반복 지급하므로 CUM보다 총 허용 악화량이 크다. 이것은 의도한 정책 차이이며 동일한 총 손상 예산의 최적 배분 비교는 아니다. CUM의 낮은 손상만으로 adaptive routing의 우위를 주장하지 않고, 실제 편집 획득량·거절·배분 변화와 함께 판단한다.

**공통 요청과 batch 구성.**

봉인된 source의 ordinal 9000 이후에서 앞 9,000개와 정규화 subject가 겹치지 않는 고유 subject를 확인했다. Paraphrase 2개·neighborhood 10개 이상, target_new와 target_true가 다른 조건까지 통과한 사례는 925개다. 그중 원래 순서의 첫 500개를 사용한다. 모델 성공/실패로 고르지 않는다.

연속한 10개씩 batch를 묶고 모든 9개 경로에 동일 ID·순서·batch membership을 사용한다. 이는 checkpoint별 원래 다음 batch를 재현하는 것이 아니라 공통 후속 입력에 대한 출발 상태 비교다. 체크포인트 형성 과정은 기존 BS100이고 새 continuation만 BS10이라는 범위를 명시한다. Subject 문자열 분리는 의미적 독립성이나 alias 분리의 보증은 아니다.

[continuation-ids.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/continuation-ids.csv)에 500개 ID, [batches.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/batches.csv)에 50개 batch를 고정했다. 기능적 실패가 있어도 다음 원래 batch로 진행하고 어려운 요청을 다른 ID로 교체하지 않는다.

**BS10의 실제 의미와 writer 계약.**

NATIVE는 원래 batch 함수에 10개를 한꺼번에 전달한다. Layer loop와 residual divisor를 원본 그대로 유지한다. 공동 방법은 10개 요청에서 얻은 key-side factor를 이용하며 rank는 층당 최대 10이다.

\[
A_\ell=[P_\ell(K_\ell K_\ell^T+M_\ell)+10I]^{-1}P_\ell K_\ell,
\quad Q_\ell=\operatorname{orth}(A_\ell),
\quad\Delta W_\ell=U_\ell Q_\ell^T.
\]

이 식은 확인한 native source의 raw-P key-side operator를 사용한다. Symmetric projector solve로 임의 교체하지 않는다. Q는 얇은 SVD의 상대 singular cutoff 1e-6으로 만든다. 버린 mode, effective rank, factorization 잔차를 기록한다. Rank truncation 뒤에도 원 native family 전체와 동일하다고 주장하지 않는다.

모든 층 U는 하나의 실제 모델 forward에서 최적화한다. Q는 batch 안에서 고정하고 다음 batch에서 own-state W/M으로 갱신하지만, 뒤층의 실제 입력은 앞층 U를 반영해 매 forward 다시 계산한다. Subject 위치에만 상수 residual을 주입하지 않는다. 별도의 scalar allocation gate나 강제 균등 분담을 넣지 않는다.

메모리 때문에 loss를 나눠 계산하더라도 전체 10개 요청과 보호 loss의 gradient를 모아 공동 step을 한다. 10개 singleton을 차례로 commit하는 것으로 바꾸지 않는다. 각 batch의 최종 모델을 정한 뒤 전체 10개 post-write key Gram을 L4–L8 각각에 정확히 한 번 append한다.

BASE의 다섯 weight와 다섯 native Gram을 모두 복원한다. 방법 간에는 W/M/RNG/context를 출발 상태로 되돌리고, 경로 안에서는 각각 자기 상태를 누적한다. FP32/eager, TF32 matmul=false/cuDNN=true, 원래 tokenizer/native context를 유지한다. NATIVE는 blue=false, L2=10, target loss evaluation 최대 25회, lr=.1, decay=.5, clamp=.75, KL factor=.0625로 이전 BASE 규약을 따른다. Target cache는 끈다.

**보호 입력과 관측 입력을 분리한다.**

| 입력 | 크기 | 접근 범위 |
|---|---:|---|
| Base control | 공통 16개 | 공동 목적·guard |
| Base observer | 공통 64개 | 기록/평가만 |
| 기존 history control | checkpoint당 16개 | 공동 목적·guard |
| 기존 history observer | checkpoint당 48개 | 기록/평가만 |
| 새 history canonical | 해당 경로가 수용한 요청 전체 | 이후 보존 guard |
| 새 history paraphrase | 모든 요청의 고정 두 paraphrase | 기록/평가만 |
| Neighborhood | 상세 시점의 현재 10개 요청당 10개 | 기록/평가만 |

Base 입력은 continuation 및 앞 9,000개와 subject가 겹치지 않는 reserve에서 고른다. History는 해당 checkpoint까지의 최신 유효 subject–relation 중 오래된 quarter와 최근 quarter를 절반씩 선정한다. 고정 subject hash로 control/observer를 모든 checkpoint에 걸쳐 분리하며 모델 결과는 선정에 사용하지 않는다. [panel-ids.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/panel-ids.csv)

기존 history의 functional 제약은 선정한 control 16개에 대한 것이다. 원래 누적된 1천/5천/9천 edit 전체를 functional guard로 보호했다고 표현하지 않는다. Native Gram에는 원래 이력이 남아 있다. Observer로는 이 제한된 보호가 다른 기존 지식에도 통하는지 본다. 이 factual 진단은 범용 능력 benchmark나 R512의 대체가 아니다.

**진단용 수치 조건을 사전 고정한다.**

- 새 편집: 10개 요청 각각의 native 여섯 context 평균 target NLL ≤ 1.0 nat/token, canonical의 mean NLL(new) ≤ mean NLL(true). 모든 요청을 만족해야 공동 batch를 수용한다. 이는 자유 생성 성공을 보장하지 않으므로 pairwise·teacher-forced strict·생성 성공을 구분해 기록한다.
- Base: ε_B=0.05 nat/token. CUM은 `b_B=R_B(출발 checkpoint)+ε_B`를 끝까지 유지하고, STEP은 `b_B,t=R_B(이번 batch 진입)+ε_B`로 갱신한다. R_B teacher는 둘 다 W0다.
- History: ε_H=0.10 nat/token. CUM은 기존 control의 시작 loss 또는 새 요청의 수용 loss에 ε_H를 한 번 더한다. STEP은 같은 유효 목표의 매 batch 진입 loss에 ε_H를 다시 더한다. 기존 정답으로 teacher를 바꾸지 않는다.
- Write energy: `Ω=0.5 Σ_l ||U_l||²_F/||W0_l||²_F`; 전체 trust bound는 `Σ_l ||U_l||²_F/||W0_l||²_F≤1e-4`다. 이 bound를 batch 도중 자동 확장하지 않는다.
- Numerical feasibility tolerance는 1e-5다. Bounds와 실제 residual을 함께 저장한다.

이 값은 이번 진단의 명시적인 시작 조건이며 최적값·검증된 안정성 임계값이 아니다. Observer 결과를 보고 같은 실행 안에서 조정하지 않는다. 거의 모든 batch가 거절되면 보존 성공 대신 현재 optimizer/검색공간/제약 조건에서 편집을 실현하지 못한 결과로 보고한다.

**공동 solver와 history 비용을 제한한다.**

Inequality augmented Lagrangian을 최대 5 round, round당 primal proposal 최대 8회로 제한한다. 총 최대 40회다. Backtracking 후보는 proposal당 최대 6개다. β는 round별 1/2/4/8/16으로 고정한다. Primal은 전체 U를 함께 다루며 실제 augmented loss의 descent를 확인한다. 마지막 iterate를 무조건 선택하지 않는다.

Primal 좌표는 V_l=U_l/||W0_l||F로 둔다. 전체 joint gradient를 L2 norm으로 정규화한 음의 방향을 사용하고, 초기 trial step 1e-3에서 1/2씩 줄인다. 전체 V를 반경 .01의 ball에 project하며, projected displacement에 대한 Armijo 계수는 1e-4다. 여섯 trial이 모두 실패하면 기록 후 해당 primal round를 끝내고 정해진 dual 갱신/다음 round로 간다. 이 유한 solver의 수렴이나 편집 품질을 사전 보장하지 않는다.

Gradient용 history working set은 기존 control 16개와 새 수용 요청 중 남은 constraint slack이 가장 작은 최대 32개로 둔다. 각 round 끝에서는 **기존 control과 수용한 새 canonical 전체**를 평가해 guard를 확인한다. 누락된 위반은 다음 working set에 반영한다. Full guard는 batch당 최대 5회다. 최종 후보는 전체 guard를 통과해야 하므로 active subset만 통과한 것을 전체 수용 history 보존으로 표현하지 않는다. 새 history의 paraphrase는 이 선택에도 사용하지 않는다.

Gradient working set, source checkpoint, teacher/tokenization, numerical rank, 원래 constraint residual, dual, merit, backtrack, 실제 gradient/forward token 수를 기록한다. GPU 시간은 실행 전 추정값으로 보장하지 않는다.

**거절과 실패를 손상 감소로 오인하지 않는다.**

CUM/STEP은 feasible 후보가 없으면 batch 전체를 거절한다. W/M/새 history anchor는 commit하지 않고 다음 원래 batch로 넘어간다. 기존 보호 제약의 dual 추정은 별도로 남겨 다음 시도의 warm start로 사용할 수 있다. 새 편집 multiplier는 다음 batch에서 초기화한다. 요청 교체·단층 fallback·상한 완화·추가 retry는 하지 않는다.

NATIVE는 finite weight update를 원법대로 누적하고 실패한 개별 편집도 기록한다. NaN·복원 실패 같은 기술 오류는 해당 경로를 중지하고 원인과 마지막 finite 상태를 보관한다. Observer 성능 저하만으로 예정된 native 체인을 조기 종료하지 않는다.

주 곡선의 x축은 offered 요청 수 0…500이다. 수용된 요청 수 및 실제 획득·최종 유지한 요청 수를 함께 표기하고, 가능하면 겹치는 획득 수 구간의 손상도 보조 비교한다. 모든 요청 분모의 최종 RS/PS와 수용 조건부 지표를 둘 다 보고한다. 서로 다른 수용 집합을 같은 문제를 푼 결과로 가장하지 않는다. 동일 quality 구간이 없으면 Pareto 개선 근거 부족으로 둔다.

**측정과 해석.**

매 batch 현재 10개 요청의 canonical/두 paraphrase, 원법과 같은 RS/PS/NLL 규약, 실제 수용 여부와 거절 사유를 기록한다. 배분 기록은 층별 실제 delta norm, 정규화 write energy, effective rank, target loss gradient와 보호 gradient의 방향, dual·남은 여유다. Write norm의 비중은 parameter 사용량이며 기능적 기여율이 아니다.

상세 시점은 batch 0/1/5/10/25/50이다. Fixed base/history observer를 평가하고, 지금까지 들어온 continuation 전체의 canonical/두 paraphrase retention을 측정한다. 그 시점의 마지막 batch 직전에도 fixed observer를 평가해 한 batch의 추가 손상과 누적 손상을 구별한다. Step 0 값은 같은 출발 checkpoint에서 공유할 수 있다.

- `W0→출발 checkpoint`: 이미 존재하던 손상.
- `출발 checkpoint→현재`: 이번 500개 continuation에서 추가된 손상.
- `batch 직전→직후`: 이번 공동 write의 직접적인 추가 변화.

Base는 W0 teacher KL, true-answer NLL, 정확도 상실/회복을 분리한다. 기존 history는 checkpoint 진입 시 성공한 사례의 망각과 처음부터 실패한 사례의 회복을 분리한다. 새 history는 실제 획득 시점 성공과 현재 유지를 연결한다. Pairwise mean NLL 차이를 sequence log-odds나 자유 생성 정확도라고 부르지 않는다. 고정 scoring 규약으로 마지막 시점의 greedy completion도 별도 기록하되 optimizer에는 노출하지 않는다.

손상 발생 시점의 기술적 표시선은 `Base observer의 entry 대비 KL 증가 ≥0.05 nat/token` 또는 `entry-success history의 추가 상실 ≥5%p`로 고정한다. 이는 모델의 보편적 붕괴 기준이 아니며 체인 중지나 solver tuning에도 쓰지 않는다. 해당 history 분모가 0이면 지표를 NA로 둔다. 표시선에 도달하지 않으면 이번 조건에서는 뚜렷한 추가 손상을 유도하지 못했다고 기록한다.

선택층과 뒤층의 fixed-prefix key/readout 변화를 상세 시점에 포착한다. 여기서는 여러 층이 변하므로 singleton의 selected-key 불변성을 가정하지 않는다. Original/live와 batch pre/post에서 같은 token prefix·valid 위치를 사용한다. 필요 관측은 누적 E_l, key 이동, E_l δK_l, 실제 보호 출력 변화의 연결이다. Norm만으로 손상의 원인이나 가산적 기여율을 확정하지 않는다. 큰 Hessian·SVD나 모든 층 조합의 추가 편집을 실행하지 않는다.

세 checkpoint는 한 BASE trajectory의 서로 다른 상태다. 독립 seed 세 개처럼 통계 처리하지 않는다. Batch 및 층별 결과도 독립 반복으로 세지 않으며 raw paired curves와 분모를 우선 보고한다.

**실제 weight를 적게 저장한다.**

각 경로에서 **25번째·50번째 batch 직후** 두 번 저장한다. x축은 offered batch이므로 거절이 있어도 예정 시점은 바꾸지 않으며 실제 commit 수를 metadata에 남긴다. 총 9×2=18개 snapshot이다.

이제 다층 편집이므로 각 snapshot은 **L4–L8의 실제 full FP32 down_proj weight 다섯 개**를 포함한다. 이전 단층의 한 개 tensor 저장 규약은 적용하지 않는다. 각 tensor는 [4096,14336], 224 MiB이고, snapshot당 1.09375 GiB다. 전체 신규 tensor는 19.6875 GiB이며 header·metadata는 별도다.

Pinned 원본 모델에 다섯 저장 weight를 적용하면 그 시점의 모델을 복원할 수 있다. 선택된 다섯 weight 이외의 parameter 불변성을 확인한다. Snapshot에는 source checkpoint SHA, 실제 weight SHA, offered/accepted ID와 순서, 원본/과거 anchor, budget·dual·loss trace, source/config/runtime/context identity를 함께 연결한다. 독립 CPU tensor copy를 atomic save하고 재로드 equality와 모델 출력 parity를 확인한다.

기술 오류로 조기 종료되면 마지막 finite snapshot이 다음 예정 snapshot을 대체한다. 경로당 최대 두 개를 유지한다. 여러 시점의 weight가 같으면 payload를 hash로 중복 제거할 수 있지만 시점 receipt는 남긴다. 이 저장은 사후 모델 분석용이며 중간 M/RNG를 모두 담은 editor의 bitwise 재개를 보장하지 않는다.

[weight-snapshots.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/weight-snapshots.csv)에 18개 저장 지점을 고정했다. 현재 실제 생성된 weight 파일은 없다.

**규모와 실행 준비 상태.**

주 실행은 native batch 150회와 joint batch solve 300회다. Native target fit은 요청별 총 1,500회, joint primal proposal 상한은 12,000회다. Joint backtracking trial forward 상한 72,000회와 full-history guard 최대 1,500회는 별도 비용이며 같은 단위의 fit 횟수로 합치지 않는다. 모든 상세 평가·키 포착·I/O도 별도로 기록한다.

첫 batch의 원본성·복원·materialization 검증에 경로당 최대 한 번의 기술 replay를 허용한다. 이는 native 3회/joint 6회의 batch 호출 상한이며 scientific trajectory를 늘리지 않는다. 새 seed, grid, 다른 모델 또는 자동 연장은 없다.

데이터와 source lock hash를 대조했고 원본 모델 네 shard의 로컬 존재를 확인했다. BASE checkpoint 경로·크기·기록 hash는 [checkpoint-bindings.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/checkpoint-bindings.csv)에 연결했다. 보관 기록은 server2이고 현재 host에서는 그 archive 경로가 보이지 않는다. 이번에 원격 payload를 검증했다고 주장하지 않는다.

남은 실행 준비는 저장 W/M 및 원실행 P/context/runtime 실제 결속, BS10 joint solver 구현, reference/observer 분리, native와 실제 공유 weight forward의 검증, 저장·재로드다. 설계/입력 생성기는 GPU runner가 아니다. [contract.json](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/contract.json), [batch-cells.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/batch-cells.csv), [design-checks.json](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/design-checks.json)에 실제 생성·점검 범위를 기록했다.

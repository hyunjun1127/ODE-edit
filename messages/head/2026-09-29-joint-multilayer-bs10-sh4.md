# GH → SH4: 다층 공동 편집 BS10 진단 구현·제출

Instruction / ACK nonce: `ODEEDIT-GH-SH4-JOINT-MULTILAYER-BS10-20260929-R1`.
사용자: “/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/design-ko.md 이거 SERVER4에게 TASK 진행시키자”.
수신 server4 / session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, CWD `/data/janghj/ODE-edit`, repo hyunjun1127/ODE-edit.
발신 GH `01a04939-8873-7673-8dca-4c7fc5e31af0`.

## 1. 새 실행 권한 / 이전 단층 task와 분리

이번 다층 설계에 대해 자산 결속·구현·최소검산·실제 실험 제출을 승인한다. 단순 계획이나 설계 재작성 요청이 아니다. 단계별 GH 재승인을 기다리지 않는다.
이전 `TEMPORAL-ROUTING-DIAGNOSTIC` 단층 task는 사용자 철회 상태 그대로 유지한다. 그 15branch/1500 BS1 fits/selected-one-weight 저장/선택층 key 불변성은 이번 실행에 적용하지 않는다. 기존 partial 구현·CPU 검산은 참고/적합 구성요소만 명시적으로 재사용 가능하며 새 joint 검증으로 가장하지 않는다.

필독 정본:
- `plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/`의 design/contract/builder/입력·실행·snapshot CSV/asset/design-check 총11개.
- `plans/global/2026-09-29-joint-multilayer-preservation-method-ko.md`.
- 원 native 본체 `audits/global/2026-09-22-alphaedit-original-failure-audit/official/AlphaEdit/AlphaEdit_main.py` 및 실제 runtime dependency/context/P/hparams.
- PROTOCOL/현재 registry·session/resource 정책. 같은 bytes의 이전 FULL_READ는 SHA 결속 재사용 가능.

Design SHA `a16eede37bc30945c3069989b53d22798412dc5c1cc056bb123e7da9c54c4935`, contract `d4e2f441235c86f3f9ce7bde8bec742cc2fdb6a25ea30c90c427eef52b4393cd`, METHOD `5909c1ad674e392d03f6e6fa0d81885259c49a7e35c874588f5a17a8aeeb2bfe`.
Exact13-member manifest는 `audits/global/2026-09-29-joint-multilayer-bs10-sh4-dispatch/input-manifest.json`.
METHOD는 수학 의미의 기준, 구체적인 실험 설정·수치·예산은 이번 design/contract가 우선한다. METHOD의 과거 감사·문헌 링크는 provenance이며 그 선행 실험을 재개하거나 새 선행 gate로 만들지 않는다. 재사용하는 코드의 실제 closure는 별도 검증한다. DESIGN_ONLY/NOT_EXECUTED 표시는 설계 작성 당시 상태다. 정본 CSV CRLF bytes 보존.

## 2. 실행 범위와 입력

기존 BASE_ALPHAEDIT B010/B050/B090에서 다섯 L4–L8 down_proj weight 및 다섯 history Gram 전체를 복원한다. 각 CP에 **NATIVE / JOINT_STEP / JOINT_CUM**, 총9개 독립 trajectory.
모든 경로는 **BS10 × 50batch = 같은500요청**. 450 scientific batch attempts / 4500 arm-request exposures / unique500. 처음부터 W0로 1k/5k/9k를 재편집하지 않는다. W0는 teacher·weight norm의 원본 reference다.

`continuation-ids.csv`·`batches.csv`의 고정 순서/10개 membership을 그대로 따른다. fixed10k ordinal9000 이후 eligible925 중 첫500 metadata-only fresh unique subjects이며 일반 first500이나 CP 원래 다음 batch로 대체하지 않는다. 모델 결과로 요청 재선별0. Dataset SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1, sample lock a8d22c230611b6c14740d1d00658a53856bbde26d060189d5ddf30f2ffdbac92 및 orderedroot를 결속한다.

각 경로 출발 W/M/RNG/context 복원, 이후는 자신의 accepted state만 사용한다. 서로 다른 arm의 later target/Q/update/anchor/dual을 공유하지 않는다. CP-entry의 동일 관측/고정 W0 teacher는 exact identity 확인 후 공유 가능하지만 live gradient/candidate/판정은 독립이다.

## 3. NATIVE 및 실제 joint writer

NATIVE는 실제 원 5층 batch 함수에 요청10개를 한 번에 전달한다. blue=false/L2=10, L4–L8 순서와 residual divisor5/4/3/2/1, native target/solve/history 의미 유지. v_steps25/maxAdam24/lr.1/decay.5/clamp.75/KL.0625, fresh targets/cacheoff. 이번에는 별도 z batching/BF16/clipping 등의 실행 변경을 추가하지 않는다.

Joint는 entry K/P/M으로
`A_l=solve(P_l@(K_l@K_l.T+M_l)+10I, P_l@K_l)`,
thin SVD 상대 cutoff1e-6으로 Q_l, rank≤10, `ΔW_l=U_l Q_l.T`.
Raw-P operator를 대칭 projector 식으로 바꾸지 않는다. Thin factor SVD는 승인됐지만 full weight SVD/Hessian/KFAC는 아니다. discarded modes/rank/residual을 기록한다. Joint에는 native z fitting을 추가하지 않으며 functional edit constraints를 직접 최적화한다.

각 batch에서 모든 U=0으로 시작해 **하나의 실제 모델 forward에서 다섯 층을 함께 변경**한다. Q는 batch 안에서 고정하되 downstream key는 upstream U를 반영해 매 forward 갱신한다. Low-rank hook은 전체 유효 token에서 `U_l(Q_l.T k_l(U_<l))`를 계산해야 한다. Entry key를 실제 forward에 고정하거나 subject 위치 상수 vector로 대체하지 않는다. Singleton의 selected-key 불변성을 가정하지 않는다. Joint all-token 경로와 실제 FP32 five-weight materialization을 연결한다.

Microbatch는 정확한 전체 objective/constraint gradient 누적을 위한 계산 분할만 허용한다. 10개의 singleton optimizer/commit로 바꾸지 않는다. Native 및 joint accepted final endpoint에서 모든10 post-write keys를 사용해 5층 history를 batch당 각각1회 append한다. Wrapper 이중 append0. Joint reject는 W/M/새 anchor commit0이다.

## 4. STEP/CUM 차이와 고정 solver

STEP/CUM의 teacher·유효 목표·Q/write family·optimizer·trust·budget·history 절차는 같고 **보존 상한의 시간축 갱신만** 다르다. 두 arm 모두 Base/history multipliers warm start, current-edit multipliers 새 batch에서0. 교차 arm dual 공유0.

- 각 요청의 native6context 평균 target NLL≤1.0, canonical NLL(new)≤NLL(true), 10개 모두 충족해야 수용.
- W0 teacher 고정 Base KL: ε_B=.05. CUM은 CP-entry risk+.05를 고정, STEP은 이번 batch-entry risk+.05.
- 항목별 유효 history canonical NLL: ε_H=.10. CUM 기존control CP-entry anchor 및 신규 수용시 anchor+.10 고정; STEP은 동일 유효목표 batch-entry loss+.10. 옛 목표로 teacher 변경0.
- Ω=.5 Σ||U_l||²/||W0_l||², trust Σ||U_l||²/||W0_l||²≤1e-4, feasibility tol1e-5. 기준 완화0.
- Inequality AL `Σ(([λ+βc]_+²−λ²)/(2β)) + Ω`; projected dual λ←[λ+βc]_+. β=1/2/4/8/16, max5round×8primal proposal, proposal당6backtrack.
- V_l=U_l/||W0_l||F, 전체 joint gradient 방향 `−g/max(||g||,1e-12)`; 초기 step1e-3, half-backtrack, full concatenated V ball radius.01, 실제 projected displacement의 Armijo1e-4. 6trial 실패는 해당 round 종료 후 정해진 dual/다음round, 무한retry0.
- Gradient history는 기존16control + 새 accepted 중 slack최소 최대32. Round끝 기존control+accepted canonical **전체** guard, batch당 최대5회. 누락 위반은 다음 working set에 반영한다.
- 실제 full guard를 통과한 후보 중 최소 energy를 선택한다. Working set 통과/마지막 iterate를 전체 feasible로 간주하지 않는다. 매 trial마다 extra full-history pass를 숨겨 예산을 초과하지 않는다. 수용은 최종 실제 materialized endpoint의 원 제약과 결속해야 한다.
- Feasible 후보가 없으면10개 batch전체 atomic reject, W/M/newanchor unchanged, 기존보호 dual은 설계대로 유지 가능. 다음 원래 batch로 진행하고 native fallback/단층전환/요청교체/상한확대0.

제약 거절은 과학적 정상 결과이지 기술오류·전체run 실패가 아니다. 거의 전부 거절돼도 보존 성공으로 포장하거나 재튜닝하지 않는다. NATIVE의 finite acquisition failure는 원 update/history와 분모에 남긴다. NaN/restore/identity/IO 오류만 해당 trajectory 기술중단으로 분리한다.

## 5. 평가/해석/비용

Base control16/observer64 공통, history control16/observer48은 CP별(전체 panel272행). 새 accepted canonical은 full-history guard에 들어가고 두 paraphrase 및 observer/N은 기록 전용이다. 원1k/5k/9k 전체를 functional guard했다고 주장하지 않는다. 원Gram은 native preconditioner/history 통계이고 live full functional history 대체물이 아니다.

매batch current10 canonical+2P의 true/new NLL·RS/PS preference·TF 정확도(token micro/prompt macro/strict), acceptance/rejection/원 residual/slack/duals/층별actual delta·energy·effective rank·edit/protection gradient 방향/cost를 저장한다.
상세 batch0/1/5/10/25/50: fixed observer의 마지막 write 전/후, all-offered canonical/2P retention, current10×10N. NS는 true-target 지표, preference와 TF와 마지막greedy completion은 구분한다. 마지막greedy의 token cap/정규화/EOS·truncation 처리 등 구현 설정은 결과를 보기 전 lock하며 optimizer에 노출하지 않는다.

x축 offered0..500 및 accepted/acquired/retained 별도, rejected 요청도 최종 전체분모에 남긴다. 조건부 지표와 공통 획득량의 보조 비교를 함께 보고한다. W0→CP / CP→현재 / batch전→후, entry-success oldhistory losses/recoveries와 신규 at-write retention을 구별한다.
다층 fixed-prefix key/readout, E_l δK_l, actual output 변화는 진단이지 norm 가산적 인과기여율이 아니다. Damage marker Base observer ΔKL≥.05 또는 oldhistory entry-success 추가loss≥5pp는 설명선만, 중단/튜닝 기준0.
STEP는 매번 ε를 지급하므로 CUM과 총 허용악화량이 다른 정책 비교다. CUM 저손상만으로 routing우위0. CP3개는 같은 BASE trajectory의 시점이며 independentseed3개가 아니다. NATIVE와 joint 차이를 배분 단독 인과효과로 해석하지 않는다.

계획 scientific native150batch/1500targetfits, joint300batch/≤12000proposal/≤72000backtrack trial F/≤1500full-history passes. 단위별 실제 calls/tokenF/B/accepted/rejected/IO/teacher 준비/peak/timing을 별도로 남긴다.
기술 firstbatch replay는 **경로당 최대1회**, native3batch/joint6batch까지만 별도 계상한다. 원 단층1502fit 상한은 적용하지 않는다. 필요 없는 반복·성능선별·자동seed/grid/추가chain0.

## 6. 새 저장 예외와 입력 재사용

이번 design은 no-checkpoint 기본정책의 별도 명시 예외다. **9경로 각각 offered batch25/50에5층 actual full FP32 weight**, 총18snapshot/90tensor, payload21,139,292,160B=19.6875GiB(+headers/metadata). 한 층만 저장하거나 factor/hash만으로 대체0.
거절이 있어도 offered 시점 유지. 기술조기중단은 마지막finite가 다음예정snapshot을 대체하고 경로당 최대2개. 동일payload중복제거는 가능하지만 시점receipt보존.
독립CPUcopy/atomic save/reload equality/model reconstruction parity, 다른 parameter 불변. 원모델+5savedweights로 모델 재구성, offered/acceptedIDs·sourceCP·anchor/budget/dual/trace/config/runtime/source/context identity 기록. **exact_editor_resume=NOT_AVAILABLE**: 중간M/RNG 전체 미저장. 원model/parentCP/source/raw 보존·삭제0.

S4 이미 검산한 동일 BASE B010/B050/B090 실물을 우선 재사용한다. 단층 중단task의 입력 binding을 exact hash/schema/stat 범위에서 재사용하되 신규 모델검증으로 확대하지 않는다. 이번 dataset/control/observer는 새500/272표로 다시 결속한다.
새 `transfers/approvals/2026-09-29-joint-multilayer-bs10-sh4-inputs.json`에 정한 정확3CP 중 **S4에 없는 파일만** server2→S4 task input으로 sole pull 가능. 기존 CP가 있으면 재전송0; SOURCE_KEEP/no overwrite/no delete. 이전 철회된 transfer approval을 부활시키지 않는다. 공용model/P/context/통계는 기존원본 재사용, R512 teacher재생성/전송0. 필요한작은source/context는 exactinventory만 전용경로에 확보한다.

## 7. 구현 검산·자원·진행 경계

Dedicated clean branch `codex/server4-joint-multilayer-bs10-20260929-v1`; 새 runtime namespace를 사용한다. CPU 회귀에 raw-P/얇은basis, joint모든층 gradient, context/token 가중치, STEP/CUM bound·dual/anchor lifecycle, full guard≤5, rejection rollback, history5회/acceptedbatch, offered분모, snapshots 및 collector완결성을 포함한다.
실제 모델 확인은 승인 firstbatch/replay 안에서 identity·fullstate restore·joint forward/live downstream interaction·FP32 materialization/observer비변이를 검증한다. CPU PASS를 actual Llama PASS로 바꾸지 않는다. 별도 긴 FD/재현 캠페인/성능 개선 gate를 추가하지 않는다. 실제 근접성 수치는 raw를 먼저 저장하고 확립된범위를 보고하며, 과거 타task tolerance waiver를 자동 상속하거나 원 과학적feasibility guard를 삭제하지 않는다.
구현오류는 선보고 후 좁은수리·새immutableattempt 가능하지만 원source/raw/비용 보존, 수식/상한/총replay예산을 임의 변경하지 않는다.

S4 project/task cap2, 각1GPU/8CPU/host≤60416MiB, gpu/server4/exportNONE/Requeue0. 다른프로젝트 admitted capacity 포함 cap2. 기존job 취소·변경0.
Model+5history/P/FP64solve/autograd activation/controlteacher/fullguard/atomicIO와18snapshot+capture/raw+여유의 RAM/disk/시간 계획을 제출 전에 봉인한다. 이전 추정/waiver를 그대로 가져오지 않는다. 실제 native와 joint첫batch 비용은 구분하고 wall request를 예상runtime으로 쓰지 않는다. 자원 부족을 과학식·guard축소로 우회하지 않는다.

필요준비/최소검산 뒤 **9경로 모두 upfront 등록·held owner/fullargv/source/GPU/mem/dependency 검사→release**, array%2 또는 동등cap-safe lane과 CPU afteranycollector를 연결한다. 기술·과학완결성과 scheduler종료는 별개다.
기존 운영대로 전량 정상제출 뒤 대표 실제 joint B1의 accepted commit **또는 정상reject**→B2 state/anchor/dual/RNG연결까지 bounded 초기확인 후 SH4/worker 능동 모니터링을 중단한다. 정상reject를 초기gate실패로 삼지 않는다. NATIVE 초기의 history5연결도 실제관측수준을 따로표시한다. 모든9gatePASS를 기다리는것이 아니며 대표수준을 명시한다.
모든경로정상release 후 실제 GPU/자원부족으로 main대기이면 pending/INITIAL_NOT_OBSERVED 그대로 인계하고 종료가능. 기술준비PENDING만으로 science미등록을 정상제출로보고하지 않는다.
이미등록프로그램은50batch/예정저장/평가/compactreducer를 자연진행한다. Agent는 초기/자원pending 인계 후 polling/heartbeat/callback/자동recall0. 상세완료리뷰는 사용자recall때. GH 중복raw/GPU감사·추가승인을 선행조건으로 기다리지 않는다.

## 8. 쓰기·게시 및 M0

Local `/data/janghj/ODE-edit/local/joint-multilayer-bs10/20260929-v1/`.
Own-scope만 검산후 nonforce branch/main 게시 승인:
- `project/run_scripts/joint_multilayer_bs10/` (task-local vendor/adapter 포함, 공유native/env/helper수정0)
- `experiment-reports/servers/server4/joint-multilayer-bs10-20260929-v1/`
- `audits/servers/server4/joint-multilayer-bs10-20260929-v1/`
- `plans/updates/server4/joint-multilayer-bs10-20260929-v1/`
- `tasks/status/joint-multilayer-bs10-20260929-v1/server4.json`
- `messages/acks/server4/2026-09-29-joint-multilayer-bs10.md`
- `messages/server-heads/server4/2026-09-29-joint-multilayer-bs10.md`
- `runs/odeedit_joint_multilayer_bs10_s4_20260929/` (exact namespace 명시허용)
- `transfers/verifications/2026-09-29-joint-multilayer-bs10-sh4/`
- ignored local 실제boundary/cap task값; 다른task불변.

Git에는 source/소형표·보고/manifest만, raw/teacher/weight/prompt/fullstdout0. 원dirty/단층중단/provenance/과거실패보존.
최초 nonce ACK 후 FULL_READ/hash/이전중단scope분리/부모입력reuse·새500order/실제남은joint구현·source/9경로·cap2·18snapshot 자원계획을 보고하고 진행한다. 아직없는 job/actualGPU/완료를 주장하지 않는다. NO_BROADCAST_NOT_REQUIRED.

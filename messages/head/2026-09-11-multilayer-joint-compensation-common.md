# GH → SH1(A) / SH2(B): 공동 편집 부담 분산과 손상 보정 독립 실행

- campaign_id: multilayer-joint-compensation-ab-20260911-v1
- A instruction_id: ODEEDIT-S06-MULTILAYER-JOINT-EDIT-A-SH1-V1
- B instruction_id: ODEEDIT-S06-MULTILAYER-DAMAGE-COMPENSATION-B-SH2-V1
- nonce: ODEEDIT-GH-A-SH1-B-SH2-COMMON-FIXTURE-20260911-R1
- GH: 01a04939-8873-7673-8dca-4c7fc5e31af0
- SH1: 01a04939-f93a-7b50-bca0-65438eab2062 / server1(devbox) / /mnt/raid5/janghj/ODE-edit
- SH2: 01a0493a-074c-7f91-9a13-769116326fef / server2 / /mnt/raid5/janghj/ODE-edit
- repository: hyunjun1127/ODE-edit; 배정 전 main cbeb7a031ca52051c0cf88bd5b19cf6c8ea037b6
- 사용자 추가 조건: SH1=A, SH2=B. 동일 sample·동일 checkpoint에서 시작하여 AlphaEdit/MEMIT/BLUE/BLUE-L4_ONLY와 최종 비교한다.

## 1. 원문과 권한

기준 설계: plans/global/2026-09-11-multilayer-joint-edit-and-compensation-design.md
280행/35133B/SHA a8f32937ddd2fc4f87cb3a2cfb9ef95de67d2fd39ce877142fa7b80a1884bf05.
첨부 원문: /mnt/raid5/janghj/.codex/attachments/cb28c4e5-2f55-4a68-ab49-2691d5de5559/pasted-text.txt
114 LF/8940B/SHA 9094300705dca8f07b73dc9a6ba116c8016f660081f3dc9171a57544ece66400 (CRLF 원문 보존).
GH는 첨부 전체·설계 전체·두 참조 리뷰 전체와 실제 native AlphaEdit write/compute_z/entry binding 및 repaired dual source를 읽었다. 새 GPU/수치 검증을 했다는 뜻은 아니다. 각 SH도 전체 원문/설계/참조 source를 읽고 실행 checklist→code 위치를 짧게 남긴다.
설계 math-check.json SHA 2cbf00c4a3ddc79bec3019f1bff2a6bf150fef38fd0169b7e3c229ccd681613f는 과거 소형 CPU 검산이며 새 구현/GPU PASS가 아니다.

사용자는 구현·진단·지정 short chain·계측·분석·최종 한국어 보고까지 승인했다. Slurm=ALLOWED. SH1/SH2는 담당 범위의 최소 correctness, 실행, scoped technical repair, 단계 전환, 결과 분석, own-scope main 통합을 자율 수행한다. GH 중복 code/raw/test 감사를 기다리지 않는다. 반복 재승인 대기0; 실제 과학 조건 변경 또는 실행 불가능 자원만 구체적으로 보고한다. 문서의 작성 당시 '모델 미실행'은 이번 실행 금지가 아니다.

이번 instruction 한정으로 SH factual-only 기본규칙보다 사용자의 분석/해석 요청을 우선한다. 관측·가능한 설명·미분리 요인·한계를 나누고 scientific_promotion=false. A와 B를 결합하지 않는다. B의 성공은 부담 분산 증거가 아니다. 기존 v2/EP/ABC 식을 새 설계 대신 사용하지 않는다.

## 2. 소유 분할과 정확한 실행 목록

|범위|owner|독립 endpoint 수|
|---|---|---:|
|Early/Middle/Late N4와 BLUE 공통 reference|SH1, 두 트랙 공유|6|
|A0/A-OS/A-BF4 ×3 entries|SH1|9|
|Middle A0-bal0, A0-L4, A-Frozen-BF4|SH1|3|
|B-OS/B-BF4 ×3 entries|SH2|6|
|Middle B-Frozen-BF4, N4-anchor L4-only functional OS|SH2|2|
|설계 primary+aux 합|공통6+A12+B8|26|
|사용자 baseline 추가: native AlphaEdit와 native MEMIT ×3 same-entry|SH2, 한 번 계산해 공유|6|

따라서 지정26 endpoint와 추가 baseline6=32 endpoint rows이며 정확한 기존 N4를 재사용하면 새 write 수는 달라진다. 반복 fidelity/packing/probe 평가와 endpoint 관측을 새 과학 arm으로 세지 않는다. 자산과 관측이 exact identity일 때만 재사용하며 원래 숫자만 복사하지 않는다.

Short chain은 별도로 **같은 Middle We=W50에서 B51…B60의 B100 10회**다. Middle 단일-batch 진단 B51은 이 chain의 첫 batch와 겹친다. Endpoint W와 finalization M/ledger까지 같으면 그 첫 state를 재사용할 수 있지만, B51 편집 후 다시 B51부터 시작하거나 11 batches로 세지 않는다. SH1=N4/BLUE/A-BF4 3 chains, SH2=B-BF4 1 chain. 전체4×10 arm-batch state observations; 이 범위 밖 새10k chain/추가 seed/넓은 sweep/추가 BF1은 자동 실행0. Baseline AlphaEdit/MEMIT의 추가10-batch chains는 이번 지정4개에 넣지 않는다.

Middle A0/A-OS/B-OS를 먼저 실제 cost/correctness 계측한 뒤 Early/Late/aux 및 short chain을 완주한다. 성능 gate가 아니며 작은/음성 결과를 제외하지 않는다. 다른 트랙의 성능 결론·완료를 기다리지 않는다. 공유 input/kernel identity를 기다리는 동안 본인 CPU/runner 구현·검증을 계속한다.

## 3. 공통 sample·checkpoint 계약 — SH1 봉인, SH2 독립 수신 확인

고정 data: counterfact-fixed-10k-v1. 양 host 경로
/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/
dataset SHA 3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
ordered root 5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
모델 load 전에 기존 load_prefix/verify. Shuffle/재추출/replacement0.

|fixture|공통 pretrained+BLUE-L4 state|Current 원래 순서(1-based)|
|---|---|---|
|Early|W10/M4_10, 1000 edits 이후|1001…1100 = B11|
|Middle|W50/M4_50, 5000 이후|5001…5100 = B51|
|Late|W90/M4_90, 9000 이후|9001…9100 = B91|
|short chain 시작|동일 Middle W50/M4_50|5001…6000 = B51…B60|

이들은 같은 기존 AlphaEdit BLUE-L4 chain의 checkpoint다. 다른 native/BLUE/MEMIT lifelong chain의 Wk/M을 가져와 same-entry라고 하지 않는다. Full We는 정확한 pretrained snapshot + checkpoint의 selected L4 weight로 복원하며 나머지 모든 parameter는 같은 W0 identity다. Llama3-8B-Instruct revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2와 tokenizer/context/evaluator/source를 pin한다.

기존 S1 ABC root:
 /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/
imports/entries/B010/B050/B090/W-method-state.pt,
imports/entries/B011/B051/B091/native-targets.pt 및 entry/context/commit metadata,
A/Early/native-r4/prepared.pt, A/Middle/native-r1/prepared.pt, A/Late/native-r4/prepared.pt.
기존 N4는 **같은 We/Current/context/z/native config/적용 bytes**가 맞으면 재사용한다. Active-current 정책으로 유효 요청이 달라지면 억지 N reuse 금지; 같은 새 manifest에서 N4 한 번 재계산/공유하고 이유/비용을 남긴다. B100 source의 assert100 API를 B1/B7 helper에 그대로 복사하지 않는다.

S2에는 기존 ABC/EP imports 및 S4에서 이관한 checkpoint archive가 있으므로 exact SHA/schema로 우선 재사용한다. 최신 source→retained 경로는
transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv
및 /mnt/raid5/janghj/ODE-edit/local/checkpoint-migration-server4/20260911-v1/ 를 사용한다.
Server4 checkpoint source는 삭제됐으므로 그 경로를 읽거나 복구·재전송 요청하지 않는다. 보존본/source references를 혼동하지 않는다.

SH1이 common-fixtures-v1을 create-once 봉인한다. fixture별 다음을 포함:
- case IDs/order/raw/effective current/latest-wins 및 nested B1=첫 유효1, B7=첫 유효7, B100 전체 mapping
- W0/We/WN/M4, contexts/RNG/reference source/model/tokenizer/precision/backend
- original native L4 targets와 native receipt, Praw/P* physical mapping, C0/native config
- 공통 K4/K8/M8 준비와 history provenance
- Base128/Past128/BaseAudit128/PastAudit128 fact/version+문자열 inventory와 실제분모
- 평가 Current/Fixed/Past canonical RS/PS/NS·generation/decoding inventory
- 새 설계의 teacher packing/tokenization/normalization/reference identities 및 native rB/rP/sigma

Bank 선정은 이번 설계의 fact/version hash 정책만 사용한다. 필요 namespace는 ODEEDIT-MULTILAYER-JOINT-COMPENSATION-20260911-V1로 봉인하고 deterministic hash ascending/tie by canonical identity를 사용한다. 기존 v2의 native structural-influence selection/bank를 이름만 바꿔 재사용하지 않는다. 부적격·중복·overwrite는 결과 독립적으로 처리하고 raw/effective inventory를 남긴다. 풀 부족/문자열 overlap은 기록; 임의 외부bank나 성능 기반 replacement0.

SH2는 공통 input SHA/size/schema/order 확인과 같은 entry forward/reference의 최소 parity 후 해당 bytes를 사용한다. 서로 독립적으로 contexts·bank·M8·native WN을 다시 만들지 않는다. Base teacher는 We, W0는 audit-only. **v2의 기존 tau=.01/OS-calibrated risk를 재사용하지 않는다**; 이번 tau=.1과 새 동일 bank로 WN risk를 계산해 sigma=max(r,1e-3)를 공유한다. N4 weight 재사용과 새 bank evaluation 비용을 구분한다.

L8 history가 없으면 해당 We에서 그때까지의 승인된 과거 요청 native-context keys로 Gram을 재구성한다. M8=0/다른 chain M8 금지. M4는 원래 누적값 유지. 이후 각 chain은 자기 endpoint에서 필요한 history를 한 번 append하고 자기 state를 이어간다. Bank/fact 선정은 순서·active ledger로 결정하며 방법별 손상 순위로 달라지지 않게 한다. 단, 이후 W/M/teacher/keys/native targets가 자기 chain state에 따라 달라지는 것은 올바른 sequential 비교다.

## 4. 같은-entry baseline 추가 조건

네 명칭을 혼동하지 않는다:
- BLUE-L4_ONLY / N4: native blue=True, layers=[4], L2=1, layer4 target.
- BLUE: native blue=True, layers=[4,8], L2=1, L4 write 후 current-state L8 target.
- ALPHAEDIT_NATIVE: 같은 pinned BLUE repository의 blue=False, 원본 layers=[4,5,6,7,8]/L2=10 및 원본 native z/hparams 그대로.
- MEMIT_NATIVE: 같은 pinned repository의 blue=False, 원본 five-layer/MEMIT covariance·FP64 solve 예외·edit_layer=-1·mom2_update_weight15000 등 원본 config 그대로.

새 A/B는 AlphaEdit-native 정보에 기반한 설계이며 MEMIT판 A/B를 새로 만드는 것은 아니다. User의 baseline 추가는 위 두 same-entry native reference의 추가 실행으로 구현한다. Full target/z는 baseline 자체의 native 정책으로 자기 branch에서 수행한다. A0와 native z를 강제로 공유하지 않는다.

Native AlphaEdit에 필요한 M5/M6/M7도 해당 동일 We와 과거 inventory에서 native keys로 재구성한다. M8은 공통 준비본, M4는 원본을 쓰며 reconstructed-state comparison이라고 명시한다. 각 native baseline의 hparam/L2/target/write 차이를 공개한다. MEMIT에는 새 Alpha history를 넣지 않는다. Baseline5층의 selected weight/P indexing과 재구성 가능한 final W/M를 모두 저장한다.

기존 14-chain final10k 결과는 별도 historical lifelong 표에 source/config/자기 chain history 차이를 명시해 싣는다. 이 표를 이번 같은-entry 비교나 새4chain short-run 점수로 혼합하지 않는다. 같은 We 비교에도 알고리즘/native hparams가 다르므로 layer-only 인과효과라고 주장하지 않는다.

## 5. 구현 소유와 공동 kernel handoff

기존 helper/native 파일은 read-only, 신규 package:
project/run_scripts/multilayer_joint_compensation/

SH1 owner:
- contracts.py, fixture.py, history.py, banks.py, evaluation.py, observations.py
- track_a/ (A0/A-OS/A-BF/Frozen/ablation/runner/chain)
- common_reference/ (N4/BLUE 생성·재사용·history·shared fixture)
- tests/test_contracts*, test_fixtures*, test_history*, test_a*, test_evaluation*

SH2 owner:
- functional.py, linear_solve.py, elastic_qp.py (공유 functional quadratic/JVP-VJP/PCG/joint dual)
- track_b/ (B-OS/B-BF/Frozen/L4-functional-control/runner/chain)
- native_baselines/ (native AlphaEdit/MEMIT same-entry)
- tests/test_functional*, test_pcg*, test_elastic*, test_b*, test_native_baselines*

Top-level __init__.py/package integration owner는 SH1. 각 SH analysis/plot/launch files는 자기 track 하위에 둔다. Shared code API/작은 CPU fixture를 먼저 합의·commit/push하여 다른 owner가 exact source로 import할 수 있게 한다. mutable model/module-global/optimizer 객체를 양 트랙에 공유하지 않는다. 읽기전용 봉인 tensors/source는 공유 가능하다. SH2 shared functional kernel은 support=(4,8)/(8,)/(4,)를 처리하되 A의 functional cross block을 유지하도록 SH1과 작은 CPU+actual derivative tests를 함께 결속한다. SH1은 A0를 구현하는 동안 SH2는 functional kernel/B를 구현한다. 한 SH가 다른 SH 파일을 임의 변경/되돌리지 않는다. 공동 버전 변경은 source/manifest API 합의 후 진행하며 오래된/미완성 commit의 controller만 골라 섞지 않는다.

## 6. 반드시 유지할 과학·수치 계약

본문 수식이 우선이며 아래는 흔한 오구현 방지 checklist다.
- A0 We/R4=R8=0, same logical Current objective, differentiable fixed-entry writer map을 통해 actual weights/full-sequence forward. 25 Adam updates/lr.1/betas.9,.999/eps1e-8, bal.1(지정bal0제외), ridge1. 독립 compute_z 두 번/first-layer full native 선소진 금지. 초기 writer image의 q/pseudoinverse와 지정floor, sigmaE 규약 유지. Raw-sum→mean 시 K/M/ridge 모두 일관 변환.
- R(module-coordinate RHS), D K(writer response), actual joint delivered local response를 분리. Block z와 down_proj output 값을 무근거로 동일시하지 않는다. 실제 L4→L8 입력변화는 전체 forward와 derivative에 포함한다.
- A correction full P* support4+8, 두 layer를 같은 snapshot에서 공동 결정, negative L4 correction 허용, Current equality는 joint 하나.
- B는 exact WN 시작, L4 byte-identical, support8만; L8 target_new compute_z0. Base teacher We/Current teacher WN. N4에서 L4-functional-control만 예외 support4로 명명.
- Base KL(pWe||pW); Past psi_tau(.1) nats/token 악화, token→context→request 평균. Current profile tauE=.1, γE1, phi=2 fB+fP, native metric 평균고유값 정규화, tauH=.01. Praw baseline과 P* projected SPD writer 차이 공개.
- Output gradient/GGN 전체sequence, Current KL+NLL-profile GGN, A의 L4–L8 cross block 유지. Dk 보호 proxy/두scalar coefficient/full Hessian/direct inverse0. Native block preconditioner와 functional joint operator를 혼동하지 않음.
- OS anchor A=WA/B=WN, h1, tE0, A balance cumulative delta 포함/B 없음. BF predictor A=현재W+hD0/B=현재W, horizon1/N4/h.25, fixed Current reference A=We+(s+h)D0/B=WN. 누적 Current loss를 새 teacher로 흡수0.
- Barrier reference 공통 WN. A raw budget Base .5*s²*rB / Past s*rP; B Base(1-.5*s)*rB / Past rP. rhoB2/rhoP1, slackcost rho/(2h); 두 위험은 하나의 nonnegative2D solve. Full primal/equality/stationarity/complementarity/dual/slack residual을 기록.
- Frozen은 첫 predictor의 a/profile-gradient/risk-gradient/curvature 고정, 실제 risk/RHS/tE/budget/nominal/state는 갱신. A balance는 실제 누적 delta에서 계산.
- PCG RHS당 relres1e-4/max20, actual residual과 RHS/vector/token/calls 기록. 미수렴 finite 근사는 typed approximate endpoint로 보존하고 정확한 quadratic PASS나 capacity 실패로 오인하지 않는다. Nonfinite/negative-curvature/잘못된 derivative·weight·dual domain은 technical boundary; 성능을 보고 tolerance/ridge/gain 바꾸지 않는다.
- 이전 v2 controller.py의 음수 dual 허용 bug 경로를 가져오지 않는다. controller_repair.py의 차원별 residual/정확한 nonnegative domain 원칙을 새수식에 맞춰 구현한다. Matrixnorm을 dual sign tolerance로 쓰거나 결과를 posthoc clip하여 PASS시키지 않는다.
- 실제 독립 branch entry restore, endpoint materialization/evaluation/history once, nonselected weight guard. Restore는 실험 제어이며 adaptive rollback이 아니다. Microbatch별 optimizer/history update0.
- no rank100/256 truncation, no performance hardgate/rollback/line-search/native rescue/endpoint deletion, no target/bank/sample 선택 누수.

## 7. 최소 gate·실행 진행·자원

SH1 server1 cap2 / explicit --mem=182272M, SH2 server2 cap2 / --mem=60416M, 기본1GPU/process, exportNONE. 실제 실행/admitted pending 전체를 집계하며 다른 job 종료/선점/동일GPU 중복탑재0. 초과분은 cap-safe dependency/throttle pending. 미배정 GPU-hour budget은 null로 기록, 기존 8h/48h/v2비용 상속0. 이번 고정26+baseline6+지정4 shortchain의 실행 권한이며 null 자체를 추가 승인 gate로 삼지 않는다.

다만 full-space PCG 비용은 기존 v2보다 클 수 있다. Middle A0/A-OS/B-OS의 actual edit-core/load/teacher/history/eval/peak/hostoffload와 RHS별 PCG를 먼저 측정해 전체 잔여비용을 보고한다. Host memory 60416M 초과는 CPU/GPU staging/offload·physical microbatch 조정으로 대응하되 logical B/보호 문항/cross block/precision을 조용히 바꾸지 않는다. 실제 안전한 실행 불가나 할당/디스크 부족이면 해당 항목을 HOLD하고 CPU/다른 독립 범위 진행한다. 추가 유료 자원/서버 확장은 불허.

Preflight: source/runtime/science/input/teacher/history identity, model/module orientation, basic differentiable weight actual parity, CPU constrained2D algebra, loss weighting/logical accumulation, B0/no-op/zeroa typed behavior. 실제 모델 directional derivative, functional operator symmetry/PSD/cross-layer action은 최소 필수다. User가 이번 gate skip을 요청한 것은 아니다. 중복되는 기존 primitive test는 reuse하되 새 GGN 구현을 과거PASS로 대체하지 않는다.
Middle nestedB1/B7/B100과 동일 logical B100 physical1/2/4의 loss/gradient/finaldelta 비교는 지정 engineering 검사다. 방법/fixture/재사용/추가비용을 구분하고 full endpoint Cartesian grid로 확대하지 않는다. Tolerance는 dtype/PCG 정확도별로 사전기록한다.

SH들은 task-local 필요한 진행확인·단계전환까지 맡는다. 초기 gate 뒤 무조건 중지하는 과거 다른task 정책을 상속해 이번 실험을 멈추지 않는다. 불필요한 빈번 polling/범용 heartbeat/automation 설치/타task monitoring 재개0. 보고는 FULL_READ/common seal→Middle validity+actualcost→26 core/aux+baseline 표→shortchain→최종이다. Science결과 warn는 보존, technical identity/NaN/수식오류 block은 영향범위 수리후새attempt. 원시실패와모든비용유지.

## 8. 분석·비교·산출물

두 SH는 공통 comparison schema를 사용한다. state/arm/fixture/common-lockSHA, source+execution+analysis SHA, Current IDs/raw-effectiveB, W0/We/WN/Wendpoint/M signatures, P/context/banks/audit/teacher/profile/metric/calibration, precision/backend/GPU/PCG/reuse를 row identity로 결속한다.
같은 source/state여도 cross-server numeric parity는 측정한 범위만 주장한다. SHA는 equality 증거이고 GPU forward parity를 대신하지 않는다.

모든 endpoint의 실제 Current NLL/RS/PS/NS와 strict/token, Current request/context mean/median/tail/success→failure, Past 및 독립 PastAudit retention, Base 및 독립 BaseAudit We-KL/W0-audit/NS loss-recovery를 기록한다. Native과 calibration panel 재사용은 request-state 중복분모로 세지 않는다. Missing은 NA/status로 남긴다.
We, We+D4, We+D8, We+D4+D8의 actual interventions에서 signed e4/e8/interaction, 이번batch delta만 제거, negative/zero total progress를 단순share로 만들지 않음. B C8 제거 시 exactWN identity. Native Alpha/MEMIT5층에 이2layer attribution을 억지 적용하지 않는다.

최종 각 report에 A0 효과/extra-layer/AOS/BF/refresh/balance/기여분산/사후보정/current약화/bankoverfit/Audit일반화/shortchain지속성/비용을 분리한다. 동일 We 진단표에 N4/BLUE/ALPHAEDIT_NATIVE/MEMIT_NATIVE 기준선을 포함하고 raw-free 상대 track canonical 결과도 같은 common-lock으로 join한다. 상대가 아직 완료되지 않았으면 missing으로 먼저 보고하되 이후 둘 완료시에 통합 비교 CSV/표를 완성한다. GH는 두 canonical reports의 비교해석을 맡고 중복 raw audit는 하지 않는다.

Owner-specific write:
- SH1 reports: experiment-reports/servers/server1/multilayer-joint-edit-a-2026-09-11-v1/
- SH2 reports: experiment-reports/servers/server2/multilayer-damage-compensation-b-2026-09-11-v1/
- 각각 audits/servers/<server>/2026-09-11-multilayer-joint-compensation-ab/
- messages/server-heads/<server>/2026-09-11-multilayer-joint-compensation-ab.md
- tasks/status/multilayer-joint-compensation-ab-v1/<server>.json
- runs/multilayer-joint-compensation-ab-v1/<server>/
- plans/updates/<server>/2026-09-11-multilayer-joint-compensation-ab.md
- local/multilayer-joint-compensation/20260911-v1/<track-or-common>/<attempt-id>/
- transfers/verifications/2026-09-11-multilayer-joint-compensation/<server>/

선택weights/full-space corrections/history/teachers/rawprompts/outputs는 local-only. 각 run-index, trajectory, request metrics, same-state/attribution, paired summary, solver/compute ledger, reconstruction snapshots, Korean diagnostic-report-ko.md, small manifest/receipt를 저장한다. 필요 snapshot과 chronological delta만 보존, 매nodefullmodel복제0. PNG는 직접 코드 생성/재현, imagegen0.
전용 non-main branch에서 완료본 source/tests+rawfree report를 push 가능. Latest main clean integration에서 본인 scope nonforce mainpush까지 승인; 다른 owner 미완성 code와 shared dirty를 임의 변경·통합하지 않는다. Shared stable kernel/contract commit의 명시적 교환·통합은 본 task 정상 단계다. Conflict는 중단/보고. 완료 이후 새 실험 자동추가0/STOP.

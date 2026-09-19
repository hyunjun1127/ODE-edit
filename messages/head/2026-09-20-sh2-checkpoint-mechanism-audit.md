# GH → SH2: 회수 L4-only checkpoint 기전 분석 구현·실행

Instruction ID: `ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1`
Nonce: `ODEEDIT-GH-SH2-CHECKPOINT-MECHANISM-20260920-R1`

## 1. 권한과 완료 경계

사용자 첨부 전문을 [정본](2026-09-20-sh2-checkpoint-mechanism-user-instruction.md)에 byte-exact 게시했다
(SHA256 `f207a7c4c28ea5b5ce4a2dc29f7b1f2950979747816f9ca105afbc3b2c66dad1`).
첨부·설계·contract·31-row CSV를 함께 FULL_READ하고 원 bytes를 보존한다.
**입력 준비 → 새 runner 구현 → CPU 종단/actual W·M → B1 수치 gate → 검증된 operator·demand·activation 분석 → 한국어 최종 보고/main 게시**를 수행한다.
계획서/검토/CPU toy만으로 종료하지 않는다. 가능한 승인 범위를 최소 완료 조건에서 임의 축소하지 않는다.

이 task에 한해 이전 INITIAL_GATE_ONLY/PENDING_HANDOFF를 상속하지 않는다.
GPU 대기 중 CPU 독립 작업을 계속하고, 초기 gate 후에도 계약에 따라 확장하며 최종 보고까지 진행한다.
실패는 해당 의존 cell/claim만 차단한다. 모든 cell의 성공이 아니라 PASS/FAILED/BLOCKED/SKIPPED terminal 상태를 모아 Z00 보고를 반드시 생성한다.
공유 환경·다른 task·기존 job·다른 서버의 실험은 재개/취소/수정하지 않는다.

사용자가 이번 지시문에서 H1–H4의 판정까지 명시했으므로 **본 instruction에 한정하여**
SH2 보고서에 SUPPORTED/MIXED/NOT_SUPPORTED/UNRESOLVED와 근거를 작성하도록 허용한다.
PROTOCOL의 일반 SH factual-only 규칙에 대한 task-local 사용자 예외를 보고에 기록한다.
H5는 후속 연구 질문이며 새 optimizer/GSS/EN/layer arm을 실행하는 권한이 아니다.

## 2. 실제 session·코드 소유

- SH2/server2, session `01a0493a-074c-7f91-9a13-769116326fef`.
- 원 CWD `/mnt/raid5/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit`.
- hostname/session/CWD/origin, 최신 registry·PROTOCOL·원 dirty 상태를 확인한다.
- 새 clean branch `codex/server2-checkpoint-mechanism-audit-20260920-v1` 및 별도 worktree.
- 모델·행렬·평가·GPU/CPU 분석은 전부 Server2. Server4는 아래 승인된 완료 자료의 read/hash/selective pull source만.
- 원 native writer/evaluator/다른 runner를 수정하지 않는다. 필요한 실제 source closure는 읽기 전용으로 pin한다.
- 격리 dependency overlay/venv 설치 및 필요한 import 경로 연결은 본 task 안에서 허용한다.
  system/shared venv를 덮지 않는다. 원 Torch2.9.1+cu128/Transformers4.44.2 및 실제 tokenizers/dependencies를 결속한다.

허용 write:
- `project/run_scripts/checkpoint_mechanism_audit/**` (새 runner, CPU tests, Slurm/control/plot code)
- `local/checkpoint-mechanism-audit/20260920-v1/**` (새 attempt별 inputs/source/cache/env/logs/results)
- `plans/updates/server2/2026-09-20-checkpoint-mechanism-audit*.md`
- `agents/server2/checkpoint-mechanism-audit-20260920-v1.json`
- `messages/acks/server2/2026-09-20-checkpoint-mechanism-audit.md`
- `messages/server-heads/server2/2026-09-20-checkpoint-mechanism-audit*.md`
- `tasks/status/server2-checkpoint-mechanism-audit-20260920-v1/**`
- `runs/odeedit_checkpoint_mechanism_s2_20260920/**`
- `audits/servers/server2/2026-09-20-checkpoint-mechanism-audit/**`
- `transfers/verifications/2026-09-20-checkpoint-mechanism-server4-to-server2/**`
- `experiment-reports/servers/server2/checkpoint-mechanism-audit-2026-09-20-v1/**`

원 checkpoint·source·과거 receipts는 read-only. 공유 plans/contract/threshold 수정권한은 없다.
일반 helper가 exact envelope 경로를 지원하지 않으면 거부/명시 권한을 별도 기록하며 helper 전역완화0.
검토한 own source/compact 보고의 전용 branch 및 main non-force 통합까지 승인한다. 타 변경 보존, conflict는 보고.

## 3. 필수 정본과 입력 상태

필수 6문서:
1. `plans/global/2026-09-20-server2-checkpoint-mechanism-audit-design-v1.md`
2. `plans/global/2026-09-20-server2-checkpoint-mechanism-audit-contract-v1.json`
3. `plans/global/2026-09-20-server2-checkpoint-mechanism-audit-cells-v1.csv`
4. `audits/global/2026-09-20-server2-checkpoint-mechanism-design/server2-preflight.json`
5. 같은 폴더 `functional-source-audit.json`
6. `audits/global/2026-09-20-server2-geometry-gss-history-review/server2-checkpoint-inventory.json`

같이 게시한 design-consistency/math checks는 구조·synthetic CPU 증거일 뿐 실제 입력/모델 PASS가 아니다.
이전 geometry/GSS review는 배경이며 새 GSS 실행 권한이 아니다.
정본 manifest: `messages/head/2026-09-20-sh2-checkpoint-mechanism-authority.json`.

S2 기존 checkpoint root는 contract 그대로 사용한다. W/M12개 재복사0.
S4 companion 선택 수신은 [승인](../../transfers/approvals/2026-09-20-checkpoint-mechanism-server4-to-server2.md)에 따라 SH2가 sole destination writer/puller로 수행한다.
원본 SOURCE_KEEP, hash/size/state/request identity, 전송량·source/dependency bytes·disk를 기록한다.
archival_eval/order/W-M/W0/source/runtime/P4/각 target를 독립 status로 관리한다.
정확 범위의 설치/선택 수신은 이미 승인됐으므로 단계별 재승인 대기를 만들지 않는다.

## 4. 연구 실행 계약의 핵심 확인점

### 원 실행과 평가

- FP32/eager/autocast=false, TF32 matmul=false **cuDNN=true**. 최신 EN 규약/기존 gate-skip를 상속하지 않는다.
- W0 safetensors BF16→FP32 후 기대 tensor hash 확인. P[5,14336,14336]의 slot0=L4.
- 원 writer/evaluator BOS 차이와 evaluator MB16/수동 left-padding/implicit position_ids를 그대로 보존한다.
- B1 archive 기대값 **RS100/100, PS190/200, NS867/1000**, EN100/194/865로 대체0.
- Migration39283_3/runtime40426 표기 불일치는 기록하고 model/source/sample/context/P/W/M 결속으로 판단한다.
- 새 z optimization 호출0. 저장 targets를 사용한 solve 재구성은 새 native z fitting이 아니다.

### A/B: 모델 없이 먼저 가능한 결과

- current100개를 실제 at-write anchor로, seen12개와 (arm,metric_tag,identity)로 join.
  중복 endpoint current/seen은 일치 확인 후 한 번 집계한다.
- safety margin RS/PS=true−new, NS=new−true; tie 실패, token strict/joint 포함.
- all-case를 유지하며 conflict-free/active-version을 별도 보고. 같은 batch conflicting target은 BATCH_INTERNAL_CONFLICT.
- request-cluster2000/seed20260920와 subject-relation sensitivity; sparse 최초실패는 interval-censored.
- actual W 차이는 FP64 cast 후 subtraction. M은 post-write Gram이며 static C0/weight와 구분.
- J sketch는 모든11구간 같은256 random vectors;64/128은 진행 진단일 뿐 조기중단 기준 아님.
  정규화/MC SE/zero·negative flag/per-batch squared-norm 교차항을 빠뜨리지 않는다.

### C–G: 실제 검증 뒤 정해진 범위 확장

- B1/B2 + probe first32에서 시작. Native K는 group[1,5] 내부평균→group평균(.5,.1×5).
  Bare key/h0와 mean K는 따로 저장하고 residual에 bare를 사용한다.
- prefix early-stop는 physical full-forward 대조 뒤 사용한다.
- B1 M1/key, h0+EK, 원 FP32 dense solve와 actual delta/response, row별 원 evaluator parity, W0 first100 anchor.
- contract의 elementwise·RHS-column/request별 최대오차·actual-delta 분모·zero-reference 규칙을 그대로 구현한다.
  CPU PASS를 실제 모델 PASS로 대체0, 결과를 맞추려 tolerance 확대0.
- native700 + stream-ID-disjoint geometry512, 동일 key를13history에 재사용.
  geometry512와 기존 reference512/G256은 별개이며 기존reference 변경/재생성0.
- H=λI+PM의 일반 LU/solve. raw CG/Cholesky/명시 inverse0. history별 factor/RHS는 재사용하되
  native100 단위 S/B를 합치지 않는다. Raw score clipping0.
- pilot0/1/10/100 통과→나머지9history. Target pair B1←0, B2←1, B6←5, B11←10,
  B21←20, B51←50, B91←90. B1 외 RECONSTRUCTED_NATIVE_WRITE 명시.
- B1/B91 dense-vs-factor와 직접 thin SVD B의 gain×target-loading energy, 잔차·near-degenerate band.
- B1/B2 고정 K,R×pilot4, R column permutation20은 대수적 counterfactual이지 편집 arm이 아니다.
- 두 suffix 구간1→10/50→100 모두 first100의 NS1000에서 lost16+matched-retained16.
  부족하면 축소 실수를 보고하고 모집단 확대0. Target별 all-valid-token TF 경로/gradient,
  s=0,.5,1 실제 margin/strict, E K/D K 교차항과 선형 예측·remainder, parent R/P를 보고한다.

설계 간 실제 충돌은 source-backed 해결 근거를 기록하고 영향받는 부분만 보류한다.
누락/실패로 independent CPU 결과까지 폐기하지 않는다. 과학 조건 변경·최소범위 임의축소는 금지다.

## 5. 자원·구현 검토·실패 처리

Slurm submission **allowed**: 이 분석 task 전용 작업과 의미 불변 technical repair/retry만.
Server2 project GPU cap **2**, 본 task 기본/최대동시 **1 GPU**(48GB급), CPU8.
두 번째 slot을 채우려고 중복 모델/arm을 만들지 않는다. 제출 직전 실제 cap/기존 allocation/pending을 점검한다.
다른 job 취소·hold·throttle 변경0. 지정 job만 held inspection→release, exportNONE/Requeue0/명시 --mem.
설계 RAM64GB는 작업 budget이다. S2 tracked hard request ceiling **60416MiB(59GiB)**가 더 작으므로
host request≤60416M, 이 안에 chunking한다. 과학 수치/토큰 packing을 바꾸는 우회는 금지한다.
이 운영 차이는 resource lock에 명시하며 64G로 제출하지 않는다.

Source/input/environment/resource locks 후 CPU 단계부터 진행. wall/storage/time은 staging 크기와
pilot prefix/LU/RHS/suffix 실측으로 산정하고 추정/실측을 구분한다. 명시 GPUh hardcap은 없음.
모델 load/keys/history별 LU 및 유효한 대용량 hash·finite receipt를 재사용한다.
prefix/suffix F/B, solve, verification/hash, I/O, peakRAM/VRAM/allocated 비용을 분리한다.

Preflight/postrun: identity/order/분모/복원/num-contract/선정편향/자원/원본불변/Git raw-free를 검사한다.
Red warn은 근거와 한계를 기록, block은 관련 제출·claim만 정지하며 독립 작업은 진행한다.
복잡한 독립 구현·감사만 bounded 분담 가능하며 간단한 점검/보고는 직접 처리한다.
별도 red 미사용이면 owner audit와 독립 reducer를 그렇게 명시하고 red PASS를 만들어 쓰지 않는다.
기술 오류/OOM은 원 failure/cost 보존·선보고 뒤 동일 의미 chunking/코드 최소수리·새 attempt로 재시도 가능.
효과 없음/수치 gate 불충족을 method 변경·tolerance 완화·과학적 fallback으로 수리하지 않는다.

## 6. 저장·보고·인계

`save_checkpoints=false`: 새 edited W/M/resume/복원 delta bundle 저장0.
기존12CP는 분석 입력으로 보존한다. 사용자가 명시한 key/h0/identity, LU/cache, raw평가·분해표는
분석 artifact로 local에 저장할 수 있으며 모델 checkpoint로 둔갑시키지 않는다.
진단용 W(s)는 RAM에서 사용·복원하고 새 lifelong checkpoint로 commit하지 않는다.

Contract required_outputs15개와 cell별 상태/실행명령/source/env/시간/메모리 및4종 CSV 기반 PDF/PNG를 생성한다.
막힌 결과는 숫자를 채우지 말고 상태/원인/의존성을 남긴다. 한국어 report-ko.md에 H1–H4 각각 판정·근거·한계,
H5 후속 질문을 포함한다. 상관→인과, 작은 singular value→필연증폭, weight norm→locality손상,
높은 RS→무제한 single-layer capacity로 확대하지 않는다.

중간보고: FULL_READ/M0 및 실제 staging/구현/자원 계획 → 최초 CPU 수치 → B1 parity
→ operator 확장/비용 → H1–H4 최종판정·완료보고.
최종 경로 `experiment-reports/servers/server2/checkpoint-mechanism-audit-2026-09-20-v1/report-ko.md`.
Source/raw-free report/receipt를 검산 후 own-scope main nonforce 통합하고 compact 완료 인계한다.
Raw/CP/model/teacher/전체stdout Git0. **NO_BROADCAST_NOT_REQUIRED**: 이번 exact S4→S2 입력만 수신,
새 대형 결과를 타 서버로 중복 배포하지 않는다. 종료 후 새 실험/자동후속0.

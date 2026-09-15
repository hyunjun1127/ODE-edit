# GH → SH4: EP-TW-1 saved-episode FD/gradient repair 실행

Instruction ID: `ODEEDIT-S06-EP-TW1-SAVED-EPISODE-FD-REPAIR-SH4-V1`
Nonce: `ODEEDIT-GH-SH4-EP-TW1-SAVED-EPISODE-FD-REPAIR-20260915-R1`
Parent: `ODEEDIT-S06-EP-TW1-C4-V3-OURS-FIRST-SH4-V1`.
수신: server4 / session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` / `/data/janghj/ODE-edit` / hyunjun1127/ODE-edit.
기준 main `e665d32f27fcb07f2a9d86670e93f46cbbcda859`, actual failed execution `3fb0bfb27773ec9d016e76fcdc978dc996f010a7` / tree `46dfaff451159cefef1230372e54a93e01c58ef3`.
이 지시는 사용자의 2026-09-15 명시 repair-run 요청이다. 이전 DIAGNOSIS_ONLY의 수정/제출 금지는 **아래 범위에서만** 대체한다. 기존 EP-TW-1 v3 과학 계약·cap2·초기 gate 후 대기 정책은 유지한다.

## 1. 원문 정독 및 소유

사용자가 전달한 점검문과 `audits/global/2026-09-15-ep-tw1-failure-review-ko.md` 전체(SHA4c2241e0709a98d2d16ae401633bb7b4afe0018c8c67dac7855d637a85e88c89), 같은 날짜 failure-review-checks.json(SHAac50462b1301faa13187cb7ac99e73cffdc9c04c410e9a06aa998583909ddef8)을 읽고 보존한다. 기존 v3 설계·method/reference/dispatch·PROTOCOL과 본인의 failure-diagnosis-r1 보고/receipt, 실제 실패 source/lock도 연결한다. 사용자 제공 검토36checks는 scalar/source 검사이지 새 Llama GPU PASS가 아니다.

공식 user 점검의 요지: Native100/solve1은 완료, commit0; E-AD .02677 vs FD .25586/.22181, 양·음 외곽구간 slope 비율약660. 우선 FD 국소성 확인하되 AD 정답 확정0. 저장 Vp/A/target 재사용, 실패 전 진단 저장, 독립 direct-weight gradient 대조, 작은간격 FD 수렴, 그후 D 별도 검증. Quality restoration/ODE를 추가하지 않고 검증 뒤 W0 first1000 평가 유지.

SH4가 source 수정·CPU/실제 GPU 기술검사·복구·조건부 본실험 제출까지 전담한다. GH 중복 raw 감사/매 단계 재승인 대기0. 기술 수리 범위를 벗어나는 과학조건 변경만 typed HOLD. 단순 체크는 직접 하고 복잡한 독립 구현/감사만 bounded worker; 전체 pause 경계를 전파한다.

## 2. 불변 근거와 정확한 재사용

실패 job47884는 FAILED1:0/473 allocated GPU-sec, commit0. 기존 failure/partial/source/teacher/보고를 덮어쓰거나 삭제·재분류하지 않는다. Old runtime·source·473초는 그대로 보존; teacher47592 완료98초와 중복 계상0.

진단 재사용 파일:
`/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/attempt-v1/scientific-v1/B001/native-targets-map.pt`
256307027B / SHA `67b1aaaf765fca753e3399016967f62b2e0cf3f5b2f374874b1eb9d9aae88451`.
Vp raw SHA `124e7a3d6a6ead73785703075dd88022ac1c0e7900dc99c7c1e872bee946810f`, header+bytes SHA `e2a584c279634e7d6de0ea99a9d4ab38677d26c4ee502e40c62ee3b519053cf1`는 별개 convention이다.

Pinned W0의 다른 parameter, 동일100request/tokenizer/contexts/P/history와 저장 Vp/A/Z/anchor/radius/K/H, 완료 S64 teacher를 identity로 결속한다. 기술 진단에서 native target/compute_z/native solve를 **다시 수행하지 않는다**. 저장 A를 재사용하여 factorized map 새구성이나 native RA로 Vp 대체0. Source 동일 input을 확인하고 성능이 좋은 saved state를 고르지 않는다. 기존256MB 실물 검증과 입력 source/샘플/teacher linkage에 한정하고 모델/teacher/과거 raw 전량 불필요 재해시 반복0.

단, post-fit RNG·원 gE/gD tensor·optimizer/local teacher payload·accepted commit이 없어 이 파일은 complete continuation checkpoint가 아니다. 재계산 gradient는 old gradient와 byte-identical이라고 주장하지 않는다. 기술 진단은 진단용 RNG를 새로 봉인하고 probe마다 복원한다. 원 실패를 B1 committed 또는 B2부터 resume로 취급하지 않는다.

## 3. 허용 source/기록 경로

새 clean `codex/server4-ep-tw1-fd-repair-v1` worktree와 create-once `local/ep-tw1-c4/20260915-v1/repair-r1/` 또는 동등 새 attempt를 사용한다. 기술 진단과 scientific-r2는 subnamespace·locks·분모 분리.

- 수정 가능: `project/run_scripts/bg_tw_reference/ep_tw/technical.py`, `model_adapter.py`, `runner.py`의 필요한 기술/기록/연결 부분. 입증된 custom-backward/누적/gradient 연결 오류는 수학·native 계약을 보존하는 최소 수정.
- 새 기술 entrypoint/tests, README, 필요한 `control.py/preparation.py/operations.py/run.sbatch/handoff.py`의 repair-mode/conditional admission/출처/기록 지원은 같은 ep_tw package 안에서만 수정.
- `policy.py/ledger.py` 과학 의미, native vendor/fitter/native_map, 기존 teacher source, shared model/tokenizer/P/hparams/환경/기존 frozen source는 read-only. Semantic change가 필요하면 HOLD.
- 기록: `audits/servers/server4/2026-09-15-ep-tw1-fd-repair/`, `experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/fd-repair-r1/`, `messages/acks/server4/2026-09-15-ep-tw1-fd-repair.md`, `messages/server-heads/server4/2026-09-15-ep-tw1-fd-repair.md`, `runs/odeedit_ep_tw1_fd_repair_s4_v1/`, `tasks/status/odeedit_ep_tw1_fd_repair_s4_v1/server4.json`.
- 검사/소스가 검증된 compact repair/G0 factual + own source의 non-force main 게시 허용. Raw/prompt/gradient/probe tensors/teacher/log Git0; global 설계/타SH/공유프로토콜 수정0.

## 4. 실패 이전에 저장하는 진단 경계

각 stage와 probe를 반환/raise **전에** create-once 저장한다. 기록 실패도 명시적 technical failure이며 조용히 진행0.
- raw E(0)/D(0), 요청별 current NLL/strict/margin, 문서별 D/loss, gE/gD tensor·hash·norm·수집 counters/RNG.
- 매 probe의 objective/direction ID/hash, h, C 정의, ± 실제 weight hash와 rounded delta/action norm, E/D(+/−) 및 요청/문서별 rows.
- nominal h*v*A와 실제 FP32 perturbation 차이, actual nonzero/중복 action, sign별 비대칭과 필요 ULP/roundoff 요약.
- 각 기술 단계 자체 receipt, actual restore/selected·nonselected/M/P/cache/hooks/RNG guards.
- 중앙차분, AD 절대·상대오차, E(±h)−E0∓hAD Taylor remainder, 요청별 loss 변화/지배 rows는 local-only; publication aggregate만.
- 동일 endpoint E0/D0 반복 측정과 forward jitter, 실제 perturbation resolution을 관측한다. 기존 8*epsilon*loss-scale은 보조지표이지 full-model error upper bound로 쓰지 않는다.
- 큰 full-model copy/전체 parameter gradient/Jacobian 영구 저장0. 필요한 gW는 L4 selected weight 한 개 범위, local snapshot/재구성 identity와 비용을 기록한다.

## 5. 별도 경로 검증 + bounded FD 계약

### R0: CPU/실제 gradient 연결

작은 CPU FP64 fixture에서 동일 custom affine node forward/VJP/gradcheck를 확인한다. 전체 Llama FP64 전환이나 FP32 default-gradcheck tolerance 적용으로 해결0.
실제 동일 Vp에서 custom residual C 경로의 gE/gD와 **L4 weight를 직접 leaf로 둔 경로**의 gW를 각각 구해 gC≈gW Aᵀ 및 <gC,v>≈<gW,vA>를 확인한다. Direct gW 경로는 검사할 custom map을 또 통과시키지 않는다. Canonical token/score/tie/reduction·100request/64doc mass는 동일하게 유지한다. E+muD 결합0.
이 검사는 map/누적 연결 검증이지 neural backward 전체의 독립 증명은 아니다. 실제 source/class/kernel과 roundoff를 포함한 비교 tolerance를 GPU 관측 전에 기술 manifest로 고정한다. 충분한 신호에서 차이가 나면 그 원인을 최소 수리하고 같은 입력에서 검증한다.

### R1: E FD, 그후 D FD

기존 coarse point를 보존한 **h/2^k, k=0...9**, 최대10간격의 full diagnostic grid를 사전 lock한다. E self-direction은 같은 정의 gE/||gE||, h=1e-3*max(||Vp||,1)/||vA||; 기존 h=.23469502058911304 및 old AD/scalars와 차이도 기록한다. 기존 gradient tensor가 없어 exact old-direction byte identity는 주장하지 않는다. 방향/초기h가 달라졌으면 원인을 기록하며 old scalar를 새 측정으로 대신0.

E 해결 뒤 D64도 별도 검사한다. 각 objective 자기 gradient 방향 외 **고정 seed/규칙으로 만든 독립 방향1개**를 사전 봉인한다. 같은 kernel/objective를 쓰고 별도 방향·h 정의/seed와 최대 probe/recheck 수, resolution·수렴 판정 규칙을 GPU 전에 lock한다. 네 objective-direction 조합(E self/independent, D self/independent), 각10간격의 ±는 최대80 objective observations이며 반복 E0/D0·제한 재검사/gradient 비용은 별도 명시한다. 독립 방향을 AD가 잘 맞는 방향으로 사후 재선택하지 않는다.

새 판정은 모든 coarse point의 일치를 강제하지 않고, **충분히 resolved인 연속한 작은 간격 최소2개**에서 AD 일치와 FD 안정성을 요구한다. 원 derivative relative .15, absolute1e-7, convergence relative .15는 확대하지 않는다. Convergence 분모도 max(|AD|,1e-7)로 유지하며 FD 분모로 바꿔 통과시키지 않는다. 여러 유효 window가 있을 때 선택 규칙은 사전 deterministic lock, 전체 grid 공개. 고립된 한 점의 우연 일치나 Richardson 외삽은 PASS 근거0. Nearzero AD/action·작은h 해상도 소실은 typed UNRESOLVED이지 자동 PASS0.

실행 전 반복/추가 probe 총상한을 정하고 이 grid 밖 무제한 축소/epsilon sweep/kernel sweep은 하지 않는다. No stable window 또는 resolved 일정배수 불일치면 source/score/reduction/gradient를 분리하고 UNRESOLVED/HOLD. 성능조건이나 .15 tolerance를 완화하지 않는다. E만 통과하고 D가 미검증/불일치면 전체 technical PASS0. Old coarse FAIL은 영구 유지하고 새 converged-window 규약은 별도 version으로 기록한다.

Current 한 observation=7 microbatch groups(16×6+4). 한 방향10±=140 groups. D한±scale은128doc forward이며 E와 같은 호출 비용으로 세지 않는다. 재계산 gE/gD/direct gW는 **기술 검사 추가 backward 비용**, method의 per-batch current1/S641 sweep와 별도다.

## 6. 기술 PASS 후 본실험 — 이미 승인된 조건부 진행

새 direct-route/FD E&D/actual map·복원 검증이 모두 해결되면 동일 EP-TW-1 v3의 **fresh W0, coldM0, first1000 B100×10 단일 scientific replacement**를 진행하도록 승인한다. 별도 단계별 사용자 재승인 불필요. 기술진단에서는 원 nativefit286.54s를 재사용했지만 본실험은 complete resume가 아니므로 W0에서 B1을 정상 계산한다. 이는 baseline rerun이나 진단용 native 재계산이 아니라 새 유효 scientific attempt의 첫 batch이며 1000요청 분모·새 비용에 정확히 기록한다. 과거 failed100target을 성공 분모에 더하지 않는다.

기존1000sample/order/W0/model/source-native/teacher/C4/P/L2/fixed-map/quality screen/후보 메뉴/ζ=.25/history1/accepted-ledger 규약은 불변. 새 native1000 외 extra scientific target0, N4 calibration/teacher재생성/baseline rerun/다른method0. Quality restoration, 추가 correction sweep, ODE, scalar/unprojected, old-feedback, norm/hparam tuning, quality 양의 허용량 추가0. 낮은 성능·RAW fallback 비율은 관측이지 실패 gate가 아니다.

기술과 본실험 source/lock 차이를 결속하고 통과한 기술 receipt는 **source/model/A/schema/controller identity가 적용 가능한 범위에서만** 재사용한다. 새 B1 actual Vp가 old saved Vp와 다르면 same-state라고 쓰지 않으며, 구체 endpoint 의존 검사는 새 Vp에서 필요한 부분만 실행한다. 반대로 검증된 동일 검사 전체를 이유 없이 중복 실행하지 않는다. 실패 전에 early diagnostic save는 새 scientific runner에도 적용한다.

기술→본실험은 같은 sealed program의 명시적 conditional 단계 또는 SH4가 gate receipt 검산 후 단일 job 제출로 구성 가능하다. 실패했어도 science가 시작되는 afterany/bypass, agent callback 자동재개, technical PASS 위조0. 사전 본실험을 등록한다면 직접 검증된 기술 PASS artifact를 fail-closed로 요구하며 총 동시/예약 cap을 맞춘다. 원47884 requeue0, 모든 출력은 새 attempt. Full1000 후후속run0.

## 7. 자원·비용·모니터링 경계

Server4 cap2; existing active/admitted pending+technical+newscience 모두 합산. 기본1GPU/1process/8CPU/명시mem≤60416M/GPU, hour cap=null. 실제 disk/inodes/보존 출력 여유·벽시간·측정 계획을 제출 전 resource.lock으로 고정. 부족하면 출력 삭제나 old자료 덮어쓰기로 처리0. GPU 둘을 쓰기 위해 같은 diagnostic/scientific을 복제하지 않는다. 독립 CPU/GPU 검사 병렬은 각 입력·출력 소유를 분리해 cap내 허용하되 E→D→science gate 의존성은 지킨다.
Source freeze→CPU/red preflight→held owner/source/resource/args/dependency inspection→release. 필요한 좁은 technical repair와 검증은 자율 수행하지만 사전 probe 상한/과학 조건을 넘어가면 typed HOLD.

모니터링은 기존 **초기 gate만** 유지한다. 이 repair의 saved-episode 기술 PASS는 scientific G0가 아니다. 실행이 바로 가능하면 기술 PASS 검산→승인된 단일 science 제출→actual firstB100/B2 state 연결 G0까지 최소 확인하고 WAITING_USER_RESUME. 과학적 G0 뒤 B1–B10 persistent job/eval/저장은 그대로 자연 진행; agent polling/terminal대기/자동분석·최종report/main확장0.
Scheduler PENDING이면 실제gate NOT_RUN으로 인계하고 시작을 장기 polling하지 않는다. 그때 이미 승인·봉인된 conditional 프로그램은 자연 실행할 수 있지만 새 agent callback/추가 submit 자동화로 task를 깨우지 않는다. Terminal technical FAIL/UNRESOLVED/HOLD 확정·인계 후 자동repair0; 사용자 recall을 기다린다. 기존 paused/other jobs/ORBODE 변경0.

보고에는 original473GPUsec, 기존 teacher98sec, 재사용nativefit286.54s(기존473의 중첩component), 신규technical·신규scientific allocatedtime를 비중복 분리한다. E/D/direct-route/probe forward-backward/token/IO/restore/peak를 실측 기록. PNG 필요시 코드만. NO_BROADCAST_NOT_REQUIRED.

GH 인계: FULL_READ+수리source/입력/probe lock → 첫 실제 기술 수치·판정 → scientific 제출/G0 또는 PENDING/HOLD. Old failures/불확정 원인과 새증거를 구분하고 “품질 gate 탈락/효능 실패”로 소급하지 않는다. Resume manifest에 source/locks/jobs/마지막상태/검증수준/준비된조건부실행/추가필요행동/예상결과경로를 적고 `monitoring_active=false, automatic_resume=false, resume_trigger=explicit_user_call`로 종료한다.

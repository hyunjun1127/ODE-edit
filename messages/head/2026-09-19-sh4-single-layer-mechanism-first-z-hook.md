# GH → SH4: Single-layer mechanism-first + native z hook 실행

Instruction ID: `ODEEDIT-S06-SL-MECHANISM-FIRST-ZHOOK-SH4-V1`
User Instruction ID: `GH-SL-MECHANISM-FIRST-W0-B100-S3-S10-20260919-V1`
Nonce: `ODEEDIT-GH-SH4-SL-MECHANISM-FIRST-ZHOOK-20260919-R1`

## 1. 최신 사용자 권한과 종료 경계

사용자 첨부 지시문을 수행하고 이전에 검토한 z 연산 효율화 hook도 추가 구현한다. **구현 → 한정 T0 → cold B100 → 계약 gate에 따른 S3 → S10 → 실제 도달 범위의 상세 사실보고와 own-scope main 게시**까지 승인한다. 단계별 정상 준비마다 재승인을 요구하지 않는다. 본 task에는 과거 initial/PENDING pause, B1-only/sequential=false, T/FD skip, storage waiver, EN 삭제/취소 지시를 상속하지 않는다. Gate 실패면 다음 단계 미제출, 실패 단계의 결과·비용·제약까지 보고하고 STOP한다. 초기 gate나 제출만을 완료로 삼지 않는다.

현재 task와 관계없는 job을 취소/hold/변경하거나 checkpoint를 삭제·이동하지 않는다. 새 구현은 전용 namespace에서 수행하며 기존 native/EN 실행·raw는 보존한다. 과학적 fallback을 technical bug로 고쳐 재실행하지 않는다. 재현 가능한 기술 오류는 먼저 보고하고 같은 방법/threshold/범위의 최소 수리·새 immutable attempt 재제출까지 허용한다. 실패 bytes/비용을 보존하고 유효한 완료 산출물은 재사용한다. 수리로 방법·데이터·허용오차가 달라져야 하면 typed HOLD로 보고하며 임의 변경하지 않는다.

대상 server4 / session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` / CWD `/data/janghj/ODE-edit` / repository `hyunjun1127/ODE-edit`.
GH session `01a04939-8873-7673-8dca-4c7fc5e31af0`.
본 게시 commit을 fetch하고 dedicated clean `codex/server4-single-layer-mechanism-first-v1` branch/worktree를 생성한다. 실제 host/session/CWD/repo/registry를 결속하고 shared dirty와 공용 Git identity는 변경하지 않는다.

## 2. 정독 및 정본

아래 전체와 실제 사용 source closure를 읽고 SHA를 기록한다. 문서의 DESIGN_ONLY/runner_implemented=false는 작성 당시 사실이지 구현 완료 증거가 아니다.

1. `project/proposals/2026-09-19-single-layer-mechanism-first-gh-instruction.md`: 사용자 첨부 전문. 원 attachment SHA와 게시 LF/끝개행 정규화 SHA는 authority-manifest에 별도 기록, 문구 변경 없음.
2. `plans/global/2026-09-19-single-layer-mechanism-first-experiment-design-v1.md`
3. `plans/global/2026-09-19-single-layer-mechanism-first-contract-v1.json`
4. `plans/global/2026-09-19-single-layer-mechanism-first-cells-v1.csv`
5. `audits/global/2026-09-19-single-layer-mechanism-first-dispatch/reference/mechanism-and-method-review.md`
6. 같은 reference 폴더 `en-capacity-scope-audit.md`.
7. 최신 PROTOCOL, 실제 EN/native/runtime/config/import 및 기존 reference capsule provenance.

5/6은 원 `/mnt/raid5/janghj/layer_allocation/` 지정 문서의 exact-byte 사본이다. 원 절대링크가 S4에서 없으면 authority-manifest의 게시 경로를 사용하며 본문을 몰래 수정하지 않는다. 원문의 외부논문 링크만으로 새 문헌 검증을 수행했다는 주장은 하지 않는다.

GH 시작 source `17b5a133cd050ea94195e690aab33342e754965b`의 EN runtime 반복검증 제거/GPU FP64 누적은 **CPU157 PASS만 있고 실제 GPU parity/speed 미검증**이다. 완료·parity 확인된 부분만 재사용하고 변경 경로의 필요한 actual 검증은 이번 T0에서 한정 수행한다. 과거 EN actual execution/현재 최적화 source/새 task execution을 서로 구분한다.

## 3. 추가 사용자 요구: native z 효율화 hook

원본 `/data/janghj/tmp/dnm/hooking.py`는 read-only 참고이다. GH가 현재 전체 읽은 SHA256은 `feb3509940a40e8b8027ee7daae4c7486fc43ec80394383627699be4f34f072b`다. SH4는 실제 파일·필요 caller/config/native compute_z를 읽고 현재 SHA를 결속한다. 다른 bytes면 변경점을 먼저 기록한다. 원 tmp나 공유 native를 수정하지 말고 새 `project/run_scripts/single_layer_mechanism_first/` 아래 hook adapter로 구현한다.

필수 구현:
- 고정 own-entry·요청/context/tokenization에서 delta=0 prefix/layer output을 한 번 포착하고 이후 z step은 캐시된 출력+현재 delta에서 nonlinear suffix만 실행한다. Mask/position/RoPE/tuple-return 및 cache kwargs를 실제 pinned transformers에 맞춘다. Cache는 해당 z call·entry에 한정하며 branch/batch/weight/input 변경 시 무효화한다.
- Native loss가 쓰는 target prediction 위치와 KL 위치에서만 full-vocabulary head를 계산한다. Vocabulary 축소·token 누락은 없다. Native rewrite loss layer와 KL의 **최종 모델 logits 경로**가 다르면 각각 올바른 hidden을 사용한다. 원 hooking.py의 loss_layer hidden으로 KL을 대체하지 않는다.
- 독립 요청 z batching을 구현하되 요청별 NLL/context 평균·KL 방향/정규화·norm penalty·Adam·clamp·종료 규칙을 원 native와 동일하게 유지한다. Active loss 합은 요청별 독립 delta의 gradient 의미를 유지해야 한다. 종료된 row는 gradient masking뿐 아니라 Adam step 뒤 frozen delta를 복원한다. Tail/가변 target length/다중 context/padding을 처리하며 request order를 바꾸지 않는다.
- **FP32 model/hidden/head/log-softmax/loss, eager, TF32-off** 유지. BF16 전환·새 grad clipping·다른 decay/step/clamp/25회 상한으로의 임의변경 없음. Right-padding 및 BOS/lookup/target shift는 원 실제 token IDs와 결속한다. 외부 batch_invariant_ops가 있다는 가정이나 미확인 kernel import 금지.
- 구현 설정은 성능을 보기 전에 봉인한다. 기본 batching 후보는 원 코드의 16이나 실제 memory와 의미 동등성이 미확인이다. T0에서 고정 소수 요청으로 unhooked native / cache+head batch1 / 동일 고정 요청의 batched 경로를 구분하여 z·loss·gradient·종료 iteration·최종 actual write·시간·peak를 비교한다. 수치 기준은 상속 계약에서 검사 전에 명시하고 실패 결과에 맞춰 완화하지 않는다. Batch16 미충족이면 이유를 보고하고 CPU/GPU 기술 수리 또는 **batch1 cache+head 실행 최적화**로 봉인할 수 있다. 이를 batch16 성공이나 완전 bitwise 동일로 표기하지 않는다. 과학 성능을 보고 batching 크기를 선택하지 않는다.
- 위 최소 기술 비교만 별도 비용이며 모든 request를 old/new로 중복 fitting하지 않는다. 본실험은 동일 봉인 hook을 N4와 모든 arm의 own-entry native에 적용한다. B1 native 공유1회, B2+ 각 arm 자기 entry에서 요청당 fitting1회, correction의 추가 z0.
- Native 의미 동등성/재사용 조건을 만족하지 않는 옛 endpoint는 직접 비교를 대체하지 않는다. Fresh native가 필요하면 B1 공통1회만 만든다. 기존 자료와 같다고 가정해 새 hook의 z 결과 대신 끼워 넣지 않는다.

이전 주석의 약27%는 layer-call 계산일 뿐 실측 speedup이 아니다. Prefix/head/batching의 실제 적용 여부와 효과를 각각 보고하며 matched 조건이 아니면 속도 인과효과를 주장하지 않는다.

## 4. 반복비용 제거와 필수 T0/수용 조건의 경계

문서별 dense weight gradient의 GPU→CPU 이동은 없앤 상태를 유지한다. GPU에서 문서 순서·가중치를 보존하여 누적하고 필요한 최종 gradient만 CPU로 보낸다. A_i는 전체 valid input-token activation gradient factor이며 작게 저장/streaming할 수 있지만 512개 dense G_i를 저장·전송하지 않는다. GPU FP64 누적·추가 메모리를 실제 결속한다.

불변 teacher/key/residual의 매문서·매후보 SHA/finite/argmax/normalization, prefix/session 반복 full-byte readback 같은 검증 loop를 되살리지 않는다. 한정 T0·고정 provenance/소비 buffer/ownership binding, 실제 수치 finite/IO/transaction, **방법 자체의 Q_E/Current/Past/full512 candidate acceptance**는 삭제하지 않는다. 이번 첨부가 명시한 T0는 새 변경 경로에 대해 수행하며 CPU toy를 대체 evidence로 쓰지 않는다.

Decision 후보는 W0 token-ID capsule만 사용하고 full-vocab teacher probability를 매후보 읽지 않는다. Full-vocab competitor forward는 유지한다. EN-KL 및 postseal KL observer에 필요한 teacher는 별도 bounded streaming으로 취급한다. T0 native/affine/full-token/gradient/FD/basis/cross-term/coverage는 정본4reference+4Current panel과 설계 범위로 제한한다. 검증을 명분으로 전체 과학 batch를 중복 실행하지 않는다.

## 5. 실행·데이터·방법의 고정 범위

첨부·설계·contract의 모든 식/규칙을 따른다. 핵심 누락 방지:

- Pinned Llama revision/FP32 L4 down-projection 하나, W0/zeroM, fixed10k first1000 B100. Warm5k·추가layer·추가z·paraphrase학습0.
- R512 전체와 실제 W0 generated EOS/max256, 별도 Dev128. 기존640 capsule은 identity와 실제길이에 따라 재사용한다. 130235 train positions는 capsule이 같을 때만 인용한다. Native/current/history/selected마다 factor의 endpoint identity를 분리한다.
- B1 고유4arm=N4/EN-KL-Q/DEC-LINE/DEC-MODES-CUM. STEP은 CUM alias로 중복실행0. 독립 chain state clone만 한다. Legacy EN8trial를 새 EN4trial와 이름만으로 동일시하지 않는다.
- Reference Phi_R 및 entry-anchored active-history Phi_H, Psi=두 block 평균의 합. 全512/fullvocab/validposition; 모든도착 registry, 최신validtarget만 active, C_hist와 별도. GSS/recency/Past표본추출0.
- 하나의 nonlinear center, derivative factor 재사용 J, top3+residual functional≤4 + STEP/CUM covariance방향1, 최종 rank≤5. Dense fullSVD/C_R 불필요생성0. 원 gradient span error/angle/rank 기록.
- 고정 two-phase QCQP/minnorm, 1/.5/.25/.125 최대4 actual후보, 추가relinearization0. Reference 모든 native-safe token ID no-new-flip, worstdeficit/Phi/Psi/Current/전체activePast guard 및 실제 full512 coverage를 그대로 적용.
- Native fallback/Past violation/local infeasibility certificate/finite search unresolved/zero-risk/tie/zero-gradient를 구분한다.
- Official R/P/N/Dev는 selection seal 뒤 observer. B1/S3 확대에만 계약상 development gate로 사용하고 후보선택에 역류시키지 않는다.

## 6. 조건부 확대와 보고까지 계속

B1→S3: primary CUM nonzero valid correction, Current/full512, 실제choice복구 또는 수치오차 초과 Phi상대감소≥5%, N4 대비 PS/strict/joint 점손실0. 계약 실패면 S3 제출0, B1 factual report까지 완료한다.

통과 시 N4/STEP/CUM B1 CP에서 독립 B2..B3, B2 N4 own-native same-entry STEP/CUM shadow pair는 no N4 commit/별도비용이다. Arm 분기 뒤 target/update/gradient 공유0.

S3→S10: primary CUM vs N4 B3 allseen의 Current/history/reference 조건, PS/P-strict/joint 손실0 및 W0-correctN grossloss 감소 또는 actualrecovery. 통과 시 STEP 대조도 함께 B10까지 이어간다. STEP만 좋다고 CUM gate 대체0. 실패 시 S10 제출0. 비용2×기준은 standalone accounting이며 초과는 QUALITY_SIGNAL_COST_UNRESOLVED로 별도 기록, 임의timeout으로 쓰지 않는다.

Pending/initial을 완료로 넘기지 말고 승인된 stage의 완료·gate판정·보고까지 계속한다. 실제 storage/환경/authority blocker이면 증거와 미완료범위를 인계하며 임의자료삭제/waiver는 하지 않는다. 새 설계·arm·threshold·full10k로 범위 확장하지 않는다.

## 7. 자원·실행 권한

Slurm submission **allowed**: 본 T0/B1 및 gate 통과한 S3/S10과 고정 B1/B2 진단만. Server4 project **GPU cap2**, 본 task도 running 합계≤2. 각 job1GPU/8CPU/host memory≤60416MiB/exportNONE/Requeue0를 기본으로 실제 peak/시간/storage를 preflight에서 봉인한다. GPUh 사용자 hardcap은 미지정이며 추정치를 허가된 실측으로 쓰지 않는다. 독립 chain을 cap2로 병행할 수 있으나 B1/shared derivative를 슬롯 채우기 위해 중복하지 않는다. 다른 project allocation도 resource-only admission에 포함하고 pending/dependency/throttle로 cap을 보장한다.

Source freeze→held owner/name/command/fullargv/source/config/CPU/memory/GPU/dependency 검사→release. Job별 immutable attempt/output/lock, actor/session/source/runtime import/data/reference order identity를 기록한다. GPU/host/disk 실제 peak 및 temporary factor 상한을 먼저 계산한다. 옛72GiB waiver/noCP는 상속하지 않으며 복원 가능한 W/M/RNG/context/ledger CP를 단계별 보존한다. 이미 실행중인 다른 task는 건드리지 않는다.

## 8. 소유권·write envelope

SH4는 아래에만 구현·분석·보고·작업 상태를 쓰고 own-scope branch 및 main nonforce 통합까지 승인한다.

- `project/run_scripts/single_layer_mechanism_first/**` (runner/oracle/solver/z-hook/analysis/tests/config/launcher)
- `experiment-reports/servers/server4/single-layer-mechanism-first-20260919-v1/**`
- `audits/servers/server4/single-layer-mechanism-first-20260919-v1/**`
- `plans/updates/server4/single-layer-mechanism-first-20260919-v1/**`
- `messages/acks/server4/2026-09-19-single-layer-mechanism-first.md`
- `messages/server-heads/server4/2026-09-19-single-layer-mechanism-first*.md`
- `tasks/status/single-layer-mechanism-first-sh4-20260919-v1/server4.json`
- `runs/odeedit_sl_mechanism_first_s4_20260919/**` (compact-only 명시 허용)
- ignored `local/single-layer-mechanism-first/20260919-v1/**` 및 dedicated task worktree.

기존 native/EN/shared runtime/global plans/다른 서버 소유 산출물은 read-only이며 보고는 한국어로 작성한다.
위 기존 경로의 수정이 불가피하면 exact file/diff/이유를 먼저 GH로 전달한다. 원 tmp hooking.py는 read-only; 포팅 코드의 원 source/license/caller provenance를 남긴다. 기존 다른 task가 수정한 source를 되돌리지 않는다. Complex 독립 구현/감사만 bounded blue/red 분담, 단순 검사/보고는 SH가 직접 한다. 실제 쓰지 않은 독립 red를 했다고 기록하지 않는다.

Preflight: data/observer separation, hook native semantics, source/import, fixed solver/threshold, cap/memory/storage/paths/no unintended arm을 감사한다. Postrun: independent NLL/selector/gate/cost/CP/namespace/raw-free 검산. 명시 해결가능 기술오류를 수정하되 미해결 correctness/leakage/resource block은 제출/확대를 멈춘다. Warn은 근거를 공개하고 진행 가능하다. Generic helper가 본 명시 경로를 다루지 못하면 exact envelope 허용과 helper NOT_PASS를 함께 기록하며 공용 helper/정책을 넓히지 않는다.

## 9. 중간·최종 산출물

FULL_READ/M0에는 source확보/미구현/기존산출물재사용/zhook설계/자원계획을 보고한다. 이어 실제 T0 및 hook비교/첫B1표/각 gate·제출 mapping/중요 기술오류/최종보고를 전달한다. 변화 없는 heartbeat나 GH 중복 raw/GPU감사를 요구하지 않는다.

첨부의 execution-manifest.json, preflight-and-parity.json, stage-gates.json, batch-metrics.csv, neighborhood-nll-paired.csv, reference-decision-summary.csv, history-entry-native-selected.csv, candidate-solver-ledger, writer-mechanism-summary.csv, same-entry-step-cum.csv, compute.csv, artifact-index.json, terminal.json, report-ko.md를 완성한다.

NLL paired 분석은 fullN / W0correct→ownnativebroken / entrycorrect→이번nativebroken / stable / past atwrite→now를 구분한다. true/new NLL 각각과 new−true margin, 요청cluster uncertainty·ID·denominator·grosslost/recovered·부족margin대비이동을 남긴다. Ownnative는 독립N4가 아니고 같은endpoint observer는 재사용한다.

H1–H5는 actual writer metric/targetloading/realization/reference action/cumulative cross/global component32+32panel과 matchedcontrol/currentquality까지 postselection 기전 자료로 기록한다. B2 pair비용 별도. 연구 actual공유비용1회와 각method standalone공유필수비용전액 두 회계를 남긴다. 준비/기존실패·재사용/native/zhook/factors/solve/candidate/Past/observer/I-O/peak를 분리하며 미분리는 NOT_SEPARATED이다.

SH는 사실·수치·계약의 기계적 gate만 보고한다. 해석/방법승격은 GH 별도 global review다. 한국어 표·링크·render·code plot 재현과 exact provenance를 검토하고 own-scope main 게시 후 TASK_COMPLETE_STOP. Raw/tensor/teacher/prompt/fullstdout Git0. 같은 host 완결자료를 사용하므로 `NO_BROADCAST_NOT_REQUIRED`를 명시하고 새 대형 원격전송은 하지 않는다. 추가 storage/transfer가 필요하면 exact 요청으로 인계한다.

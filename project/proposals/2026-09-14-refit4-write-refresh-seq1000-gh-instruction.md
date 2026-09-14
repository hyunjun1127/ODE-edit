# GH 전달 지시문 — REFIT4 write-refresh 1,000요청 순차 비교

Instruction ID: GH-REFIT4-WRITE-REFRESH-SEQ1000-20260914-V1

수신: Global Head. 담당 실행 서버: Server4 우선.

상태: 사용자 전달용 지시문. 작성 자체로 GH 전송·SH 배정·GPU 제출을 수행하지 않았다.

GH는 아래 확정 설계를 기준으로 **기존 근거 재사용 확인 → 구현 및 기술 검증 → 정책별 B100×10 순차 실행 → 완료 사실 보고 → global claim 판정**을 끝까지 진행하라. 연구 질문은 기존 target 최적화 budget을 중간 weight write 전후로 나누는 것이, 단순 repeated fitting과 기존 REFIT4에 비해 품질·보존·비용상 유용한지를 확인하는 것이다. 첫 회신이나 단일 batch 결과만 남기고 종료하지 말라.

## 1. 기준 문서와 사용자 우선 지시

- [확정 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-seq1000-final-design.md)
- [정책별 60개 batch cell 목록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-seq1000-cells.csv)
- [실험 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-seq1000-contract.json)
- [기존 six-arm 상세 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-14-lowcost-seq10-review-ko.md)
- [보충안 수치·수식 재계산](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-design-checks.json)

최신 사용자 요구는 **“1개 batch가 아닌 1,000개 sample sequential로 돌려봐야 실험 결과가 유의미하게 해석될 것 같다”**이다. 따라서 방법 선택의 기본 단위는 각 정책의1,000요청 순차 경로다. 이전 G1/G1-R의 단일 batch 대조를 과학적 성능 선별 gate로 적용하지 말라. 단일 batch는 코드·state·optimizer 동작을 확인하는 기술 검사와 보조 진단에만 사용하라.

Audit128/MMLU68/FutureN은 이번 주 실험에서 조회하지 않고 `DEFERRED_NOT_EVALUATED`로 남겨라. Audit 선행 통과, E01 20-cell 완결, 광범위한 기전 분석의 완결을 본 실험의 조건으로 추가하지 말라. 미측정을 성공이나 성능 동등성으로 기록하지 말라.

기존 사용자 판정 원칙도 유지한다. **손실0·모든 지표 비악화·CI 하한 양수·online1.5배 이하를 AND gate로 만들지 말라.** 소수 R/P 손실, mixed NLL, 작은 효과가 있어도 해석 가능한 좁은 claim이 남으면 허용할 수 있다. 결과와 손실·비용은 냉정하게 공개하라. 특정 refresh 수나 ODE 명칭이 이겨야 한다는 전제를 두지 말라.

## 2. 실행 범위와 여섯 정책을 고정하라

공통 entry는 기존 Server4 six-arm의 **L4 W50/M50/context/RNG**다. Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, FP32, physical L4만 수정, BLUE-style singleton/L2=1 및 기존 tokenizer/evaluator를 유지하라. 나머지 native hyperparameter는 봉인된 설정을 상속하라.

각 정책은 fixed10k의 **B51–B60, ordinal [5000,6000), B100×10**을 같은 순서로 처리한다. 각자 자기 endpoint를 다음 batch entry로 사용한다. 새 unique 요청은 정책마다 동일한1,000개이며, 최종 full-seen은 old5000+new1000=6000이다. 이를 새 unique6000 또는 full10k 정책 실험이라고 부르지 말라.

| 정책 | Target Adam 최대 updates/request | Write 계수 | Target/optimizer 규칙 |
|---|---|---|---|
| N4 | [24] | [1] | 원 native 첫 fitting |
| REFIT4 | [24,24] | [.75,1] | 각 단계의 현재 모델에서 fresh reset |
| FROZEN2 | [24,0] | [.75,1] | 자기 batch 최초 absolute Z를 두 fitting에 고정 |
| I2 | [12,12] | [.75,1] | target/Adam 상태 유지, 중간 write 후 최적화 계속 |
| FROZEN4 | [24,0,0,0] | [.75,.75,.75,1] | 자기 batch 최초 absolute Z를 네 fitting에 고정 |
| I4 | [6,6,6,6] | [.75,.75,.75,1] | target/Adam 상태 유지, 각 write 후 최적화 계속 |

FROZEN2는 이전 G1의 FZ4, FROZEN4는 보충안의 F4다. 새 ID에서 숫자는 write 횟수다. 표의0은 추가 target optimization이 없다는 뜻이며 residual solve가 없다는 뜻이 아니다.

주 대조는 **REFIT4−FROZEN2, I2−FROZEN2, I4−FROZEN4**다. 각 정책−N4와 I4−I2를 함께 보고하라. B52 이후에는 trajectory 차이가 포함되므로 최종 차이를 동일 state에서 단일 요소만 바꾼 직접 효과로 확대하지 말라.

## 3. N4/REFIT4 재사용과 실행 source를 결속하라

기존 완료 보고의 검토 snapshot은 `7e67befa1c73b67768d16ab98029468c945f2ac0`, 실제 실행 source는 `5e96dcb3745977b1f273e3f5afbee61167248d49`다. 보고서 위치는 `experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/`이다. 이 SHA를 새 구현의 실행 SHA라고 쓰지 말라. 새 source와 기존 reference source를 각각 기록하라.

기존 N4/REFIT4의 전체1,000요청 경로가 공통 entry, source 의미, model/config, context/RNG, precision, host 및 평가 identity에서 비교 가능한지 확인하라. 재사용 범위와 한계를 manifest에 남겨라. 두 reference가 재사용 가능하면 **신규4경로×10=40 batches**를 실행한다. 하나만 가능하면50, 둘 다 불가능하면 같은 환경에서60 batches를 실행한다. 과학적 비교는 항상 여섯 정책의1,000요청 결과로 구성한다.

기존 source·capsule이 그대로인 비교까지 역사적 모든 tensor의 bitwise 재현을 요구하며 막지 말라. 반대로 symbolic 동등성이나 E01의 작은 총점 차이를 근거로 실행 차이를 무시하지 말라. E01 차이를 현재 실험의 noise floor로 차감하지 말라. 재사용이 성립하지 않는 reference만 새로 만들어 진행하라.

기존 S75/S875/RES8/FULL8은 참고로 남기되 다시 제출하지 말라. 기존 capture에 target 최종값만 있고 Adam moments·teacher가 없다면 I2/I4의 continuation state까지 재사용할 수 있다고 가정하지 말라.

## 4. I2/I4는 별도 target-stepper로 구현하라

기존 `NativeSingletonFitter.fit()`을 여러 번 호출하는 것으로 chunked continuation을 대신하지 말라. 원 native fitter와 fresh REFIT4는 reference로 유지하고 별도 stepper를 구현하라.

각 요청의 B100 entry anchor를 a0, optimizer의 leaf parameter를 entry-offset u로 두어 Z=a0+u를 유지하라. Chunk j의 변경된 모델에서 같은 target-input/lookup의 unhooked anchor aj를 읽고 hook delta를 **u+(a0−aj)**로 구성하라. Anchor는 detach하고, optimizer parameter와 m/v를 재기준화 때문에 바꾸지 말라. 과거 solver/weight trajectory 전체로 역전파하지 말라.

다음 규칙을 고정하라.

- Adam m/v와 실제 update counter는 같은 요청의 chunk 사이에 유지한다. Counter는 실제 Adam update 때만 증가시킨다. 요청 사이·다음 B100으로 넘기지 않는다.
- KL teacher, regularizer 기준, clamp 중심/반경은 해당 branch의 B100 entry 기준으로 유지한다.
- Regularizer는 source의 `v_weight_decay * norm(u) / norm(a0)^2`다. Squared numerator나 AdamW로 바꾸지 않는다.
- KL은 source의 `kl_div(entry_log_probs,current_log_probs,log_target=True,reduction=batchmean)` 순서를 유지한다.
- Target tokenization, rewrite/KL contexts, lookup, loss layer, learning rate·betas·epsilon, projection 뒤 moments 처리도 native 의미를 유지한다.
- Chunk0에서는 native delta=u가 되도록 구현하고 I1=[24] reference와 연결을 확인한다.
- Chunk 내 total loss<0.05에서 해당 요청을 조기 종료하되 다음 chunk에서 다시 평가한다. 0-step 요청을 영구 제외하지 않는다. 사용하지 않은 quota를 다른 요청·다음 chunk로 이월하지 않는다.

REFIT4의 fresh reset 규칙을 I2/I4로 소급하거나 반대로 I2/I4의 carry 규칙을 REFIT4에 적용하지 말라. 첫 campaign에서 I4-reset과 모든 teacher/optimizer 조합을 추가하지 말라.

## 5. B100 동기화·writer·history 의미를 보존하라

각 chunk에서 모델 W_j를 고정하고100개 요청의 target chunk를 모두 처리한 뒤 하나의 batch write를 수행하라. 요청 하나마다 즉시 write하는 B1 순차 편집으로 바꾸지 말라.

모든 write는 **현재 absolute Z−현재 canonical Y(W_j)**의 residual을 사용한다. FROZEN2/FROZEN4도 현재 Y를 다시 읽는다. 첫 residual을 재사용하거나, entry 기준 누적 endpoint를 incremental D처럼 계속 더하지 말라. FROZEN target은 그 정책의 자기 batch 첫 target이며 다른 정책의 미래 Z를 공유하지 말라.

FP32 materialization은 기존 six-arm과 맞춰라. 현재 entry에서 native full candidate를 만들고 gamma=1이면 exact endpoint copy, gamma=.75이면 실제 저장 FP32 candidate−entry를 .75배 하여 entry에 더한다. 이상적 solve delta를 직접 축소한 다른 반올림 경로를 조용히 섞지 말라. Arithmetic device도 기록하라.

첫 campaign에서는 native direct solve와 연산 순서를 유지하라. 같은 batch의 K/PK/G 재사용은 identity를 확인하고 비용과 적용 범위를 남겨라. RHS map 선계산·factorization 재사용·analytic readout 대체는 첫 비교에서 추가하지 말라. G를 무조건 SPD라고 가정하지 말고 dense inverse를 도입하지 말라.

Inner history append는0회, B100 terminal에서 native finalizer를 한 번 호출하여 M4를1회 append하라. 각 branch의 W/M/context/RNG를 독립적으로 이어가라. 다른 endpoint의 finalization이 다음 분기에 새지 않게 하라. L8 변경과 M8 준비는 필요 없다.

## 6. 기술 검사를 마치면 모든 유효 정책을1,000요청까지 실행하라

필요한 기술 검사는 다음에 집중하라.

1. I1=[24]·gamma=[1]와 native의 loss/target/actual endpoint/history 연결.
2. Weight 고정 상태에서 chunk pause/resume과 같은 총 update의 단일 최적화 연결.
3. FROZEN2의 최초 partial write와 S75 materialization 연결, 올바른 frozen residual.
4. B100 동기화, optimizer counter, clamp/reference carry, history append, branch 격리.

기존 fixture를 재사용하고 필요한 검증만 추가하라. Technical check에 쓴 실행·평가 비용은 본 순차 실행과 구분하라. 단일 B51이 좋거나 나쁘다는 이유로 본 실험 arm을 선택·제외하지 말라. 기술적으로 유효한 정책은 예정된1,000요청을 처리하라.

NaN, 코드·state 정의 불일치, OOM 등은 typed failure와 해당 범위를 기록하고 원인을 수정하라. 성능 악화와 기술 실패를 혼동하지 말라. 조용한 native fallback·case skip·resampling·성능 기반 rollback은 금지한다. 한 비교의 복원 문제로 독립적으로 유효한 다른 경로까지 멈추지 말라.

## 7. 평가와 원분모를 고정하라

매 batch 직후 Current R100/P200/N1000, R/P TF strict, 두 paraphrase request-strict, true/new NLL·margin, 기존 Historical128 R/P/N과 active 구분을 남겨라. B51/B55/B60 checkpoint와 매 batch commit→next entry 연결을 보존하라.

B55에서 suffix500을 재평가하고 B60에서 같은 첫 suffix500의 변화도 산출하라. 최종 주표는 다음을 포함한다.

- **신규1000:** R/P/N, R/P strict와 request-strict, true/new NLL·margin, matched contrast와 N4 대비 차이.
- **신규 유지:** 각 요청의 자기 at-write→W60 lost/gained와 NLL 변화. 미래 노출0인 B60은 따로 표시.
- **기존5000:** ALL과 ACTIVE_TARGET/SUPERSEDED/UNKNOWN의 성능·paired lost/gained.
- **전체6000:** old+new 합계 검산. 전체 평균으로 신규 품질을 대신하지 않음.
- **고정 first-suffix500:** W55→W60 변화. 전체 집합과 공통성공·공통실패 집합을 구분.
- **일반능력 개발 패널:** 최소 terminal Wiki128/MMLUdev32를 같은 reference와 비교.

Active 정의는 six-arm의 raw(subject,relation) 마지막 관측 target_new 문자열과 일치하는 요청이며 same-target 재발행도 포함한다. E01의 다른 active 정의와 혼합하지 말라. Entry 평가가 없는 과거 모집단에서 최종 cross-policy 차이를 직접적인 시간상 forgetting이라고 부르지 말라.

신규와 과거 각각의 paired desired-target NLL 악화량 p95/p99를 보고하라. 성공률, strict, 평균·median·tail을 구분하라. NS 상승이 true 개선인지 competing-new 약화인지 나누어라. R/P도 양쪽 target을 함께 보라.

CI를 계산하면 request cluster와 batch별 차이를 제시하라. P2개/N10개를 독립 요청으로 세지 말고, 한 fixed order의10 batches를 독립10회 반복이라고 하지 말라. CI 하한 양수를 자동 허용 조건으로 쓰지 말라. 공식 P/N·general·audit는 online controller·fallback·quota 선택에 넣지 말라.

## 8. 계산량을 실제로 측정하고 설명하라

1,000요청당 early stop이 없을 때의 최대치는 다음과 같다.

| 정책 | Adam updates | Target loss evaluations | Writer solves | M4 appends |
|---|---:|---:|---:|---:|
| N4 | 24000 | 25000 | 10 | 10 |
| REFIT4 | 48000 | 50000 | 20 | 10 |
| FROZEN2 | 24000 | 25000 | 20 | 10 |
| I2 | 24000 | 26000 | 20 | 10 |
| FROZEN4 | 24000 | 25000 | 40 | 10 |
| I4 | 24000 | 28000 | 40 | 10 |

REFIT4 기존 실측은 Adam27525/loss29525다. I2/I4는 최대 budget을 맞춘 것이며 실제 FLOPs·wall·요청별 수렴 정도까지 맞춘 것이 아니다. 새 runtime을 기존 N4/REFIT4 시간에서 단정하지 말라.

준비, target forward/backward, matrix 준비/solve, readout, write/copy/finalization, 진단, 평가, I/O, 실제 allocation을 분리하라. 중첩 timer를 더하지 말라. 기존 reference 비용과 새 연구 지출을 따로 기록하라. 사용자가 지정하지 않은 GPU-hour hard cap이나 online ratio 탈락선을 만들지 말라. 실제 자원 가용성과 저장량을 확인해 배치하되 제한을 사용자 지시로 꾸며 쓰지 말라.

Request/chunk별 u/Z/a0/aj, Y/residual, m/v/counter carry, loss 성분·stop reason, teacher ID, clamp 기준·hit를 재검산 가능하게 보존하라. Actual subwrite와 필요한 entry/terminal state를 기록하되 전체 모델 checkpoint를 매 chunk 복제하지 말라. 큰 raw/tensor는 local-only, Git에는 compact manifest·집계·보고만 남겨라.

## 9. GH가 claim과 후속 후보를 판단하라

SH는 PROTOCOL에 따라 실행 사실·수치·분모·source identity·실측 비용·오류·미측정만 보고한다. 방법 우열, 원인 추론, 후보 승격과 후속 권고는 GH가 별도 global report에서 작성하라.

GH는 ALLOW / ALLOW_WITH_LIMITED_CLAIM / NEEDS_TARGETED_CHECK / NOT_SUPPORTED / INVALID_COMPARISON 중 하나로 **claim별** 판단하라. 각 판단에는 핵심 대조, 관측 이득, 반대 근거·손실·비용, 허용하는 정확한 문장, 다음 최소 확인을 적어라.

- FROZEN2가 REFIT4의 이득을 유지하면 fresh second target 필요성을 좁혀 판단하라.
- I2−FROZEN2와 I4−FROZEN4에서 이득이 남으면 intermediate written-state 최적화의 정책적 가치를 평가하라.
- I4가 I2보다 낫더라도 추가 비용을 함께 보고 더 단순한 후보를 우선할 수 있다.
- PS/NS 이득과 strict/old-active 손실이 공존하면 좁은 trade-off claim으로 허용할 수 있다.
- Split의 품질 열세를 refresh 무효로 즉시 단정하지 말고 REFIT4의 일부48회 사용·다른 reset 규칙을 함께 적어라.

첫 campaign에 NM4 oracle chain, I4-reset,8-refresh,Euler,모든48-step 변형을 추가하지 말라. Budget 부족이 핵심 쟁점으로 남을 때만 I2-B48=[24,24]·동일 carry·gamma=[.75,1] 한 운영점을 별도로 기록해 검토하라. 이 경우도1,000요청 순차로 수행한다.

Middle 완료 후 구현 가능한 후보 하나를 고정하면, GH가 결과와 선택 이유를 기록하고 **Late 공통 W90/M90→B91–B100의1,000요청** 비교로 이어가라. 정책은 N4/REFIT4/선택후보 최대3개이며 후보가 REFIT4면2개다. 사소한 수치 참고선 초과마다 사용자 재승인을 요청하지 말라. 과학적 근거가 없는 후보를 형식적으로 승격시키지도 말라. 계획 밖 sweep는 자동 확장하지 말라.

## 10. 위임·완료 산출물

GH는 새 instruction ID, 담당 SH, host, entry, 새 run ID, exact execution source/archive, write scope, 자원·저장 계획을 담은 envelope를 발행하라. Server4 자산과 기존 실행을 우선 재사용하되 실제 환경을 확인하라.

구현 위임 범위는 **`project/run_scripts/low_cost_write_donor_pilot/`의 별도 stepper·policy adapter·평가/계측 확장**이다. 전용 `codex/` branch를 사용하고 사용자·다른 작업의 변경을 보존하라. 원 native source를 덮어쓰거나 unrelated 실험을 재개하지 말라. 기준 설계·계약·cell 목록도 실행자가 읽을 수 있게 동일 버전으로 결속하라. Commit/push/merge/실험 제출은 기존 GH/SH 역할별 절차로 관리하라.

최소 완료 산출물은 다음과 같다.

1. Evidence-reuse manifest, comparison capsule, 새 execution source/import identity와 policy lock.
2. 기술 검사 결과, 실제 실행 범위·종료 상태·누락·오류와 batch state 연결.
3. 여섯 정책의1,000요청 terminal 성능과 old/new/active 분해, at-write·first-suffix500 전이, NLL tails.
4. Request/chunk target·optimizer·subwrite 집계와 실제 비용/메모리/저장 ledger.
5. SH 사실 보고 및 분리된 GH claim decision·후속 후보 선정 근거.

첫 회신에는 **재사용할 reference, 새로 실행할40/50/60 batches의 범위, 구현 변경점, 실행 환경·비용 추정과 첫 확인 시점**을 적어라. 준비가 끝나면 기술 검증을 거쳐 예정된1,000요청 순차 실행과 완료 보고까지 진행하라. 초기 batch의 과학적 점수나 미측정 audit 때문에 이 과정을 다시 단일 batch pilot으로 축소하지 말라.

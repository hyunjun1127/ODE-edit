# EP-TW-1：사전 보존 한도 없는 첫 target–write 실험 사양

작성: 2026-09-15 KST. 상태: 로컬 설계 확정안·구현 대상 사양. **원격 dispatch, 구현 완료, 제출 또는 성능 결과가 아니다.**

근거: [전체 파이프라인 재검토](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pipeline-reset-review-ko.md). 구조화 사양: [후보 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-edit-quality-preserving-tw-contract.json). Reference: [C4 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-reference-data-contract.json).

## 1. 첫 방법에서 고정하는 결정

1. 편집 전 W0부터 B100×10, 같은1000요청의 **ours 신규 한 경로**를 시험한다. 단일 batch 성능으로 후보를 선별하지 않는다.
2. Native L4 target 한 번과 native writer를 그대로 proposal로 사용한다. 실제 native provisional endpoint에서 한 번의 residual 보정을 계산한다.
3. 고정 W0 full-vocabulary C4 KL을 최소화하되, 자기 proposal에서 달성한 canonical 평균 NLL과 성공 ID 집합을 유지한다.
4. **N4 사전 chain·보존 한도 b·TV ρ·μbase·log barrier는 사용하지 않는다.** Current-quality 우선순위, native clamp, solver trust, 작은 후보 메뉴는 명시적 설계 선택으로 남는다.
5. 첫 버전은 old replay·두 번째 target refresh·CBF·response medoid를 넣지 않는다. Old ledger와 평가 자료는 저장한다.

EP-TW-1은 보존을 전역 보장하는 방법이 아니다. 이름의 edit-quality-preserving도 아래의 **유한 current screen 범위**를 뜻한다. 원래 PDF의 endpoint frontier를 저비용으로 근사하는 후보이며 ODE의 필요성을 가정하지 않는다.

## 2. 입력·상태

- W0: 기존 pre-edit FP32 Meta-Llama-3-8B-Instruct capsule, revision·tokenizer·contexts·order hash를 봉인한다. 다른 방법의 warm W50/W90 weight/history를 가져오지 않는다.
- 현재 entry Wt-1와 history Mt-1는 자기 branch의 직전 commit이다. B1 history는 W0 native 규약으로 초기화하며 빈 old-label ledger로 시작한다.
- B100을 전부 native target 최적화한 뒤 batch writer를 실행한다. Request별 B1 immediate write로 바꾸지 않는다.
- Native target 상한은 요청당24 Adam updates/25 loss evaluations, 원 early-stop·clamp·context·내부 KL 규약을 유지한다. 첫 방법에 Adam carry/reset 실험을 추가하지 않는다.
- 이 inner episode에서 입력 key K, native projection P, covariance/history M, context 반복 T, writer A, token·scoring 위치를 고정한다. 다음 batch에서는 자기 branch에서 다시 계산한다.

실제 native 선형 map은 `S(R)=R A`다. Source의 solve/repeat/transpose에서 A를 추출하며 symmetric-positive-definite 형태를 무검증 가정하지 않는다. L4 이외 weight는 업데이트하지 않는다. 수정 L4는 모든 입력 token에서 작동한다.

## 3. 보존 reference

`C4-WebRef-v2` 문서·token 선택을 그대로 쓴다. Train512=S64+Dev128+Reserve320, validation Report256이다. 첫 목적 D64는 S64 문서64개, 각128 scored positions의 full-vocab `KL(p0||pV)`이며 vocabulary 합→position 평균→문서 평균으로 계산한다. 가중치는1/64다.

입력은 고정 자연 token256+BOS1, 입력 score `[129,257)`, logits `[128,256)`다. Prefix는 teacher-forced 고정 text다. Chat template, 자유 생성 prefix, top-k teacher를 넣지 않는다. Teacher p0는 stream 전체에서 W0로 고정한다.

Reference 768문서·token은 Server4 구축 완료 보고를 재사용한다. 기존 teacher 작업의 마지막 기록은 PENDING이므로 **실행 재개 시 봉인된 산출물의 완료·checksum을 확인**한다. 이번 설계 변경을 이유로 source/token/teacher를 무조건 다시 만들지 않는다.

Dev128은 W5/W10 개발 보고이며 gradient·후보 선택에 쓰지 않는다. 이를 보고 설정을 바꾸면 해당 run은 개발 run이다. Report256 identity는 이미 고정하고 policy lock 이후 loss를 연다. Report를 보고 재선택하면 더는 그 결과를 독립 검증이라고 쓰지 않는다.

## 4. 실제 preview와 품질 조건

Native가 계산한 실제 FP32 endpoint를 Vp, target을 Zp, 실제 delta를 Δp=Vp−Wt-1라 한다. Vp는 **현재 ours 경로에서 당연히 수행하는 native proposal**이다. 별도 N4 실행이나 미리 저장한 N4 endpoint는 필요 없다.

현재 canonical request의 desired answer를 teacher forcing하여

\[
E(V)=\frac1{100}\sum_i\frac1{|y_i|}\sum_k-\log p_V(y_{i,k}\mid x_i,y_{i,<k})
\]

를 계산한다. Native target의 augmented context loss와 이 실제 endpoint canonical loss를 구분한다. Native strict evaluator의 동일 token·tie 규약에서 desired answer 모든 token이 성공한 ID 집합을 A(V)라 한다. `Ep=E(Vp)`, `Ap=A(Vp)`다.

선택 가능 조건은 `E(V)≤Ep` 및 `Ap⊆A(V)`다. NLL은 전체100요청을 사용한다. 성공 개수만 비교하지 않는다. Ap 밖 요청의 개선도 평가하지만 실패 요청을 분모에서 빼지 않는다. 공식 P/N, Historical 정답, Audit/MMLU, FutureN은 controller에 넣지 않는다.

이 조건은 평균 품질과 기존 canonical 성공을 지킨다. 모든 요청의 NLL·margin·paraphrase 일반화·old retention을 보장하지 않는다. 이 제한은 결과표와 method 본문에 남긴다.

## 5. 실제 native endpoint를 기준으로 한 보정

후보 좌표는 `V(C)=Vp+C A`이고 초기 C=0이다. FP32에서 `Wt-1+(Zp-H)A`로 raw를 다시 만들지 않는다. **C=0은 저장한 native endpoint의 bytes를 직접 사용**한다. Candidate materialization의 실제 차이와 수학적 CA를 함께 기록한다.

Vp에서 current와 S64 gradient를 따로 누적한다.

\[
g_E=\nabla_C E(V(C))|_0,\quad g_D=\nabla_C D_{64}(V(C))|_0,
\qquad\nabla_C F=\nabla_VF\,A^\top.
\]

A는 detach하며 inverse/HVP를 미분하지 않는다. Current 요청 평균과 control 문서 평균의 microbatch 가중치를 정확히 누적한다. `E+μD` 한 scalar를 backward한 결과로 두 gradient를 대신하지 않는다. 두 gradient를 얻는 실제 F/B·token·메모리를 각각 계상한다.

`q=〈gE,gD〉`로 두고 `gE≠0`이면

\[
d=-g_D+\frac{\min(q,0)}{\|g_E\|^2}g_E.
\]

이는 `〈gE,d〉≤0` 안에서 `-gD`에 가장 가까운 residual-Euclidean 방향이다. gE=0이면 d=−gD지만 유한 quality 검사는 그대로 필요하다. 정확한 1차 부등식·KKT 값은 로그에서 점검하되 성공 보증으로 쓰지 않는다.

초기 trust는 ζ=.25, `α=min(αcap, ζ||Δp||/(||dA||+εnum))`다. αcap·εnum·near-zero 판정은 numerical manifest에 기술 검증으로 고정한다. 최적 성능값이라고 주장하지 않는다. dA=0, nonfinite gradient, zero native action 등의 상태는 typed reason으로 보정을 생략한다. Native proposal 자체가 비유한 상태면 run의 기술 실패이며 성공적인 native fallback으로 숨기지 않는다.

`Ztmp=Zp+αd`를 각 request의 **동일 native anchor/radius ball**에 투영해 Zproj를 얻고 C=Zproj−Zp로 둔다. 필요하면 C에 [0,1] 계수를 곱해 `||C A||≤ζ||Δp||`를 만족시킨다. Zp가 원 ball 안에 있다는 기술 검사가 선행되어야 convex interpolation 설명이 성립한다.

Ball 투영 후에는 원 d의 1차 부등식이 유지된다고 가정하지 않는다. 실제 FP32 materialization의 correction norm·ball·품질을 검사한다. Numerical trust 검사까지 실패한 보정 후보는 제외한다. Native clamp를 없애 얻은 효과로 비교하지 않는다.

## 6. 네 후보와 최종 transaction

| ID | 실제 후보 | 의미 |
| --- | --- | --- |
| RAW | Vp | native 품질을 이미 달성한 항상 포함되는 후보 |
| C1 | FP32(Vp+C A) | 전체 보정 |
| C05 | FP32(Vp+.5 C A) | 보정만 절반 |
| C025 | FP32(Vp+.25 C A) | 보정만1/4 |

각 후보는 같은 Vp에서 별도 materialize한다. Previous candidate에서 덧붙이지 않는다. 동일 bytes 후보는 한 번만 평가하고 중복 사유를 남긴다. `.5/.25`는 native delta 전체를 줄이는 scale이 아니다.

실제 candidate forward의 finite loss와 current quality 조건을 확인하고 **통과 후보 중 D64 최소**를 고른다. 그다음 tie는 RAW 우선, 이후 actual correction norm이 작은 순, 고정 ID 순으로 처리한다. KL 차이가 numerical resolution 안이라 확실하지 않으면 RAW를 우선한다. 요청 loss의 양의 완화량을 tuning하지 않는다. 기술 비교는 고정 kernel·reduction에서 수행하며 애매한 경계는 재검사 또는 RAW로 처리한다.

보정 후보가 모두 실패하거나 개선하지 못하면 RAW를 commit한다. Parent 유지/편집 거절은 preservation 정책의 fallback이 아니다. Native request 실패·부분 성공은 native와 동일하게 실제 모델·원 requested 분모에 남는다.

최종 native finalizer는 processed B100당 정확히1회 호출한다. Inner candidate preview에서 history append는0이다. Accepted-label ledger는 최종 모델에서 strict 성공한 요청만 실제 label·loss·ID와 함께 등록한다. Exact subject/relation conflict는 고정 task 규칙으로 active/superseded를 갱신하고 그 범위를 semantic conflict 전체 해결이라고 부르지 않는다. Unaccepted current label로 과거 accepted label을 자동 대체하지 않는다.

Preview·candidate 복원은 weight뿐 아니라 RNG, cache, history, hooks, optimizer state의 부작용을 포함한다. 장애 시 마지막 봉인된 batch entry/commit 상태를 구분해 복원하며 finalizer를 두 번 호출하지 않는다. **Committed write는1회지만 provisional materialization·forward는 여러 번**이다. 비용 보고에서 이를 한 번의 model forward로 표현하지 않는다.

## 7. 기술 검사와 admission 교체

Reference 준비와 실제 모델 기술 검사는 필요하다. N4 calibration은 필요하지 않다.

- 기존768 문서·token identity와 W0 teacher manifest·self-KL·score shift를 확인한다.
- C=0 native bytes/metrics parity, source native write와 gradient용 A의 차이를 기록한다.
- 같은 실제 endpoint의 gE/gD를 별도 방향미분으로 확인하고 ball/trust/복원 검사를 한다. FP32 해상도에 맞는 perturbation 검사를 사용한다.
- gE=0·gD=0·충돌·nonfinite·전 후보 실패·동률·중복 candidate와 요청 ID set 검사를 확인한다.
- Batch history1회, ledger, save/resume의 실제 모델 동작을 확인한다.

기존 `verify_dispatch → calibrated_budget → scientific_admission` 경로와 old `choose_candidate`를 그대로 사용하지 않는다. **EP-TW-1 전용 versioned policy/admission**을 만들고 N4를 입력 목록에서 제거한다. Old BG-1의 39개 CPU fixture PASS는 참고이며 새 adapter/GPU 검사를 대체하지 않는다. b를 매우 크게 설정하거나 calibration 상태를 PASS로 가장해 우회하지 않는다.

기존 V1 dispatch와 teacher source lock은 변경하지 않는다. 새 설계를 원격으로 집행하는 단계에서는 새 instruction ID/source manifest가 필요하다. 이번 문서는 그 전달이나 실행 재개를 수행하지 않는다.

## 8. 첫 과학 실험: W0 SEQ1000 한 경로

공통 W0·순서·contexts에서 EP-TW-1을 B1–B10에 적용한다. 과학적 run의 성능을 첫 B1로 선별하거나 native보다 좋아질 때까지 seed를 바꾸지 않는다. 기술 오류는 원인을 수정하고 새 attempt로 기록한다.

비교 목록은 AlphaEdit, MEMIT, AlphaEdit-BLUE, MEMIT-BLUE, AlphaEdit-L4_only(N4), REFIT4다. 이번 **새 chain 제출 목록은 EP-TW-1 하나**다. 기존 baseline은 W0/order/source/checkpoint identity가 맞는 범위에서 재사용한다. 남아 있는 B1·B5·B10의 C4 평가는 비교를 위한 선택적 추가 forward이며 EP-TW 시작 조건이 아니다. 없는 endpoint의 metric을 추정해 채우지 않는다.

이전 W50 suffix N4304.89초/B100은 예전 warm 환경의 참고값이다. 1.5× 목표와2× allocation 추정은 현행 자원 계약과 실제 W0 profiling으로 검토할 운영값이며 N4 새 실험을 요구하는 scientific gate가 아니다.

한 번의 correction을 위해 generic backward에 S64×128=8192 scored positions, S64×257=16448 input tokens가 노출된다. Current100 backward는 추가다. 네 후보를 모두 generic forward 평가하면 최대32768 scored positions이며 raw gradient forward를 재사용하면 중복을 다시 세지 않는다. 재검사·rejected probe·teacher read·solve·materialize·I/O를 실제 비용으로 기록한다.

## 9. 로그와 필수 결과표

| 분류 | 기록·평가 |
| --- | --- |
| 현재 batch | entry/raw/selected의 canonical 평균·요청별 NLL, strict ID·margin, R/P/N; P/N은 report process에만 노출 |
| 효과 분해 | selected−own raw의 E/D, candidate별 탈락 사유, projection 전후 inner products, clamp 이후 inner products, native fallback 비율 |
| Endpoint | W10의 전체1000, 각 batch at-write→W10, first500 W5→W10; active/superseded 구분 |
| Old | 현재 batch 이전 accepted requests의 entry/raw/selected/terminal 성능; 첫 버전은 보호 제약이 아님 |
| Reference | S64 매 batch, Dev128 W5/W10, Report256 lock 이후; 문서평균 KL·paired tail·자연 token NLL |
| 상태 | raw/projection/materialized delta hash·norm·cosine, target clamp, candidate restore, finalizer/ledger counter, RNG/cache identity |
| 비용 | native target·solve, current F/B, generic F/B, teacher setup/read, 모든 candidate probe, logging/eval, GPU/host peak |

NS ranking과 true/new NLL을 나눠 보고한다. S64 개선만으로 “원 지식 보존”이라고 쓰지 않는다. KL이 좋아도 current margin/old/P strict tail이 나빠질 수 있다. 모든 요청을 원분모에 포함하며 실제 편집 성공률을 별도로 낸다.

해당 stream은 개발 자료다. Request 단위 paired uncertainty를 쓰고 같은 request의 두 P·열 N을 독립 편집 반복처럼 계산하지 않는다. 단일 order의 문항 bootstrap을 order 재현성으로 대체하지 않는다.

## 10. 다음 모듈을 추가하는 순서

1. **같은 조건의 unprojected 보정**: d=−gD, 같은 ball/trust/menu/quality screen. Projection이 실제 추가 효용을 주는지 확인한다. 새 W0 B100×10 비교이며 비용을 실제 사용량으로 맞춰 읽는다.
2. **Scalar quality comparator**: `Wt-1+ηΔp`, 최초 η∈{1,.875,.75,.5}, 같은 Ep/Ap 조건에서 min D. 방향 변경의 필요성을 확인한다. 동일 solver step 수를 위해 쓸모없는 gradient를 계산하지 않는다.
3. **Quality restoration**: 1차 접선 보정이 finite quality screen에서 반복 탈락하는 경우에만 별도 버전/chain으로 추가한다. 최초 후속안은 finite·trust·Ap·D 개선을 만족하지만 평균 E만 실패한 후보 중 `E-Ep`가 가장 작은 하나를 선택한다. 그 trial에서 current gradient gE′를 한 번 더 계산하고 `-max(E-Ep,0)gE′/||gE′||²`로 위반 복원 후보를 만든다. Native ball·총 correction trust를 다시 적용한 후 실제 E/Ap/D를 한 번 검사한다. 최대 current backward1회·추가 unique candidate1개이며 기존 gradient와 합쳐 current backward2회, 총 후보 최대5개다. 실패하면 RAW를 포함한 기존 통과 후보 중에서 선택한다. Gradient가0/비유한 값이거나 대상 후보가 없으면 이 추가 단계는 생략한다. 이는 선형화된 위반 복원 제안이지 수렴·feasibility 보증이 아니다. 소규모 α 축소의 실패를 NLL 허용량 증가로 해결하지 않는다. **첫 EP-TW-1 실행에는 포함하지 않는다.**
4. **Old retention**: current raw quality와 entry old success의 공동 feasibility·충돌 정책을 정의한 뒤 추가한다. 기존 `.1 nats` tolerance를 자동 상속하지 않는다.
5. **Target refresh/두 committed stage**: 위 단계로 설명되지 않는 이득을 확인할 때 추가한다. 그때 같은 executable endpoint family의 one-shot·비용 대조를 포함한다.

모든 모듈을 미리 실행하는 AND gate가 아니다. 첫 방법이 품질 보존 조건 때문에 거의 native로 복귀하면 그 비율·실패 사유·비용 자체가 결과다. 유효 후보가 적다는 이유만으로 projection이나 ODE가 이론적으로 불가능/필수라고 판단하지 않는다.

Lock 후 주 lifespan 비교는 **W0 B100×100**의 선택 후보와 사용자 지정 다섯 baseline이다. 이미 개발에 쓴 첫1000과 독립 검증 범위를 구분한다. Audit/MMLU는 독립 claim 검증에 배치하며 첫 구현을 막지 않는다. 추가 order·corpus·core 실험은 해당 일반화 주장을 할 때 계획한다.

## 11. 주장 규약

첫 실험으로 확인할 것은 **자기 native 품질을 유지하는 actual-output refinement가 원분모의 sequential 편집·보존·비용에 도움을 주는가**다. 보존 한도·penalty weight 없는 것은 정의상 확인되지만 효능은 아직 미측정이다.

후속 ODE 해석은 [재검토 §8.1](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pipeline-reset-review-ko.md)의 quality-feasible projected dynamics로 분리한다. 그것은 D 한도가 없는 연속시간 이상화이며, 첫 유한-step controller가 exact flow·viability를 구현했다는 주장이 아니다.

Scalar가 같으면 방향 제어의 필요성을 보류한다. Unprojected가 같으면 projection의 독자 효용을 보류한다. C4만 좋아지면 surrogate transfer 문제를 기록한다. Old 손상이 남으면 old 보호를 주장하지 않는다. One-shot이 같으면 multistep/ODE의 필수성을 주장하지 않는다. 실제 경로를 별도 측정하지 않은 경우 barrier bypass를 주장하지 않는다.

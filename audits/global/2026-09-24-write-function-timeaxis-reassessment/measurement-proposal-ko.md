**과거 update의 시간별 기능을 측정하는 수정 실험안 — 2026-09-24**

상태: 검토용 제안. 실행된 실험·새 runner·GH/SH4에 전달한 명령이 아니다. 기존 실행 지시와 봉인된 문서는 수정하지 않았다. 사용자의 마지막 범위 지정에 따라 allocation 및 단층·다층 분석은 제외한다. 대상은 기존 `BASE_ALPHAEDIT` job 42657과 `BASE_MEMIT` job 42658이다.

**1. 연구 질문과 측정 단위**

같은 과거 update를 고정하고, 그 update가 기록된 이후 모델 상태가 바뀔 때 다음 두 궤적을 함께 관찰한다.

- 해당 사실에 대한 현재 모델의 정답·margin 궤적.
- 현재 모델에서 해당 과거 update를 제거했을 때 나타나는 효과의 궤적.

두 번째를 이 문서에서는 **현재 상태에 조건부인 update 기여**라고 부른다. update 자체의 내재적 속성, 그 사실이 저장된 유일한 위치, 그 update가 없는 이력을 처음부터 다시 실행한 결과와 동일하지 않다.

우선 개별 batch의 실제 전체 parameter 변화 `U_s = θ_s − θ_(s−1)`가 확보된 경우를 정의한다. 변경된 모든 parameter block을 포함한다. 특정 block의 성분만 제거한 값을 전체 update의 기여라고 부르지 않는다. 한 batch가 100개 사실을 함께 기록했으면 사실 q에 대한 `U_s`의 기여이지, q만을 기록한 독립 update의 기여가 아니다.

실제 저장 자산이 일부 checkpoint뿐이면 구간 `(a,b]`의 `U_[a,b] = θ_b − θ_a`를 사용한다. 그 구간에 쓰인 사실 집합을 cohort로 연결하고, 기준 시점은 구간 종료 b로 둔다. 개별 사실의 실제 삽입 시점 j와 구간 종료 b를 별도 필드에 보존한다.

**2. 주 estimand: 정확한 score 분해**

q의 목표 답변 y와 고정 경쟁 답변 c를 정하고, 클수록 y에 유리한 score를 사용한다. 기존 evaluator와 호환되는 기본값은 `m(θ,q) = meanNLL(c|q;θ) − meanNLL(y|q;θ)`이다. 답변 길이가 다르면 이것은 길이 정규화 score 차이지 단일 조건부 log-odds가 아니다. y/c 각각의 NLL도 저장한다.

생성 완료 시점 b의 고정 update U에 대해:

```
M_t(q) = m(θ_t,q)
B_t(q) = m(θ_t − U,q)
C_t(q) = M_t(q) − B_t(q)

M_t − M_b = (B_t − B_b) + (C_t − C_b)
```

`B`는 U를 뺀 나머지 모델의 score다. 여기서 배경이라는 말은 모델의 나머지 parameter 상태를 뜻하며 prompt 문맥의 변경이 아니다. `C`는 전체 U의 유한 제거 효과다. U 제거 후 forward를 처음부터 다시 계산한다. 기존 activation을 그대로 재활용한 국소 근사는 주 estimand가 아니다.

이 식은 정의에 따른 항등식이다. 항등식 자체를 새로운 이론적 발견으로 제시하지 않는다. 연구의 내용은 두 항의 시간별 크기·부호·전이·예측 가능성과 후속 update에 대한 개입 결과다.

parameter ledger에 U가 남는다는 것과 현재 weight에서 U가 독립된 기억 성분으로 보존된다는 것도 다르다. 후속 update가 정확히 −U여도 ledger에는 +U와 −U가 함께 기록된다. 선형 score의 경우 이런 완전 상쇄 뒤에도 U의 조건부 제거 효과는 원래와 같을 수 있다. 따라서 C 안정 사례는 ‘제거 기여가 유지된 망각’으로 부르고, 별도의 지식 복원 증거 없이 ‘지식 trace가 온전하다’고 확대하지 않는다.

**3. 동일한 실험을 whole-update factorial로 읽기**

`U = θ_b − θ_a`, `V = θ_t − θ_b`를 놓으면 다음 네 상태가 생긴다.

| 과거 U | 이후 누적 V | 모델 | 의미 |
|---|---|---|---|
| off | off | θ_a | 구간 이전 |
| on | off | θ_b | 기록 구간 종료 |
| off | on | θ_t − U | 현재 모델에서 과거 U만 제거 |
| on | on | θ_t | 실제 현재 모델 |

이때 `C_t−C_b = m(θ_t)−m(θ_t−U)−m(θ_b)+m(θ_a)`다. 즉, **과거 전체 update와 이후 전체 누적 update 사이의 기능적 교차효과**를 측정한다. 특정 공간 경로를 먼저 가정할 필요가 없다.

단, V는 U가 존재했던 실제 이력에서 생성된 값이다. U가 없었던 이력에서 새로 만들어질 V와 같다고 가정하지 않는다. 따라서 이 개입은 현재 parameter 상태의 유한 차분이며, 편집 요청을 처음부터 제외한 재학습의 총효과가 아니다.

또한 이 교차효과는 U와 V를 바꾸어도 대수적으로 대칭이다. 새로운 점을 ‘상호작용을 역방향으로 썼다’에 두지 않는다. 실제 birth ledger와 반복된 이후 평가를 결합하여 어느 과거 update의 기여 궤적이 어떻게 변하는지를 식별하는 것이 시간축의 의미다.

**4. ‘같은 방식’의 최소·확장 정의**

한 rewrite prompt의 C 하나는 필요하지만 충분하지 않다. 같은 margin 기여가 유지되어도 paraphrase별 반응과 비대상 문항에 대한 영향이 달라질 수 있다.

기본 패널은 각 cohort의 rewrite와 기존 paraphrase다. 동일 q, 동일 목표·경쟁 답변, 동일 tokenization으로 시간별 짝을 만든다. 사실별로 `C_t(q)`를 보존하고, cohort 평균만 저장하지 않는다. 확장 패널은 결과를 보기 전에 고정한 작은 비대상 사실 집합이다. 이것은 기능의 범위를 확인하는 보조 측정이며 allocation 분석으로 바꾸지 않는다.

고정 패널 Q에서 기여 벡터 `c_U(t) = [C_t(q)]_(q∈Q)`를 구성한다. 다음을 함께 보고한다.

- 절대 변화: 평균 및 분위수 `|C_t−C_b|`.
- 방향 변화: C의 부호 전환율, paraphrase 사이 기여 불일치.
- 크기와 방향을 분리한 벡터 norm·cosine. norm이 0에 가까우면 cosine은 정의 불안정으로 표시한다.
- 자기 사실의 기여와 비대상 사실에 대한 효과의 동시 변화.

이것은 선택한 패널과 score에 대한 기능 안정성이다. 모델의 모든 입력에서 같은 함수를 구현한다는 보장이 아니다. dose profile `m(θ_t−U+λU)`의 λ∈{0,0.5,1}은 제거 효과의 비선형성을 확인할 필요가 있는 사전 지정 부분집합에서만 보조로 측정한다.

**5. 유지·망각과 기여 변화를 혼동하지 않는 표**

| 사실의 행동 상태 | C 감소 | C 실질적으로 안정 | C 증가 |
|---|---|---|---|
| 유지 | 기여 약화에도 유지 | 양쪽 유지 | 유지하며 기여 강화 |
| 상실 | 상실과 기여 약화 동반 | 기여가 유지되는데 상실 | 기여가 강화됐는데 상실 |

이름은 관찰을 기술한다. 첫 열을 자동으로 ‘update 때문에 망각’, 유지·감소 셀을 자동으로 ‘보상’이라고 부르지 않는다. ΔB와 ΔC의 부호·크기를 추가로 확인한다. 예를 들어 ΔC<0, ΔB>0인 경우를 상쇄 방향의 변화로 표시하고, `ΔB ≥ −ΔC`면 score 감소가 완전히 상쇄된 것으로 구분한다.

기본 안정성 기준은 사전 지정한 score 허용 폭 ε다. 수치 반복 오차의 상한 ε_num과 과학적으로 무시할 변화 ε_practical을 구분하고 `ε ≥ ε_num`을 보장한다. 결과를 본 뒤 원하는 비율이 나오도록 ε를 고르지 않는다. ε 민감도 곡선을 함께 내고, 모집단 평균의 안정성을 주장하면 동등성 구간을 사용한다. 유의하지 않다는 이유만으로 안정이라고 판정하지 않는다.

at-write/anchor에서 실패한 사실, 성공했지만 `C_b≈0`인 사실도 원자료에서 제거하지 않는다. 다만 ‘기록 당시 유효하게 기여하던 update의 약화’ 분석은 `C_b>ε` 집합을 따로 정의한다. C가 0이라고 trace가 없거나 무의미하다고 결론 내리지 않는다. 다른 update가 같은 기능을 중복 수행하면 제거 효과가 작을 수 있다.

**6. 기존 자산으로 가능한 시간 격자**

첨부 목록의 checkpoint는 `W0, W1, W5, W10, W20, …, W100`이다. 현재 tensor의 실제 접근성은 이번 문서 검토에서 새로 확인하지 않았다.

| 구간 update | 기록된 batch | 구간 내 요청 수(B100) | 기준 시점 |
|---|---|---:|---:|
| W1−W0 | B1 | 100 | 1 |
| W5−W1 | B2–B5 | 400 | 5 |
| W10−W5 | B6–B10 | 500 | 10 |
| W20−W10 … W100−W90 | 각각 10 batch | 각각 1,000 | 20 … 100 |

총 12 cohort다. 각 cohort를 자기 기준 시점부터 모든 이후 저장 시점에서 관찰하면 삼각형 격자 `12×13/2=78`개 `(cohort,t)`가 된다. 두 BASE에서 156개의 논리적 ablation cell이다. 대각선 12개/arm은 U를 빼면 이미 있는 직전 checkpoint θ_a와 같으므로 새 parameter 상태가 아니다. 대각선 밖에서 필요한 신규 counterfactual 상태는 최대 66개/arm, 두 arm 132개다. 점수는 패널에 따라 추가로 계산해야 한다. 이 수는 GPU job 수나 실행시간이 아니다.

두 BASE의 실제 endpoint 12개씩과 공통 W0를 합친 25개 상태는 재사용한다. 모든 counterfactual 모델을 디스크에 복제할 필요 없이 한 endpoint를 로드하고 cohort U를 임시 제거·복원할 수 있다. 매회 원본 weight에서 복원하여 저정밀 subtract/add 누적 오차를 막는다.

비용을 제한하는 첫 파일럿의 예시는 두 BASE 모두에서 `(0,1]`을 시점 1/5/10/20/50/100, `(10,20]`을 시점 20/30/50/70/100, `(40,50]`을 시점 50/60/70/90/100에서 관찰하는 것이다. 실제 저장된 endpoint만 사용한다. 이는 방법 검증용 예시이며 효과를 보고 고르는 규칙이 아니다. 전체 보고는 12-cohort 격자로 확장한다. 마지막 cohort는 미래 관찰이 없어 age 0만 가진다는 점을 표시한다.

구간 내 j에 쓰인 사실을 b에서 처음 관찰하면 j→b 동안의 기능 변화를 놓친다. `C_b`를 진짜 at-write 기여 `C_j`라고 부르지 않는다. 이 구간은 birth window와 left truncation으로 기록한다. 실제 batch ledger가 따로 발견되면 그때 더 정밀한 단위로 대체할 수 있다. checkpoint 간 보간 또는 임의 factorization으로 개별 update를 발명하지 않는다.

**7. 여러 cohort가 해결하는 것과 해결하지 못하는 것**

여러 cohort는 B1 특이성, 같은 나이에서의 반복 여부, 같은 endpoint에서 서로 다른 과거 update의 차이를 관찰하게 해 준다. 그러나 `age = t−b`이므로 하나의 고정 순서에서는 age·birth cohort·전체 모델 시점의 자유로운 세 효과를 독립적으로 식별할 수 없다.

초기 보고는 이 격자를 기술적으로 비교한다. ‘나이의 독립적 인과 효과’라는 표현은 사용하지 않는다. 동일 크기의 10-batch cohort를 주 비교에 쓰고 최초 세 cohort의 크기 차이는 별도 표시한다. 나이·전역 상태를 분리하는 더 강한 주장은 다른 순서/배치 위치에 사실을 배정하는 후속 실험이나 명시적인 모형 제약이 필요하다. 이번 기본 분석에 새 full run을 강제하지 않는다.

**8. 후속 update로의 귀속**

과거 U를 고정하고 인접한 저장 시점 `t_(j−1),t_j`에서:

```
dB_j = m(θ_tj − U) − m(θ_t(j−1) − U)
dC_j = C_tj − C_t(j−1)
dM_j = dB_j + dC_j
```

시간 방향으로 더하면 전체 변화로 정확히 돌아간다. 현재 자산으로는 후속 구간의 귀속이고 개별 요청의 귀속은 아니다. dB와 dC를 나란히 보여 어떤 후속 구간에서 사실 score와 과거 U의 기여가 각각 달라졌는지 확인한다.

선택된 후속 구간 V_j의 현재 상태 상호작용은 다음 추가 네 상태로 확인할 수 있다.

```
I(U,V_j;t) = m(θ_t) − m(θ_t−U) − m(θ_t−V_j)
             + m(θ_t−U−V_j)
          = C_U(θ_t) − C_U(θ_t−V_j)
```

U와 V_j는 중복되지 않는 시간 구간이어야 한다. 나머지 후속 update는 실제 값으로 고정한다. I는 현재 배경에서의 쌍별 상호작용이며, 여러 I의 합이 전체 손상이 된다고 가정하지 않는다. chronological dB/dC와 endpoint ablation을 서로 다른 추정량으로 저장한다.

구간 선택은 탐색/검증을 구분한다. 개발 cohort에서 큰 변화 구간과 작은 변화 대조를 고른 규칙을 봉인한 뒤, 다른 cohort·문항에서 확인한다. 최악의 셀만 선택해서 평균 망각을 설명했다고 보고하지 않는다.

**9. relation/object exposure와 연결하는 조건**

exposure는 과거 fact의 목표 y와 경쟁 답변 c를 기준으로 구분한다.

- 같은 relation에서 이후 target이 y인 노출.
- 같은 relation에서 이후 target이 c 또는 다른 경쟁 object인 노출.
- 다른 relation의 동일 object 노출.
- 동일 subject/relation의 의도적 재편집.

이후 편집의 target을 모두 o*라는 하나의 이름으로 묶으면 방향을 혼동한다. 과거 편집 사실의 y를 강화하는 노출과, 비편집 이웃의 정답에 경쟁하는 답변을 강화하는 노출은 같은 통계가 아니다.

먼저 ΔB와 ΔC를 각각 예측한다. 나이·누적 편집 수·기준 margin·cohort 크기·U norm·relation/object 빈도를 함께 기록하고, 미래 평가 구간보다 이후의 정보는 예측 입력에 쓰지 않는다. held-out cohort 또는 relation을 사용한다. 이를 하나의 순서에서 얻은 예측 관계로 보고하며, 개별 노출의 인과 효과는 짝지은 V 제거 같은 별도 개입을 필요로 한다.

locality와 retention이 같은 노출 기전으로 연결된다는 것은 검증할 가설이다. 두 패널에서 목표/경쟁 답변을 정렬한 뒤 동일한 예측 방향과 개입 반응이 나오는지 확인해야 한다. 최초 실행의 완료 조건으로 이 가설의 성공을 요구하지 않는다.

**10. 필요한 gate와 단계 의존성**

기존 E0–E6를 같은 번호로 덮어쓰지 않고 새 검토용 단계명을 쓴다.

| 단계 | 의존성 | 해야 할 일 | 통과 기준 |
|---|---|---|---|
| T0 | 없음 | BASE lineage·checkpoint·전체 update 범위·cohort·target version 결속 | 분석 대상과 제거 단위가 명확함 |
| T1 | T0 | 원래 endpoint 출력 대조, remove/restore 수치 검증 | 동일 모델·evaluator의 재구성과 복원이 검증됨 |
| T2 | T1 | 전체 U의 M/B/C와 birth-window×t 격자 | raw paired rows·분해 residual·결측 사유가 완결됨 |
| T3 | T2 | 상태 전이, chronological dB/dC, 선택 구간 I와 exposure 검증 | 음성 포함 사전 지정 비교를 보고함 |

T0–T1은 technical gate다. 연구 가설에 유리한 효과가 나와야 T3로 가는 scientific gate를 만들지 않는다. CPU toy gate 재실행, 새 z 최적화, 새 editor 구현, continuation 재현은 이 측정의 필수 선행 조건이 아니다. 실제 tensor가 없거나 재구성 실패면 계산 불가를 명시하고 합성 수치로 대신하지 않는다.

T1에는 (1) 원본 vs 재구성 NLL, (2) 동일 forward의 수치 오차, (3) U 제거 후 복원, (4) 구간 시작/끝 차이와 ledger 일치, (5) 모델·token·target identity를 포함한다. M/P/C0 같은 editor 내부 상태는 새 update 생성에 필요하며, 고정 checkpoint forward의 필수 입력이라고 잘못 묶지 않는다.

**11. 표본과 보고 규칙**

overwrite된 옛 target은 유효한 과거 사실의 망각 분모에서 해당 시점부터 검열한다. 다만 고정 target에 대한 역사적 효과 궤적은 따로 유지한다. 분석 도중 y/c를 바꾸면 동일 q의 항등식 비교가 끊기므로 새 fact-version으로 시작한다. 같은 목표를 반복 쓴 사실은 redundancy strata로 분리한다.

rewrites와 paraphrases를 독립 사실로 세지 않는다. case/subject 및 공유 prompt를 고려한 paired cluster 분석을 사용하고, 두 BASE는 동일 문항에 대해 비교하되 각 arm 결과도 별도로 보고한다. 문항 bootstrap은 새로운 편집 순서에 대한 불확실성을 추정하지 않는다.

주 결과는 M/B/C 시간 곡선, ΔB×ΔC 산점도, 유지/상실×C 감소/안정/증가의 비율, 상태 전이, 기여 부호 전환, 후속 구간별 dB/dC다. 하나의 양수 ‘망각 기여율’로 압축하지 않는다. 두 항이 상쇄되면 비율은 음수 또는 100% 초과가 될 수 있다.

‘망각된 active facts 중 C가 안정인 비율’과 ‘전체 처음 성공한 active facts 중 사실은 잊혔지만 C가 안정인 비율’은 분모가 다르므로 둘 다 명시한다. 어느 것도 ‘전체 망각의 몇 %가 parameter trace가 온전해서 생겼다’는 인과 비율이 아니다.

**12. 최소 저장 row**

```
family, original_run_id, source_hash, model_revision, checkpoint_t_hash,
update_id, update_kind(batch|interval), update_start_a, update_end_b,
request_birth_j, birth_window, eval_t, age_from_anchor, cohort_size,
case_id, subject_id, relation_id, target_version, target_y, competitor_c,
prompt_hash, tokenization_hash, panel, target_active, overwrite_reason,
M_t, B_t, C_t, M_b, B_b, C_b, delta_M, delta_B, delta_C,
target_NLL_actual, competitor_NLL_actual,
target_NLL_minus_U, competitor_NLL_minus_U,
native_success, strict_success, generation_EM_if_measured,
epsilon_num, epsilon_practical, contribution_state,
identity_residual, restore_receipt, missing_reason
```

새 generation 평가를 하지 않았다면 TF strict를 free-generation 정확도라고 이름 붙이지 않는다. unavailable / not measured / invalid reconstruction / valid negative를 구분한다.

이 실험의 완료 조건은 **과거 사실의 행동 궤적과, 그 사실의 기록에 참여한 고정 update의 기여 궤적을 분리해 판단할 수 있는 것**이다. 새 편집법·repair·층 배치의 개선을 완료 조건으로 넣지 않는다.

**추가 원문 6편 검토 기록 — 2026-09-24**

PDF를 직접 다운로드하여 본문·수식·실험·참고문헌·부록의 전체 지문을 읽었다. 새로 읽은 분량은 실제 PDF 페이지 기준 155쪽이다. 원문을 읽었다는 말은 저자 코드 재현이나 표의 모든 수치에 대한 독립 재계산을 뜻하지 않는다. 핵심 정의·표·수식은 선택 페이지를 이미지로도 대조했다. 버전·SHA256·페이지별 범위는 `download-manifest.json`, `reading-coverage.json`에 남긴다.

아래의 비판과 이번 설계에 대한 판단은 검토자의 분석이다. 저자들이 주장한 결과와 구분한다. 기존 18편을 이번에 모두 다시 읽었다는 뜻은 아니며, 이전 원문 검토와 관련 절을 재대조했다. allocation 비교를 이번 연구 제안으로 가져오지 않는다.

**1. Spurious Forgetting in Continual Learning of Language Models — ICLR 2025, 66쪽**

[학회 원문](https://proceedings.iclr.cc/paper_files/paper/2025/file/a774503daed55eb53c634847ae071ec7-Paper-Conference.pdf). 본문 pp.1–10, 참고문헌 pp.11–15, 부록 안내 p.16, 부록 A–J pp.17–66 전체 검토.

합성 biography를 학습한 모델에서 새로운 QA task 학습 후 기존 성능이 급락하지만, 기존 인물 일부의 QA로 다시 정렬하면 다른 인물의 QA 성능도 회복되는 현상을 분석한다. 따라서 답을 못 한다는 것과 관련 지식이 완전히 소실됐다는 것을 구분해야 한다는 강한 선행 근거다. 회복 실험에는 추가 학습이 들어가므로 특정 과거 write가 동일한 기능을 유지했다는 직접 증거는 아니다.

사용자가 제공한 리뷰보다 더 가까운 부분은 **p.9 §5.1의 task-vector 제거**다. Task 1의 초기 12/14/16/18 epoch 변화량을 저장하고 여러 checkpoint에서 계수를 바꾸어 뺀다. 부록 H p.47에 설정이 나온다. ‘고정된 과거 변화량을 나중 모델에서 제거한다’는 조작 자체는 이미 존재한다. 그러나 각 사실의 기록 update를 birth cohort에 연결해 그 기여의 유지와 사실의 유지를 함께 분해한 실험은 아니다.

부록 I.3 pp.55–59의 continual knowledge editing은 Llama3-8B-Instruct, zsRE 1만 건을 10 task로 나눈 **sequential fine-tuning**이다. 이를 MEMIT/AlphaEdit의 원본 update ledger 분석으로 인용하면 안 된다. 여러 task의 triangular accuracy 표가 이미 있으므로 cohort×time heatmap 자체도 신규성으로 삼기 어렵다.

이론 검토에서 별도의 제한을 확인했다. 부록 F pp.23–25는 `(1+δ)^L−1`을 1차 근사한 뒤 `≤Lδ`라는 엄밀한 bound로 적는다. 검토자 반례로 스칼라 `W1=W2=δ>0`이면 차이는 `2δ+δ²>2δ`다. 따라서 그 finite-step bound를 그대로 우리 설계의 보장으로 인용할 수 없다. 이 지적은 spurious forgetting의 관찰 결과를 반박하는 것이 아니라 이론의 적용 범위를 제한한다.

**2. Suppressed, Not Erased: A Representational Trace of Edited Facts Survives Even Weight-Free Knowledge Editing — arXiv 2609.18985v1, 7쪽**

[원문](https://arxiv.org/pdf/2609.18985). 본문·한계·참고문헌·부록을 포함한 pp.1–7 전체 검토.

GPT-2-XL의 CounterFact 50개를 ROME, FT-L, GRACE로 편집하고, 생성상 성공 뒤에도 이전 답변 관련 내부 신호가 남는지 조사한다. 핵심은 새로 기록한 과거 write의 이후 기능이 아니라 **덮어쓴 원래 답변의 잔존 신호**다. sequential 추적은 하지 않는다.

측정 해석에는 큰 주의가 필요하다. p.3의 probe는 원래 object의 identity를 직접 복원하는 분류기가 아니다. hidden state의 original/new object 방향 투영을 비교해 이진 dominance label을 만든 뒤, hidden state로 그 label을 예측한다. 저자도 같은 페이지에서 이 정의를 설명한다. 따라서 ‘원래 사실을 96% 복원했다’고 쓰는 것은 과장이다. label 구성·class balance·best-layer 선택을 함께 고려해야 한다.

GRACE에서는 50개 중 36개의 비유한 hidden state를 제외하여 probe 분석에 14개가 남는다. ROME 0.96, FT-L 0.86, GRACE 0.79라는 수치를 같은 유효 표본 수와 같은 강도의 증거로 나란히 쓰면 안 된다. relearning savings는 저자도 신뢰할 수 없는 음성적 방법론 결과로 취급한다.

이번 연구에 가져올 것은 행동과 내부 상태의 구분 필요성이다. 이 작은 preprint를 ‘편집 사실의 기억이 일반적으로 보존됨’의 확정 근거로 쓰지는 않는다. original-answer trace와 historically observed update는 분석 대상이 다르다.

**3. Forgetting Is Not a Fix: Path Dependence in Sequential Engram Editing — arXiv 2607.24805v1, 8쪽**

[원문](https://arxiv.org/pdf/2607.24805). 본문·결과·한계·참고문헌·부록 pp.1–8 전체 검토.

세 소형 모델에서 여러 개념을 순차 삭제한다. 처음 모델에서 추출한 engram을 고정해 적용하는 조건과 매 단계 현재 모델에서 다시 추출하는 조건을 비교한다. 순서 효과, 남은 개념의 covariance 이동, 삭제된 답변의 NLL이 후속 삭제 뒤 일부 회복되는 사례를 보고한다. ‘한 번 달성한 편집 효과가 이후에도 고정되는가’라는 문제의식은 직접 겹친다.

다만 six-concept 규모와 제한된 probe의 사례 연구다. 예컨대 Qwen3의 삭제 대상 NLL 13.20→9.88은 부분 회복이며, 초기 모델의 지식 수준으로 완전히 돌아왔다는 결과가 아니다. TinyLlama의 같은 효과는 약하다. 일반 factual editing의 망각률이나 우리 BASE의 효과 크기로 옮겨 쓸 수 없다.

검토상 핵심 구분은 **고정된 벡터의 덧셈은 교환 가능하지만, 상태마다 벡터를 다시 만드는 편집 연산은 교환 가능하지 않다**는 것이다. 두 추출 정책의 차이를 발견했다고 해서 고정 벡터 덧셈 자체가 비가환이라는 결론이 나오지는 않는다. 원 논문의 넓은 compositionality 가설을 시험한 의미와 이 대수적 사실을 구분해야 한다.

기존 effect의 시간적 불안정성에 대한 인접 선행연구로 반드시 다룬다. 그러나 실제 historical U의 조건부 제거 기여 C를 birth부터 추적한 연구와는 구분된다.

**4. Reinforced Lifelong Editing for Language Models (RLEdit) — ICML 2025 / arXiv 2502.05759v4, 23쪽**

[원문 v4](https://arxiv.org/pdf/2502.05759v4). 본문 pp.1–9, 참고문헌 pp.10–12, 부록 A–F pp.13–23 전체 검토.

hypernetwork가 현재 편집 gradient로 parameter update를 만들고, 여러 편집에 걸친 보상을 통해 학습한다. p.4 §3.1.2의 memory backtracking은 과거 사실 loss를 명시적으로 포함한다. 부록 A.5는 decay 0.95, backtracking depth 10을 제시한다. 따라서 ‘후속 편집이 과거 편집에 미치는 영향을 기존 방법은 고려하지 않는다’는 넓은 주장은 틀리다.

구분점은 **과거 사실의 loss를 보존하는 것**과 **그 사실을 기록한 고정 update의 독립된 제거 효과를 관찰하는 것**이다. RLEdit의 memory loss는 전자다. 부록 C의 배포 편집 절차는 학습된 hypernetwork를 적용하며, 학습 중의 trajectory/memory 사용을 배포 때도 같은 방식으로 수행한다고 설명하면 안 된다.

실험은 여러 모델·데이터·길이·batch 구성을 포함한다. 원문 CounterFact의 pairwise probability 지표와 generation 정답률은 다르다. 주요 비교에서 RLEdit이 모든 지표·길이에 걸쳐 AlphaEdit보다 우월한 것도 아니다. 우리 두 BASE의 수치와 조건을 맞추지 않은 순위표는 유효하지 않다.

가산형 hypernetwork update에도 우리 estimand를 정의할 수 있다는 구조적 근거는 되지만, 이번 BASE 두 개의 결과를 hypernetwork에 대한 실증 결과라고 확장할 수는 없다.

**5. AI Engram: In Search of Memory Traces in Artificial Intelligence — arXiv 2606.14997v1, 26쪽**

[원문 v1](https://arxiv.org/pdf/2606.14997v1). 본문 pp.1–9, 참고문헌 pp.10–12, 부록 A–H pp.13–26 전체 검토. PDF는 ICML 2026 형식을 사용한다.

target/reference activation 통계로 trained weight의 개념 관련 성분을 추정하고, 주입·제거와 결합을 평가한다. 필요성·충분성·선택성·재활성화를 명시적으로 formalize한다는 점에서 **trace의 기능을 인과적으로 검사한다**는 넓은 주장에 매우 가까운 선행연구다.

가장 중요한 차이는 부록 H p.26에 명시되어 있다. 추정 성분은 수렴한 모델의 기능적 성분이지, 각 개념이 실제 학습 과정에서 발생시킨 parameter 변화의 복원은 아니다. 또한 현재 방법은 retrospective 추출이며 online temporal tracking을 하지 않는다고 제한한다. 이번 연구는 observed historical U의 출생 이력을 고정한다는 점을 활용해야 한다.

부록 F pp.24–25는 가산적 조합과 순서에 독립적인 이상화된 기억 상태의 가설을 제시한다. 이를 실제 모든 sequential editing의 엄밀한 보장으로 읽을 수 없다.

원문 이론에도 검토할 문제가 있다. 본문 Proposition 4.1 및 부록 A는 hard constraint `A X−=0` 아래의 해로 `A=ΔW Σ+(Σ++Σ−)†`를 제시하지만, 일반적으로 이 식은 soft least-squares 해다. 검토자 스칼라 반례 `X+=X−=1, ΔW=1`에서는 A=1/2가 나와 hard constraint를 위반한다. pseudo-inverse는 모순된 concatenated 선형 제약을 정확하게 충족시키지 않는다. 따라서 이 공식을 개념의 완전 독립성을 보장하는 증명으로 가져오지 않는다.

현재 방향의 장점은 engram 식별 문제까지 새로 풀겠다고 하지 않고, 실제 저장된 변화량을 관찰 단위로 삼을 수 있다는 점이다. 다만 historical U 역시 독립적이고 유일한 지식 성분이라는 보장은 없다는 경계를 유지해야 한다.

**6. Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing — ACL 2025, 25쪽**

[학회 원문](https://aclanthology.org/2025.acl-long.868.pdf). 본문 pp.1–9, 참고문헌 pp.9–12, 부록 A–D pp.13–25 전체 검토.

Llama3와 Qwen 계열에서 표준 편집 성공 후에도 old-object를 포함한 문맥으로 이전 답변을 다시 유도할 수 있음을 분석한다. activation 교체, attention-head 및 singular-vector ablation으로 해당 출력에 기여하는 부분을 검사한다. 행동상 편집 성공과 내부 정보·경로를 구분하는 선행연구로, 사용자가 제시한 목록에 추가할 필요가 있다.

부록 A p.13의 dataset 구성은 ROME/MEMIT/MEND에서 공격이 성공한 사례를 모아 합치는 절차를 포함한다. 그러므로 특정 조건의 70%대 수치를 원래 CounterFact 전체의 자연 발생률로 인용하면 안 된다. original-object가 prompt에 제공되는 시험이므로 재출력의 존재만으로 parameter 안의 독립적 잔존 지식을 유일하게 식별하지도 않는다.

부록 B는 efficacy/generalization 부등호가 본문 표기와 역전된 듯한 부분도 있다. 수치를 직접 재현하려면 evaluator를 확인해야 하며, 여기서는 표의 숫자를 새로 계산했다는 주장을 하지 않는다.

이번 질문과의 경계는 **문맥을 바꾸었을 때의 편집 효과**와 **입력을 고정하고 후속 parameter update가 누적될 때 같은 historical U의 효과**다. 기존 해석기법은 참고할 수 있지만 이 논문의 공간적 기전 분석을 이번 연구의 주축으로 가져오지 않는다.

**기존 원문 검토와 연결해 바뀌는 판단**

| 선행 축 | 인정해야 할 선행 내용 | 이번에 남길 좁은 질문 |
|---|---|---|
| Spurious / Superficial / Suppressed | 행동 성공·실패가 내부 지식 잔존과 같지 않음 | 같은 historical U의 기능 궤적은 행동 궤적과 어떻게 분리되는가 |
| AI Engram / Forgetting Is Not a Fix | 기억 성분 제거·조합과 효과의 상태 의존성 | 실제 write provenance를 가진 U를 birth 이후 반복 측정할 때 무엇이 달라지는가 |
| AlphaEdit / MEMIT / SimIE | 보존 제약, 누적 편집, 표현 불변 가정의 한계 | 최종 사실 보존과 해당 U의 조건부 기여를 별도로 측정하는가 |
| DeltaEdit | 과거 update의 활성과 update 사이 간섭 | 같은 과거 U에 대한 birth-anchored 기능 변화의 규모·전이·예측성 |
| RLEdit / TamEdit / StableEdit | 이력과 trajectory를 이용한 장기 편집 | 결과 수준 안정성과 historical-U 기여 안정성의 불일치 |
| WISE / MEMOIR / SoLA | 보관한 편집 기억을 다시 호출하는 방식이 중요함 | 우리 BASE에서의 관측·개입 결과를 어디까지 일반화할 수 있는가 |

이 표는 ‘이 조합이 전 세계 최초’라는 인증이 아니다. 이 검토 범위에서는 사실 유지와 observed historical write의 birth-anchored 조건부 기여를 함께 추적·분해한 동일 설계를 확인하지 못했다. 항등식, 시간 heatmap, 제거 조작 각각은 단독 신규성이 약하다.

원문 메타데이터 주의: 다운로드된 `Suppressed`와 `Not a Fix` 표지에는 arXiv 식별자의 연월 또는 표지 날짜와 불일치하는 날짜 문자열이 보인다. 제목·URL·실제 파일 hash로 식별했고, 정확한 공개 선후관계 주장은 이 날짜들만으로 하지 않는다. 제공된 논문을 독립 재현·검증한 것으로 취급하지 않는다.

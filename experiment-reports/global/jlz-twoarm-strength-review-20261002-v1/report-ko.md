# JLZ 두 arm: 1,000개 편집 구간 strength 상세 점검

검토 시점은 2026-10-02 11:36:57 KST이다. Server4 `attempt-r2-source-inventory`의 main-A(56960), main-B(56961)에서 **공통으로 완료된 B001–B010, BS100×10**을 고정해 분석했다. 예정된 BS100×20의 최종 결과가 아니다. 실행 중인 코드·모델·job은 변경하지 않았고, 모델/GPU 평가를 추가하지 않았다.

**확인된 문제는 rewrite 적합 부족보다 paraphrase 일반화 부족이다.** 두 arm 모두 원래 편집 문장은 거의 완벽하게 맞추지만, 표현이 바뀐 질문으로의 전이는 baseline보다 크게 약하다. 이 격차는 편집 직후부터 있으며 순차 망각으로 생긴 것이 대부분은 아니다. 목적함수·실제 write·commit의 불일치나 최종 clamp로 인한 강도 제한은 이번 자료에서 발견되지 않았다. 계산 낭비를 일으키는 solver 동작과, 원인을 구분하는 데 필요한 진단의 공백은 별도로 확인했다.

**같은 1,000개에서의 결과**

| 방법 | Rewrite 성공률 | Paraphrase 성공률 | Neighborhood 보존률 | Paraphrase TF strict |
|---|---:|---:|---:|---:|
| JLZ A, 추가 Ω 없음 | 100.00% | 67.30% | 87.83% | 38.55% |
| JLZ B, ηΩ 추가 | 100.00% | 68.85% | 87.90% | 40.15% |
| BASE_ALPHAEDIT | 98.90% | 92.70% | 75.10% | 68.35% |
| BASE_MEMIT | 94.90% | 89.15% | 70.18% | 59.30% |
| MEMIT-H, 기존 완료 1k 리뷰 참고 | 99.10% | 90.25% | 83.22% | 62.40% |

Rewrite/Paraphrase 성공은 `NLL(new) < NLL(true)`, Neighborhood는 반대 부등식이다. 분모는 각각 1,000/2,000/10,000이다. TF strict는 정답 prefix를 주는 teacher forcing에서 모든 target token의 argmax가 맞는 비율이며, 자유 생성 성공률은 아니다.

A/B와 AlphaEdit/MEMIT의 raw 13,000행씩을 다시 계산했다. 순서·case ID·문항 index·target token 수가 일치하고, SHA256이 같은 fixed10k 원본으로 각 구현의 문항 identity를 재계산해 모두 일치했다. Native와 JLZ의 identity hash 형식 자체는 다르므로 hash 문자열을 직접 비교하지 않았다. MEMIT-H 행은 이번에 새 raw를 가져온 값이 아니라 기존 완료 1k CPU 리뷰의 검증된 집계다.

Baseline은 과거 Transformers 4.44.2 실행이고 현재 JLZ는 4.57.1이다. MEMIT-H는 다른 서버/H200에서 실행했다. 동일 문항·동일 편집 구간의 품질 비교는 가능하지만, 모든 환경을 통제한 단일 요인 실험이나 시간 배율 비교는 아니다.

JLZ의 canonical target 평균 NLL은 A **0.01929**, B **0.01730**이고 TF strict는 **99.7%/99.8%**다. 같은 endpoint의 paraphrase target NLL은 **3.7496/3.6736**이다. 현재 strength를 단일한 weight 크기나 rewrite 성공률로 요약하면 핵심 결함을 놓친다.

**언제 약해지는가: 편집 직후부터 약하다**

각 요청을 처음 편집한 batch 직후 평가와 W10 평가를 문항별로 연결했다. 아래의 ‘편집 직후 합산’은 서로 다른 W1…W10에서 각 cohort의 current 평가를 모은 것이며, 단일 모델 endpoint가 아니다.

| Paraphrase 추적 | A | B |
|---|---:|---:|
| 편집 직후 합산 | 1,283/2,000 = 64.15% | 1,295/2,000 = 64.75% |
| W10 | 1,346/2,000 = 67.30% | 1,377/2,000 = 68.85% |
| 직후 성공 → W10 실패 | 26 | 13 |
| 직후 실패 → W10 성공 | 89 | 95 |
| 직후와 W10 모두 실패 | 628 | 610 |
| 최종 실패 중 직후부터 실패한 비중 | 628/654 = 96.02% | 610/623 = 97.91% |

Rewrite preference는 1,000개 모두 편집 직후 성공하고 W10에서도 유지됐다. Strict rewrite에는 A 1개/B 2개의 손실이 있지만, 대규모 망각 패턴은 아니다.

Replay가 없는 B1에서도 A PS **68.0%**, B **76.5%**, 두 native baseline은 **87.0%**다. Replay/history 누적만으로 초기 격차를 설명할 수 없다. Current PS가 후반 batch에서 더 낮다는 사실만으로 시간에 따른 붕괴를 주장해서도 안 된다. Batch마다 문항 구성이 다르며 동일 cohort를 추적하면 전체적으로 개선됐다.

단일 token target인 paraphrase 1,970개에서도 A **66.90%**, B **68.58%**다. 다중 token 처리만의 문제는 아니다. 평가 문항이 40개 이상인 모든 relation에서 두 arm은 AlphaEdit보다 낮다. 예를 들어 official-language relation(P37, 104문항)은 A 38.46%, B 42.31%, AlphaEdit 86.54%, MEMIT 83.65%다. 세부 relation 탐색은 사후 분석이며 새로운 학습/선택 기준으로 사용하지 않았다.

**구현과 실제 반영 경로에서 확인한 것**

| 점검 | 확인 결과 |
|---|---|
| 실제 실행 소스 | frozen source archive와 검토한 oracle/run/solver/burden/observer 및 주요 상속 모듈 SHA 일치 |
| Native KL | `current || batch-entry`, 계수 0.0625 |
| Current NLL | target token 평균 → 6 rewrite context 평균 → 100개 요청 합 |
| General reference | 16개, `current || W0` KL, 전체 계수 100×0.0625 |
| Replay | B1 없음, 이후 최대 16개 desired-new NLL, 전체 계수 100×1 |
| Loss 재구성 | 20개 final에서 smooth 재구성 최대 오차 1.30e-8, `smooth + norm = total` 일치 |
| 추가 Ω | A는 raw Ω를 기록하지만 η=0이므로 loss에 추가하지 않음; B는 η=1 |
| Actual write | L4–L8 모두 materialized FP32 weight로 모든 token에 적용; 하위층으로의 dX 유지 |
| 최종 commit | 20/20 weight bitwise 일치, 후보/commit loss maxabs=0 |
| State 연결 | 이전 post와 다음 entry 일치; 매 batch 5개 층 history append |
| Active 요청 | 20개 batch 모두 100/100; inactive 누락 문제 아님 |
| 최종 clamp | 20개 batch 모두 0 hits |
| 최종 zero block | A 7/5,000, B 3/5,000; 모든 층이 거의 모든 요청에 참여 |

선택된 가속 경로는 **dense**이며 direct-R backward 경로가 아니다. Shared preparation의 고정 후보 1개에서 original/dense loss·gradient·요청별 NLL/KL은 일치했고, final original 계산과 마지막 accepted total도 20/20 일치했다. 이 사실은 현재 후보의 일관성 증거이며 전체 수치 영역에 대한 보편적 인증은 아니다.

동일 로컬 tokenizer의 첫 2,000개 canonical 문항에 대해 native packing과 observer packing의 input IDs 및 target 위치가 모두 같음을 CPU로 확인했다. Paraphrase의 baseline/JLZ 문항 identity도 모두 재확인했다. 관측된 큰 PS 격차를 단순 target 위치나 prompt 불일치로 설명할 증거는 없다.

**가장 우선적인 방법적 가설: 학습 문장의 loss를 낮추는 경로와 일반화되는 factual 편집이 다를 수 있다**

현재 공동 최적화는 다음 실제 가중치의 출력 loss를 직접 미분한다.

\[
W_l^{\rm eff}=W_l^{\rm entry}+R_lP_l^\top,\qquad
\Delta h_l(t)=R_lP_l^\top k_l(t).
\]

따라서 다른 층의 입력 변화와 cross-layer 영향은 최적화 그래프에 포함되어 있다. ‘cross effect를 전혀 고려하지 않았다’는 설명은 맞지 않는다. 다만 실제 weight update는 모든 위치에 작용하며, **subject 위치에서 문맥을 넘어 재사용 가능한 변화가 충분히 생겨야 한다는 별도의 제약은 없다.** Subject에서 추출한 key로 write basis를 만들었다고 해서 loss 개선이 subject-mediated 경로로만 이루어지는 것은 아니다.

가능한 설명은 공동 최적화가 canonical과 5개 prefix context에 대해 비용이 작은 경로를 찾고, target 예측 주변 등의 문맥 특이적 효과로 그 loss를 거의 0까지 낮추는 것이다. 그러면 질문의 관계 표현이 바뀐 paraphrase에는 효과가 약할 수 있다. 현재의 RS≈100%, 낮은 at-birth PS는 이 설명과 양립한다. **아직 token 위치별 개입이나 gradient 성분 분해를 하지 않았으므로 원인으로 확정하지 않는다.**

Native baseline도 같은 6개 context를 사용한다. 그러므로 ‘context가 6개라서’만으로 baseline과의 차이를 설명할 수 없다. 중요한 비교 대상은 native의 subject 위치 z 개입이 갖는 구조적 유도와, JLZ의 모든 위치에 대한 실제 weight 최적화가 이용하는 손실 경로다. 실제 write를 포함한 공동 최적화가 z/write 함수 불일치를 줄이는 것과 paraphrase 일반화를 보장하는 것은 서로 다른 요건이다.

여기서 subject 위치만의 개입은 native **z 최적화 중**의 구조다. Native baseline의 최종 weight write도 모든 token에 적용된다. 로컬 EasyEdit의 native compute_z에서 동일 δ를 여러 context의 subject 위치에 넣는 구조를 확인했지만, 이 로컬 파일과 역사적 baseline snapshot의 byte 동일성까지 이번에 새로 인증한 것은 아니다.

규제도 공동 원인 후보이다. 최종 total에서 norm 항의 batch별 비중 평균은 A 88.44%, B 78.72%이며 B의 Ω는 9.96%다. Current NLL 합은 평균 A 0.1900/B 0.1943, 즉 요청당 약 0.0019까지 내려갔다. 이 상태에서 최소 norm을 향한 움직임이 넓은 전이에 필요한 변화를 덜 선호할 가능성은 있다. **작은 NLL과 큰 규제 scalar만으로 gradient 억압을 증명할 수는 없다.** NLL/KL/general/replay/Ω 각각의 gradient norm과 방향이 필요하다.

Reference도 목적에 실제 포함되어 있지만 이번 격차의 단독 원인으로 지목할 수 없다. B1은 replay 없이도 약하고, final의 general/replay scalar는 비교적 작다. Replay는 B100/16의 정규화 때문에 한 항의 직접 계수가 6.25이므로 작은 표본 수가 곧 작은 영향이라는 뜻도 아니다. 성분별 gradient 충돌은 아직 기록되지 않았다.

**층별 배분과 진단의 한계**

최종 R의 clamp radius 사용률은 평균 A 10.25%, B 10.45%, p95는 25.72%/27.29%다. 단순히 clamp가 작아서 더 강하게 쓸 수 없었다는 설명은 이 snapshot과 맞지 않는다. Norm penalty의 연속적인 억압 가능성까지 배제한 것은 아니다.

층별 `D_norm / R_norm` 평균은 A 0.5597/B 0.5643이다. Write가 거의 0으로 소실된 양상은 아니다. 그러나 구현의 `D=R(PᵀK)`에서 **K는 batch-entry에 고정한 key**다. 따라서 D는 그 key에 대한 writer 실현량이며, 공동 편집으로 상위층 key가 바뀐 후의 실제 local displacement나 L8 전달 방향 정렬을 직접 측정한 값이 아니다.

10개 batch의 step별 normalized Ω를 합산한 층별 비중은 다음과 같다.

| 층 | A | B |
|---|---:|---:|
| L4 | 48.06% | 42.88% |
| L5 | 27.37% | 28.39% |
| L6 | 14.04% | 16.44% |
| L7 | 6.63% | 7.82% |
| L8 | 3.89% | 4.48% |

이는 비교용 **step별 Ω의 합**이지 W10 전체 누적 가중치의 에너지나 층별 인과 기여율은 아니다. Raw energy 기준 L4–L6 비중은 A 77.55%, B 74.84%다. B는 L4 비중을 일부 줄였지만, 층별 배분 변화가 일반화 문제를 해결한 증거는 없다. 모든 층을 후보로 유지한다는 사용자 요구는 구현되어 있으며, 이번 결과를 근거로 처음부터 일부 층을 제외할 이유도 없다.

현재 cross panel은 pilot에서 같은 canonical prompt의 subject-last L8 block output에 대해 joint 변화와 단일층 변화의 합을 비교한다. 이는 비가법성 관측이다. 본실험 B1–B10에는 joint 후 local key 변화, 층별 local down-proj 실현량, paraphrase의 subject-to-output 전달 경로를 직접 구분하는 기록이 없다. 따라서 HJ 리뷰에서 제기한 하층 방향 분산 가설이 이번 JLZ에서 해소됐는지는 현재 scalar만으로 검증할 수 없다.

넓은 method 설명과 TeX에는 same-prompt local key/local displacement 진단이 언급되지만, 정본 실행 contract는 pilot의 L8 cross panel로 범위를 좁혔다. 실제 코드는 이 실행 contract에 맞는다. 따라서 이는 실행 contract 위반이 아니라 **방법 설명보다 실행 진단의 범위가 좁아 남은 검증 공백**이다.

**두 arm의 차이는 작고 일관된 일반화 개선으로 보기 어렵다**

B는 A보다 PS +1.55%p, NS +0.07%p다. PS 문항별로는 A 실패→B 성공 148개, A 성공→B 실패 117개다. Cohort별 방향도 섞여 있다. 한 trajectory·한 seed의 중간 결과이므로 B를 일관된 개선으로 확정하지 않는다.

추가 Ω가 있는 B가 실제로 모든 update를 더 작게 만든 것도 아니다. Step별 Ω 합은 A 6.4785/B 6.5365이며, 각 batch의 전체 R norm 평균도 A 8.722/B 8.947이다. 서로 다른 비볼록·미수렴 trajectory와 서로 다른 entry state에서 나온 값이므로 이것이 Ω의 수학 오류를 뜻하지는 않는다. 현재 비교로는 ‘부담을 줄여 일반화를 개선했다’는 의도된 효과가 충분히 나타났다고 보기 어렵다.

**별도로 확정된 계산 효율 문제**

20개 batch 모두 120-call `BUDGET_STOP`이며 최적성 수렴을 주장할 수 없다. 120회는 initial/final과 거절된 trial도 포함한다. Accepted step은 A 합계 907/1,200 calls, B 905/1,200 calls다.

| 큰 BB step에서 시작한 episode | 거절 횟수 |
|---|---:|
| A B3 | 43 |
| A B7 | 39 |
| B B1 | 41 |
| B B4 | 40 |

`1e12` step에서 시작한 halving episode가 총 **163 calls**, 전체 2,400 calls의 **6.79%**를 소비했다. 해당 호출들의 기록된 시간 합은 **2,622.17초 = 43.70분**이다. 두 GPU의 경로 시간을 합산한 값이며, 병렬 실험의 실제 종료 시간이 그만큼 늦어졌다는 의미는 아니다. A B7은 마지막 39개 trial을 모두 거절하며 예산이 끝났다. `sᵀy` 자체는 기록하지 않아 그 값의 부호까지 확인한 것은 아니다.

BB fallback을 이전에 받아들인 유한 step 또는 유한 구간으로 제한하는 개선은 우선순위가 높다. 다만 현재 반환점에서도 rewrite는 충분히 맞고, 이 효율 개선만으로 PS 격차가 사라진다고 예상할 근거는 없다. 변경 후 동일한 절감량이나 동일 trajectory도 보장하지 않는다.

반환 후보는 마지막 accepted다. 14/20 batch는 관측된 최소 total 후보이고, 나머지는 최소보다 0.20–6.32% 높다. 모든 trial의 NLL 분해가 저장된 것은 아니므로 가장 좋은 NLL/PS 후보를 복원할 수는 없다. 반환 규칙을 바꾸면 별도 방법 변경으로 비교해야 한다.

**다음 검증의 우선순위**

1. 짧은 재현 batch에서 최적화 변수를 바꾸기 전에, 실제 all-layer weight update를 이용한 subject/비subject 위치별 진단을 한다. 평가 문항을 학습에 넣지 않고, canonical과 사전에 분리한 표현 변형에서 subject key overlap, local key 변화, local down-proj 변화, L8 방향 및 target NLL을 함께 측정한다. Subject-only 적용은 원인을 가르는 진단용 개입이며 실제 editor를 그 방식으로 대체하는 실험이 아니다.
2. 동일 후보에서 current NLL/native KL/general/replay/Ω의 gradient norm·cosine·합산 상쇄를 소수 지점에서만 계산한다. Norm과 reference가 strength를 억압한다는 가설을 scalar 추측에서 분리한다. 진단을 gate로 사용하거나 최소 사용 층 수를 강제하지 않는다.
3. 위 증거에 따라, 모든 L4–L8을 유지한 채 subject 위치의 변화가 여러 문맥에서 일관되게 실현되도록 하는 공동 목적을 검토한다. 학습용 변형은 held-out 평가 paraphrase와 분리해야 한다. 우선 현재 실패가 목적 구조인지 규제 강도인지 구별하고, 단순 call-cap 확대나 clamp 완화를 첫 처방으로 삼지 않는다.
4. Solver의 큰 step 반복 거절은 별도의 계산 효율 수정으로 다룬다. 작은 고정 후보/짧은 trajectory에서 loss·gradient·채택·반환을 비교한 후 실험 arm에 적용한다. 실행 중인 A/B를 중간에 수정하지 않는다.

현재 실행은 edited checkpoint를 저장하지 않으므로, 저장 JSON만으로 W10의 token별 반사실적 개입을 새로 계산할 수 없다. 위 모델 진단은 다음 짧은 재현 실행에서 후보를 일시 보관하거나 실행 중의 별도 사전 합의된 관측으로 확보해야 한다. 이번 검토에서는 어느 것도 실행하지 않았다.

최소 위치 진단의 구체적인 예는 8요청 × 2문형(canonical 1개와 독립적인 새 paraphrase 1개) × 6상태(entry/all/subject-only/non-subject-only/target-prediction-only/target-prediction 제외) × 2target(true/new), 총 **192 prompt-target-state 평가, backward 0회**다. 이는 192개의 개별 모델 호출을 뜻하지 않으며 microbatch로 묶을 수 있다. 모든 L4–L8을 유지하고, 고정 entry/final 선형 출력 중 token mask에 따라 선택한다. 위치 효과는 비가법적일 수 있으므로 ‘기여율의 합’으로 해석하지 않고 각 상태의 NLL/margin을 직접 비교한다.

**산출물과 재현 근거**

- `snapshot-manifest.json`: 원격 완료 파일 58개의 경로·크기·mtime·SHA256 및 수집 시각. 총 61,640,740 bytes, local hash 재검증 완료.
- `metrics-analysis.json`, `subgroup-analysis.json`: raw 재집계, baseline identity 검증, cohort 추적, W0 변화 및 relation별 분석.
- `loss-components.csv`, `write-components.csv`, `invariants.csv`: 목적함수 복원, 배분, commit/state 검사.
- `solver-batches.csv`, `solver-layers.csv`, `solver-milestones.csv`, `bb-upper-bound-episodes.csv`: 최적화 진행 및 비용 분석.
- 원본 snapshot과 재현용 CPU 스크립트는 같은 worktree의 `local/jlz-strength-review/`에 보존했다. 62MB raw는 Git 추적 대상에 포함하지 않았다.

원자료의 N은 W5/W10에서 전체 seen이며 다른 batch에서는 current 100개의 이웃만 관측한다. 본 보고서의 W10 NS는 항상 10,000개 분모다. W0의 동일 문항 NS는 88.20%로 A/B의 net 변화는 -0.37/-0.30%p다. 다만 W0 성공→실패는 153/160개, 반대 개선은 116/130개여서 평균 보존률만으로 무손상을 주장할 수 없다. AlphaEdit 대비 높은 NS 역시 낮은 PS와 함께 읽어야 하며, 동등 edit strength에서 locality가 개선됐다는 결론은 아직 성립하지 않는다.

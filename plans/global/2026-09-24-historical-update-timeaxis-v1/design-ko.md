**과거 update의 기능과 사실 유지의 시간 궤적 — 실행 설계 v1, 2026-09-24**

상태는 **설계 구체화 완료, 모델 실험 미실행**이다. 이전 [재검토](../../../audits/global/2026-09-24-write-function-timeaxis-reassessment/review-ko.md)를 실행 단위로 구체화했다. 기존 BASE_ALPHAEDIT(job 42657), BASE_MEMIT(job 42658)만 대상으로 삼는다. allocation, 단층·다층 비교, 새 editor·repair·z 최적화는 포함하지 않는다. 기존 GH/SH4 실행 지시를 대체·전송한 문서가 아니다.

**1. 이번 실험에서 답할 질문**

과거 write의 정체성을 고정한 채 다음 세 질문을 순서대로 조사한다.

| 질문 | 주 측정 | 해석 |
|---|---|---|
| 사실이 유지되어도 그 사실의 기록 구간 U의 기여는 변하는가? | 유지된 사실의 ΔC 분포·감소율·부호 반전 | retention으로 historical write의 역할을 대체 평가할 수 있는지 |
| 사실이 상실되어도 U의 기여는 유지되거나 증가하는가? | 상실된 사실의 ΔB/ΔC와 상태 비율 | 망각과 과거 U의 기여 약화가 얼마나 분리되는지 |
| 후속 어느 구간에서 이 구분이 생기는가? | 인접 시점 dB/dC, 선택한 후속 구간 제거 | 과거 U의 역할 변화가 누적 편집의 어느 부분과 연결되는지 |

관측되는 시간 변화와 원인 설명을 구분한다. 앞의 두 질문은 전체 격자로 답하고, 세 번째의 현재 상태 개입은 제한된 추가 셀로 확인한다. 양성 결과를 다음 단계의 통과 조건으로 삼지 않는다.

**2. BASE 및 실제 자산 결속**

두 BASE는 동일한 Meta-Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, FP32 materialized weight, B100×100, 동일한 CounterFact 1만 건 순서를 사용했다. 이 설계에서 ‘원본’은 해당 기존 실행본을 뜻한다. 최신 저자 구현 재실행과 동치라고 하지 않는다.

이번 작업에서 dataset·sample lock·receipt의 실제 파일 hash와 1만 건의 순서를 다시 확인했다. dataset SHA256은 `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`, ordered root는 `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`이다. 모든 요청에 rewrite 1개와 paraphrase 2개가 있다.

24개 checkpoint는 기존 server2 이관 기록에 경로와 hash가 있다. **현재 작업 환경에서는 그 경로의 tensor 24개가 모두 연결돼 있지 않다.** ‘없어진 checkpoint’로 판정하지 않았고, 원격 상태를 조회하지 않았다. [checkpoint-bindings.csv](checkpoint-bindings.csv)에 경로와 기록 hash, [checkpoint-tensor-hashes.csv](checkpoint-tensor-hashes.csv)에 120개 저장 weight의 tensor hash를 결속했다. 실행할 host에서 재검증해야 한다.

편집 가능한 parameter 집합은 실제 저장된 `model.layers.{4,5,6,7,8}.mlp.down_proj.weight` 다섯 개 전체이며 각 shape은 `[4096,14336]`이다. 이는 U의 빠짐없는 재구성을 위한 명세이고, 층별 비교 축이 아니다. 나머지 parameter는 봉인된 W0에서 읽는다. 두 arm의 W0 동일성은 실행 시 hash로 확인한다.

**3. historical update와 시간 단위**

저장 시점은 `0,1,5,10,20,30,40,50,60,70,80,90,100`이다. 인접 저장 시점 `(a,b]`마다 `U=θ_b−θ_a`를 정의한다. U는 해당 구간의 모든 요청이 함께 만든 실제 순변화량이다. 한 사실 q의 독립 update라고 부르지 않는다.

| 기록 구간 | 요청 수 | 이후 저장 시점 수 |
|---|---:|---:|
| (0,1] | 100 | 11 |
| (1,5] | 400 | 10 |
| (5,10] | 500 | 9 |
| (10,20] … (80,90] | 각 1,000 | 8 … 1 |
| (90,100] | 1,000 | 0 |

각 사실의 실제 삽입 batch j와 U의 구간 종료 b를 모두 저장한다. 주 시간축은 `age_from_anchor=t−b`, 편집 노출량은 `100(t−b)`다. `t−j`도 표시할 수 있으나 C_j가 관측된 것은 아니다. j→b 구간의 기능 변화는 관측하지 못한다.

주 시간 비교는 **같은 크기의 8개 구간, b=20…90**이다. `(90,100]`은 기준 측정만 있고 이후 생애가 없으므로 변화율 분모에 넣지 않는다. 첫 세 구간은 별도 표시하고, 전체 12-cohort 표도 공개한다. 한 편집 순서에서 age·birth cohort·모델 전역 시점의 효과를 모두 독립 식별했다고 주장하지 않는다.

**4. score 및 제거 개입**

q의 target은 해당 요청의 `target_new`, 경쟁 답변은 고정된 `target_true`다. 모든 시점과 제거 모델에서 동일 prompt·답변·tokenization을 사용한다.

```
m(θ,q) = meanNLL(target_true | q;θ) − meanNLL(target_new | q;θ)
M_t = m(θ_t,q)
B_t = m(θ_t−U,q)
C_t = M_t−B_t
ΔM = M_t−M_b = (B_t−B_b) + (C_t−C_b) = ΔB+ΔC
```

U를 제거하면 forward를 처음부터 다시 계산한다. activation 재사용이나 국소 선형 근사로 대체하지 않는다. 두 답변의 NLL 및 teacher-forced token 정확도도 각각 저장한다. 여러 token의 평균 NLL 차이이므로 sequence log-odds로 이름 붙이지 않는다.

현재 실제 parameter에서 U만 제거하며, 후속 update는 실제 이력의 값으로 고정한다. ‘U가 처음부터 없었던 학습’의 총효과가 아니다. C는 현재 배경에서의 제거 기여이고, 독립적 지식 trace의 보존 여부가 아니다. 항등식 residual이 작다는 것만으로 올바른 U를 제거했다는 검증도 되지 않는다.

사실 유지의 주 지표는 기존 BASE와 일치하는 pairwise score 성공 `S_t=1[M_t>0]`다. 이는 free-generation 정답률과 구분해 **native pairwise retention**으로 보고한다. strict teacher-forced 성공과 두 paraphrase의 성공도 병기한다. 생성 실험을 하지 않았다면 생성 유지로 확대하지 않는다.

**5. 관측 패널·표본·재편집 처리**

주 실험은 1만 건 모두의 rewrite+paraphrase 2개를 사용한다. 각 U는 자신의 기록 구간에 속한 고정 문항 패널에서 평가한다. prompt별 C를 먼저 저장하며 평균으로 문항 사이 반전을 감추지 않는다.

현재 메타데이터에는 subject–relation 그룹 9,783개, 반복 그룹 125개, 서로 다른 target을 가진 그룹 120개가 있다. 같은 batch 안의 상충 그룹은 5개(요청 10개)다. `fact-ledger.csv`에 사례별 flag와 최초 후속 충돌 시점을 고정했다.

동일 subject–relation에 다른 target이 쓰인 첫 후속 batch부터 이전 fact-version을 **active-fact retention** 분모에서 영구 검열한다. 이후 같은 target이 다시 나타나도 과거 version의 연속 유지가 복구됐다고 하지 않는다. 같은 batch의 상충 요청은 임의 ordinal로 선후를 정하지 않고 해당 version을 active retention에서 제외한다. 동일 target 반복은 유지하되 redundancy 층화로 보고한다. parameter 제거와 score 원자료는 모든 요청에 대해 계산한다.

따라서 active 여부는 관찰 결과를 보고 결정하지 않는다. 현재 메타데이터 기준 anchor에서 active인 요청은 9,966개, t=100에서 active인 요청은 9,784개다. 이 수는 성공률이나 기여 유지율이 아니다. 각 셀에서는 자기 t에 맞춰 검열하며 최종 active 집합으로 과거 시점의 분모를 소급 교체하지 않는다.

**6. 숫자로 고정한 pilot와 주 격자**

pilot는 결과를 이용하지 않고 원래 batch 안에서 case hash 순위로 정한다. B1의 100개 전부, B11–B20과 B41–B50에서 batch당 10개씩 선택하여 총 300개다. 두 BASE가 같은 문항을 쓴다.

| U의 기록 구간 | pilot 평가 시점 | 요청 수/arm |
|---|---|---:|
| (0,1] | 1,5,10,20,50,100 | 100 |
| (10,20] | 20,30,50,70,100 | 100 |
| (40,50] | 50,60,70,90,100 | 100 |

pilot는 32개 논리적 셀, 3,200 fact×time 행, 9,600 prompt×time 행이다. 모든 M/B를 다시 추론하면 target sequence 38,400개다. pilot의 목적은 관측·재구성·저장 경로와 비용을 검증하는 것이며 효과가 작아도 전체 격자를 진행한다.

전체는 arm당 `12×13/2=78`, 합계 **156셀**이다. 실제 모델 상태 25개(W0 공유), 신규 단일 제거 상태 132개를 쓴다. fact×time 111,200행, prompt×time 333,600행이다. M/B 각각 두 답변을 평가하는 보수적인 합계는 **1,334,400 target sequences**다. M_b/B_b를 각 행마다 새로 추론하는 비용은 중복으로 더하지 않는다. 기준 score는 state×prompt cache에서 재사용한다.

parameter 상태의 개수와 GPU job 수는 다르다. panel이 다르면 같은 상태에서도 추가 score가 필요하다. 저장된 실제 M은 identity·kernel·수치 검증이 된 경우만 재사용하고, 재사용 없이 전부 계산하는 경우를 기본 비용 상한으로 둔다. 상태별 평가 문항은 [score-tasks.csv](score-tasks.csv)에 확정한다.

**7. 주 분석량과 판정 기준**

primary ε는 **0.10 nats/target-token**으로 고정한다. 민감도 ε는 0.025, 0.05, 0.10, 0.20이다. 이는 외부에서 검증된 보편 기준이 아니라 이번 설계의 사전 정의된 실질적 허용 폭이다. 연속 분포를 함께 보고하여 ε 하나에 결론을 의존시키지 않는다.

```
C 감소: C_t−C_b < −ε
C 안정: |C_t−C_b| ≤ ε
C 증가: C_t−C_b > ε
유익→유해 반전: C_b>ε and C_t<−ε
```

주 retention×contribution 표는 rewrite에 대해 만든다. 두 paraphrase는 같은 표를 각각 만들고 fact별로 평균 가중한다. panel 전체 기능 변화는 `D_t(q)=sqrt(mean_p[(C_t(q,p)−C_b(q,p))²])`와 문항별 부호 반전으로 측정한다. 세 prompt에서 값이 상쇄될 수 있으므로 `mean(C_t)−mean(C_b)`만으로 안정성을 판정하지 않는다.

초기 유효 기여 집합 `E_b={anchor active, M_b>0, C_b>ε}`에서 다음을 주 보고한다.

| 분석량 | 분자 | 분모 |
|---|---|---|
| 유지된 사실의 기여 약화율 | active_t, S_t=1, ΔC<−ε | E_b 중 active_t, S_t=1 |
| 상실된 사실의 기여 유지·강화율 | active_t, S_t=0, ΔC≥−ε | E_b 중 active_t, S_t=0 |
| 유지된 사실의 유익→유해 전환율 | active_t, S_t=1, C_t<−ε | E_b 중 active_t, S_t=1 |

각 분자는 E_b 전체 active_t 대비 비율도 함께 낸다. 분모 0은 0%가 아니라 NA다. E_b 밖의 요청도 all-anchor-success 및 all-facts 연속 분석에 남긴다. 두 arm의 조건부 집합이 다르므로 arm별 결과와 두 arm 모두 조건을 만족하는 paired 집합 결과를 구분한다.

2×3 셀을 원인 이름으로 바꾸지 않는다. 유지+감소는 자동으로 보상이 아니며 상실+변화도 자동으로 U가 손상된 망각이 아니다. 모든 셀에 ΔB/ΔC의 부호와 크기를 붙인다. ΔC<0, ΔB>0은 상쇄 방향, ΔB≥−ΔC는 score 감소가 완전히 상쇄된 경우로 따로 표시한다.

**8. 시간별 보고와 통계의 단위**

주 그림은 다음 다섯 개다.

1. 각 cohort의 M/B/C와 ΔB/ΔC 시간 궤적. 기준 시점과 결측 구간을 표시한다.
2. cohort×endpoint의 native retention, C 약화율, 유지 중 부호 반전을 나란히 둔 격자.
3. 같은 fact×time의 ΔB–ΔC 산점도와 유지/상실×감소/안정/증가 표.
4. 동일 크기 cohort의 age=10 비교 8개, age=50 비교 4개 및 t=100의 8-cohort 결과.
5. 인접 저장 시점별 dB/dC와 선택 후속 구간 개입 결과.

micro 평균과 cohort macro 평균을 함께 내며, 여러 age를 섞은 한 숫자만으로 결론 내리지 않는다. 최초 기능 변화·최초 상실 시점은 두 checkpoint 사이의 구간으로 기록한다. threshold를 다시 넘어올 수 있으므로 첫 사건을 영구적인 상태로 취급하지 않고 회복·재반전도 기록한다.

**1만 요청은 독립적인 update 1만 개가 아니다.** 주 시간 비교에는 arm당 동일 크기의 historical U 8개가 있다. 같은 U의 문항들은 공동 개입을 공유하고 시간 셀들도 같은 chain을 공유한다. 이 finite dataset에서는 비율·분포·cohort 간 반복을 우선 보고한다. subject–relation cluster bootstrap(2,000회, seed 20260924)을 쓰면 고정된 chain 안의 문항 구성 민감도로 명명한다. 독립 편집 순서나 모델 seed에 대한 신뢰구간으로 해석하지 않는다. rewrite/paraphrase를 독립 표본으로 세지 않는다.

**9. 후속 편집의 시간 구간 귀속**

T2의 격자만으로 인접 시점에 대해 다음을 얻는다.

```
dB_j = B_tj − B_t(j−1)
dC_j = C_tj − C_t(j−1)
dM_j = dB_j+dC_j
```

합은 anchor 이후 전체 변화로 돌아간다. 이것은 실제 이력의 prefix를 따라가는 구간 차분이다. 특정 구간을 처음부터 없앤 학습의 효과나 순서에 무관한 유일 귀속값은 아니다.

추가 제거 실험은 t=100에서 과거 U의 anchor 20/30/40/50을 사용한다. 후속 V 후보는 `(50,60],(60,70],(70,80],(80,90]` 네 개다. 마지막 `(90,100]` 제거는 W90으로 돌아가 dC를 반복하므로 추가 개입 후보에서 제외한다.

U별로 같은 relation의 `target_true`를 후속 target으로 쓴 횟수를 사실별로 세고 합한다. 후보 V 중 이 합이 최대·최소인 두 구간을 선택한다. 동률은 고정 hash로 푼다. 선택은 parameter 효과·NLL·망각 결과를 사용하지 않았다. 정답 target 노출 수도 함께 저장한다. [pair-candidates.csv](pair-candidates.csv)에 모든 후보, [pair-cells.csv](pair-cells.csv)에 선택된 **16셀**이 있다. main state bank 외의 추가 parameter 상태는 16개다.

각 셀에서 M과 minus-U score를 재사용하고 minus-V 및 minus-U-V를 자기 U 패널에서 새로 평가한다.

```
D_M(V) = m(θ_t) − m(θ_t−V)
D_B(V) = m(θ_t−U) − m(θ_t−U−V)
I(U,V;t) = D_M(V)−D_B(V)
         = C_U(θ_t)−C_U(θ_t−V)
```

I<0은 현재 나머지 parameter를 고정했을 때 V의 존재가 U의 기여를 낮추는 방향임을 뜻한다. D_B<0은 U를 뺀 배경에서도 V가 해당 score를 낮추는 방향이다. 여러 V의 효과를 더해 전체 망각 원인 비율로 사용하지 않는다. 두 노출 극단은 무작위 배정이나 완전한 norm/time 매칭이 아니므로 노출 횟수 자체의 인과 효과라고 하지 않는다. 제거되는 전체 V에 대한 조건부 개입이다.

**10. 예측과 locality는 별도 확장으로 둔다**

기본 완료 조건은 T2 전체 궤적과 T3의 위 구간 개입이다. 아래 확장은 이 결과의 존재·부호에 따라 선택하지 말고 실행 여부를 별도 명세로 정한다.

첫 확장은 다음 10-batch 뒤의 망각 예측이다. 기준 모델은 현재 margin, target/competitor NLL, age, 전역 t, arm을 사용한다. 추가 모델은 현재 C, anchor 대비 ΔC와 직전 dC를 더한다. outcome 시점의 B/C나 exposure를 입력으로 쓰지 않는다. t=20…60의 transition으로 학습하고, t=70로 검증하며, t=80/90→90/100을 test로 둔다. subject–relation fold 0–2/3/4를 train/validation/test로 분리하여 같은 그룹의 다른 시점·다른 arm이 경계를 넘지 않게 한다. 모델은 ridge logistic, 표준화는 train에서만 하고 규제 C∈{0.1,1,10}만 validation에서 고른다. Brier/log-loss와 AUROC를 보고한다. 이는 한 chain 안의 미래·다른 그룹으로의 전이이며 새로운 순서에 대한 일반화는 아니다. label 부족이면 NA로 남기며 split을 결과에 맞춰 바꾸지 않는다.

둘째 확장은 relation/object 노출과 locality의 연결이다. 과거 edited target을 강화하는 노출과 경쟁 답변을 강화하는 노출을 분리한다. locality의 정답 방향도 별도로 맞춘다. 기본 패널에 neighborhood 10개 전체를 자동 추가하지 않는다. 같은 통계로 ΔB/ΔC와 locality가 예측되는지, 해당 V 제거에 일관된 반응이 있는지는 별도 확인이다.

**11. 단계 의존성과 완료 기준**

| 단계 | 의존성 | 출력 | 다음 단계 조건 |
|---|---|---|---|
| T0 자산·문항 결속 | 없음 | runtime bindings, token/prompt manifest | 전체 U와 두 BASE의 정체성 확인 |
| T1 재구성·수치 검증 | T0 | endpoint/remove/restore receipt | 아래 runner 계약의 수치·identity 조건 충족 |
| T2P pilot | T1 | 32셀 raw 및 비용 기록 | raw 완결·복원 정확성; 효과 크기 조건 없음 |
| T2F 전체 궤적 | T2P | 156셀, 333,600 prompt 행 | 누락·검열·분모·수치 오차 모두 기록 |
| T3A 시간 구간 분석 | T2F | dB/dC, 상태 전이, primary 표 | 음성 결과도 유효 완료 |
| T3B 후속 구간 제거 | T2F | 사전 선택된 16셀의 D_M/D_B/I | 재구성·paired score 검증 |
| T4 결과 패키지 | T3A+T3B | 다섯 그림, 표, raw, 한계 | 설계와 결과의 차이를 모두 명시 |

T3A와 T3B는 병렬 가능하다. 기존 E3의 성공·실패는 이 DAG의 dependency가 아니다. CPU toy gate나 새 편집 실행은 요구하지 않는다. 자산 미확보·잘못된 복원은 technical blocked로 표시하고, 유효한 음성 결과와 구분한다.

현재의 완성물은 [experiment-contract.json](experiment-contract.json), 실제 case ID가 들어간 CSV, [runner 구현 명세](runner-contract-ko.md), [execution-dag.json](execution-dag.json)이다. 아직 GPU runner를 구현·실행하거나 scheduler에 제출한 것이 아니다.

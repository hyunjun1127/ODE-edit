# 상위 C diagnostic discussion

아래 해석은 이번에 관측한 checkpoint와 request에 한정된다. 결과를 본 뒤 만든 성능 통과 기준, arm 삭제, 보간 또는 scientific promotion은 없다.

## 반복 feedback과 고정 방향

모든 arm은 같은 WN, nominal gradient refresh, fresh momentum을 공유한다. FrozenGlobal↔RefreshedGlobal 비교는 위험 gradient와 Current J의 공동 refresh이며 J 단독효과로 분리되지 않는다. RefreshedLocal은 reference가 We, Global은 W0다. SoftGlobal은 최초 한 번 norm 보정한 penalty이므로 매step fixed-length correction과 크기 궤적이 다를 수 있다.

| endpoint | Current RS | Current PS | Current new NLL delta vs N | Past RS | Past NS |
| --- | --- | --- | --- | --- | --- |
| N_REUSED | 100/100 (100.00%) | 197/200 (98.50%) | 0 | 100/100 (100.00%) | 686/1000 (68.60%) |
| Continue/eval-008 | 100/100 (100.00%) | 197/200 (98.50%) | -0.02165 | 100/100 (100.00%) | 686/1000 (68.60%) |
| FrozenGlobal/eval-008 | 100/100 (100.00%) | 197/200 (98.50%) | -0.0214164 | 100/100 (100.00%) | 691/1000 (69.10%) |
| RefreshedGlobal/eval-008 | 100/100 (100.00%) | 197/200 (98.50%) | -0.0214301 | 100/100 (100.00%) | 691/1000 (69.10%) |
| RefreshedLocal/eval-008 | 100/100 (100.00%) | 197/200 (98.50%) | -0.019967 | 100/100 (100.00%) | 686/1000 (68.60%) |
| SoftGlobal/eval-008 | 100/100 (100.00%) | 197/200 (98.50%) | -0.0213293 | 100/100 (100.00%) | 691/1000 (69.10%) |

Step별 실제 correction norm과 path를 함께 보며 성능 차이를 방향 정보만의 효과로 단정하지 않는다. 작은 correction, 음성 결과, finite defect 모두 같은 분모에 남는다. Classical CBF, monotonicity, 안전성 certificate를 주장하지 않는다.

## 실제 계산량

| entry | process | actual forwards | nominal backward | group backward | small penalty backward | objective/probe sequences | objective wall s | group-J wall s | full-eval wall s | peak allocated bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Middle | C | 37570 | 14000 | 4500 | 40 | 40500 | 1752.846332512796 | 556.3291521798819 | 732.7288042344153 | 46745803264 |

Process total만 합산 가능하다. Child cumulative ledger는 같은 process 내부 차분이며 process total과 다시 더하지 않는다. FLOPs는 측정하지 않아 NOT_RECORDED로 둔다. Native z는 정확한 sealed cache hit이며 cold compute-z 비용을 포함한 속도 비교가 아니다. GPU allocation의 실제 비용은 job ledger를 따로 본다.

## 해석 한계와 다음 질문

세 entry는 서로 다른 Current cohort와 history를 가지므로 시간 경과만의 인과효과가 아니다. Fixed/Past의 exact subject/relation conflict 및 neighbor overlap은 metadata strata로 공개하되 제거하지 않는다. 주된 bootstrap은 request-cluster2000회, subject/relation cluster는 보조이며 optimizer seed/order population uncertainty가 아니다.

가장 중요한 후속 질문: 직접 최적화의 현재 품질과 실제 추가 write 크기를 함께 고려했을 때, 누적위험 방향의 refresh가 단순 contraction보다 재현 가능한 과거 편집 보존 이득을 남기는가? 이번 관측만으로 분리되지 않는 항목은 다음 실험의 사실로 가장하지 않는다.

scientific_promotion=false.

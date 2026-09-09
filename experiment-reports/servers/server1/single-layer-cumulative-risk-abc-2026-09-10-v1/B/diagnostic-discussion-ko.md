# 상위 B diagnostic discussion

아래 해석은 이번에 관측한 checkpoint와 request에 한정된다. 결과를 본 뒤 만든 성능 통과 기준, arm 삭제, 보간 또는 scientific promotion은 없다.

## 방향·amplitude의 관측 비교

GF−/GF+ 부호쌍은 공통 WN에서 시작하고, LF−는 batch-local contraction 설명을, Random1/2는 방향 특이성을 비교한다. OP−와 COV−는 서로 다른 위험함수이며 크기는 같은 native Frobenius 단위로 맞췄다. 수치적으로 unresolved/zero인 방향은 증폭하지 않는다.

| entry | predeclared .1 endpoint | Current RS | Current new NLL delta vs N | Past RS | Past NS | Past NS margin delta vs N |
| --- | --- | --- | --- | --- | --- | --- |
| Early | N_REUSED | 100/100 (100.00%) | 0 | 100/100 (100.00%) | 792/1000 (79.20%) | 0 |
| Early | COVminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.000517898 | 100/100 (100.00%) | 801/1000 (80.10%) | -0.19792 |
| Early | GFminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.000299997 | 100/100 (100.00%) | 798/1000 (79.80%) | -0.0949975 |
| Early | GFplus-amplitude-0.1/eval | 100/100 (100.00%) | -0.000195567 | 100/100 (100.00%) | 789/1000 (78.90%) | 0.0933015 |
| Early | LFminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.00117063 | 100/100 (100.00%) | 795/1000 (79.50%) | -0.0196207 |
| Early | OPminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.00135701 | 100/100 (100.00%) | 805/1000 (80.50%) | -0.202956 |
| Early | Random1-amplitude-0.1/eval | 100/100 (100.00%) | 1.71276e-06 | 100/100 (100.00%) | 792/1000 (79.20%) | 0.00023407 |
| Early | Random2-amplitude-0.1/eval | 100/100 (100.00%) | -2.50411e-06 | 100/100 (100.00%) | 794/1000 (79.40%) | 0.000176417 |
| Late | N_REUSED | 100/100 (100.00%) | 0 | 100/100 (100.00%) | 688/1000 (68.80%) | 0 |
| Late | COVminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.00243699 | 100/100 (100.00%) | 699/1000 (69.90%) | -0.0911628 |
| Late | GFminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.0010902 | 100/100 (100.00%) | 689/1000 (68.90%) | -0.0354724 |
| Late | GFplus-amplitude-0.1/eval | 100/100 (100.00%) | -0.000938017 | 100/100 (100.00%) | 682/1000 (68.20%) | 0.0345769 |
| Late | LFminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.00719297 | 100/100 (100.00%) | 686/1000 (68.60%) | -0.00367309 |
| Late | OPminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.00358751 | 100/100 (100.00%) | 699/1000 (69.90%) | -0.0382089 |
| Late | Random1-amplitude-0.1/eval | 100/100 (100.00%) | 6.29878e-06 | 100/100 (100.00%) | 686/1000 (68.60%) | -3.88218e-05 |
| Late | Random2-amplitude-0.1/eval | 100/100 (100.00%) | 6.89606e-05 | 100/100 (100.00%) | 689/1000 (68.90%) | 0.000508931 |
| Middle | N_REUSED | 100/100 (100.00%) | 0 | 100/100 (100.00%) | 686/1000 (68.60%) | 0 |
| Middle | COVminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.0042828 | 100/100 (100.00%) | 701/1000 (70.10%) | -0.175894 |
| Middle | GFminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.00168307 | 100/100 (100.00%) | 693/1000 (69.30%) | -0.0639133 |
| Middle | GFplus-amplitude-0.1/eval | 100/100 (100.00%) | -0.00137188 | 100/100 (100.00%) | 679/1000 (67.90%) | 0.0631204 |
| Middle | LFminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.0149049 | 100/100 (100.00%) | 685/1000 (68.50%) | -0.00543278 |
| Middle | OPminus-amplitude-0.1/eval | 100/100 (100.00%) | 0.0057133 | 100/100 (100.00%) | 711/1000 (71.10%) | -0.156555 |
| Middle | Random1-amplitude-0.1/eval | 100/100 (100.00%) | 0.000124149 | 100/100 (100.00%) | 687/1000 (68.70%) | -0.00139511 |
| Middle | Random2-amplitude-0.1/eval | 100/100 (100.00%) | -5.91123e-06 | 100/100 (100.00%) | 687/1000 (68.70%) | -0.00153551 |

추가 방향이 현재 성능을 유지하지 못해도 관측값을 제외하지 않는다. Risk 1차항·2차항은 실제 FP32 delta로, intended delta는 별도 열로 기록한다. GF sign-pair 중심차분은 유한 amplitude의 대칭 관측이며 해석적 local derivative의 정확값은 아니다.

## 실제 계산량

| entry | process | actual forwards | nominal backward | group backward | small penalty backward | objective/probe sequences | objective wall s | group-J wall s | full-eval wall s | peak allocated bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Early | B | 11088 | NOT_RECORDED | 300 | NOT_RECORDED | 600 | NOT_RECORDED | 36.826615611091256 | 1028.9790310170501 | 44868391936 |
| Late | B | 11088 | NOT_RECORDED | 300 | NOT_RECORDED | 600 | NOT_RECORDED | 36.80380617454648 | 1034.4148720651865 | 44868391936 |
| Middle | B | 11088 | NOT_RECORDED | 300 | NOT_RECORDED | 600 | NOT_RECORDED | 36.75861998461187 | 1024.7183553446084 | 44868391936 |

Process total만 합산 가능하다. Child cumulative ledger는 같은 process 내부 차분이며 process total과 다시 더하지 않는다. FLOPs는 측정하지 않아 NOT_RECORDED로 둔다. Native z는 정확한 sealed cache hit이며 cold compute-z 비용을 포함한 속도 비교가 아니다. GPU allocation의 실제 비용은 job ledger를 따로 본다.

## 해석 한계와 다음 질문

세 entry는 서로 다른 Current cohort와 history를 가지므로 시간 경과만의 인과효과가 아니다. Fixed/Past의 exact subject/relation conflict 및 neighbor overlap은 metadata strata로 공개하되 제거하지 않는다. 주된 bootstrap은 request-cluster2000회, subject/relation cluster는 보조이며 optimizer seed/order population uncertainty가 아니다.

가장 중요한 후속 질문: 직접 최적화의 현재 품질과 실제 추가 write 크기를 함께 고려했을 때, 누적위험 방향의 refresh가 단순 contraction보다 재현 가능한 과거 편집 보존 이득을 남기는가? 이번 관측만으로 분리되지 않는 항목은 다음 실험의 사실로 가장하지 않는다.

scientific_promotion=false.

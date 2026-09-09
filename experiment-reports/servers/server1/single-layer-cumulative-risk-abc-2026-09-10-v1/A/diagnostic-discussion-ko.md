# 상위 A diagnostic discussion

아래 해석은 이번에 관측한 checkpoint와 request에 한정된다. 결과를 본 뒤 만든 성능 통과 기준, arm 삭제, 보간 또는 scientific promotion은 없다.

## 직접 최적화와 native-assisted support 비교

Direct-B는 native solve의 column support, Direct-C는 저장 P의 전체 orthobasis를 사용한다. Alpha와 eta는 native delta norm에 의존하므로 native-free 실험이 아니다. Middle에서 training objective만으로 선택한 alpha를 Early/Late에 옮겼다.

| entry | method | Current RS | Current PS | Current NS | rewrite new NLL | rephrase new NLL | rephrase new p90 | Fixed RS | Past RS | Past loss/entry-success | Past recovery/entry-failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Early | N | 100/100 (100.00%) | 195/200 (97.50%) | 813/1000 (81.30%) | 0.0015487 | 1.62719 | 5.21841 | 100/100 (100.00%) | 100/100 (100.00%) | 0/100 | 0/0 |
| Early | B | 99/100 (99.00%) | 191/200 (95.50%) | 802/1000 (80.20%) | 0.139581 | 1.83322 | 5.49011 | 100/100 (100.00%) | 100/100 (100.00%) | 0/100 | 0/0 |
| Early | C | 100/100 (100.00%) | 192/200 (96.00%) | 746/1000 (74.60%) | 0.00395517 | 1.56159 | 5.00649 | 100/100 (100.00%) | 100/100 (100.00%) | 0/100 | 0/0 |
| Middle | N | 100/100 (100.00%) | 197/200 (98.50%) | 711/1000 (71.10%) | 0.0266691 | 1.19442 | 3.82367 | 100/100 (100.00%) | 100/100 (100.00%) | 0/100 | 0/0 |
| Middle | B | 100/100 (100.00%) | 196/200 (98.00%) | 716/1000 (71.60%) | 0.058624 | 1.31152 | 4.30185 | 100/100 (100.00%) | 100/100 (100.00%) | 0/100 | 0/0 |
| Middle | C | 100/100 (100.00%) | 192/200 (96.00%) | 678/1000 (67.80%) | 0.00406219 | 0.971624 | 2.97075 | 97/100 (97.00%) | 100/100 (100.00%) | 0/100 | 0/0 |
| Late | N | 100/100 (100.00%) | 193/200 (96.50%) | 626/1000 (62.60%) | 0.016226 | 1.38731 | 4.71889 | 99/100 (99.00%) | 100/100 (100.00%) | 0/100 | 0/0 |
| Late | B | 100/100 (100.00%) | 190/200 (95.00%) | 632/1000 (63.20%) | 0.0104557 | 1.33983 | 4.58239 | 99/100 (99.00%) | 100/100 (100.00%) | 0/100 | 0/0 |
| Late | C | 100/100 (100.00%) | 189/200 (94.50%) | 568/1000 (56.80%) | 0.00252073 | 1.19468 | 3.38491 | 98/100 (98.00%) | 100/100 (100.00%) | 0/100 | 0/0 |

이 표의 loss/recovery는 We 대비이며, W0 대비 inherited degradation과 구분한다. Native와 Direct의 격차는 target fitting, support, regularization 및 update 크기가 함께 바뀐 결과다. 동일 RS라고 동일 NLL/PS/NS인 것도 아니다.

## 실제 strength 범위 — 외삽·보간 없음

| entry | family | metric | observed points | min success % | max success % | min new NLL | max new NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Early | N | RS | 5 | 75 | 100 | 0.000817614 | 2.64447 |
| Early | N | PS | 5 | 62 | 98.5 | 1.50044 | 4.78057 |
| Early | B | RS | 5 | 59 | 99 | 0.139581 | 4.15918 |
| Early | B | PS | 5 | 53 | 95.5 | 1.83322 | 5.66609 |
| Early | C | RS | 5 | 70 | 100 | 0.00395517 | 3.1278 |
| Early | C | PS | 5 | 59 | 96 | 1.56159 | 5.0359 |
| Middle | N | RS | 5 | 93 | 100 | 0.00665054 | 2.17064 |
| Middle | N | PS | 5 | 77 | 98.5 | 1.0566 | 4.21139 |
| Middle | B | RS | 15 | 76 | 100 | 0.00283787 | 3.24915 |
| Middle | B | PS | 15 | 64 | 99.5 | 1.01395 | 4.86672 |
| Middle | C | RS | 15 | 87 | 100 | 0.000973397 | 2.04804 |
| Middle | C | PS | 15 | 69.5 | 99 | 0.405405 | 4.10224 |
| Late | N | RS | 5 | 88 | 100 | 0.00555503 | 2.21094 |
| Late | N | PS | 5 | 69.5 | 97 | 1.25355 | 4.24947 |
| Late | B | RS | 5 | 77 | 100 | 0.0104557 | 3.35255 |
| Late | B | PS | 5 | 58.5 | 95 | 1.33983 | 5.07876 |
| Late | C | RS | 5 | 87 | 100 | 0.00252073 | 1.85564 |
| Late | C | PS | 5 | 66.5 | 94.5 | 1.19468 | 4.2486 |

위 범위의 중첩은 정확한 strength-matched endpoint를 보장하지 않는다. 모든 관측점과 metric을 공개하며, NS가 좋아 보이는 점을 operating point로 선택하지 않는다. Curve neighbor는 2/request이고 Full NS 10/request와 합산하지 않는다.

## 최적화 진행과 native anchor

| entry | candidate | step1 train NLL | step32 train NLL | step32 objective | native action ratio | essence KL | batch net norm | path length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Early | B-alpha-0.02 | 7.87101 | 0.126438 | 0.194489 | 0.4172 | 0.421295 | 5.46473 | 6.28134 |
| Early | C-alpha-0.02 | 7.47843 | 0.0023464 | 0.142718 | 1.22136 | 0.291777 | 4.79321 | 5.38664 |
| Middle | B-alpha-0.02 | 7.59094 | 0.0707366 | 0.136345 | 0.426128 | 0.367929 | 6.90895 | 7.6782 |
| Middle | B-alpha-0.06 | 6.04271 | 0.00585607 | 0.156824 | 1.29722 | 0.339935 | 12.1432 | 13.1263 |
| Middle | B-alpha-0.2 | 4.44895 | 0.00282196 | 0.580957 | 5.56643 | 0.343863 | 25.6366 | 27.0839 |
| Middle | C-alpha-0.02 | 6.79373 | 0.00172856 | 0.191276 | 1.74789 | 0.236138 | 5.12798 | 5.73269 |
| Middle | C-alpha-0.06 | 5.11647 | 0.000900242 | 0.341948 | 3.27171 | 0.222032 | 9.05334 | 10.3972 |
| Middle | C-alpha-0.2 | 4.13284 | 0.000820074 | 1.0455 | 10.3074 | 0.223059 | 21.8217 | 27.679 |
| Late | B-alpha-0.02 | 7.1673 | 0.00985127 | 0.0691134 | 0.419969 | 0.276243 | 7.36167 | 8.22651 |
| Late | C-alpha-0.02 | 6.17671 | 0.0017267 | 0.190269 | 1.77883 | 0.170546 | 4.98741 | 5.59852 |

Train 값은 post-update state에서 측정한다. Native/common observation은 optimizer가 없고 We teacher로 같은 six-context NLL/penalty/KL을 읽었다. 첫 gradient와 업데이트 후32step finite endpoint를 별도로 보존한다. 경로 길이의 합을 net weight norm으로 해석하지 않는다.

## 실제 계산량

| entry | process | actual forwards | backwards | training/observation sequences | objective wall s | eval wall s | peak allocated bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Early | native | 11616 |  | 9800 | 322.19139520451427 | 433.5155772957951 | 40494490624 |
| Early | direct-B-r4 | 15610 | 11200 | 23100 | 1339.2928656563163 | 146.59320946410298 | 37140951040 |
| Early | direct-C-r4 | 15610 | 11200 | 23100 | 1342.2235456239432 | 146.53045925870538 | 39709856768 |
| Late | native | 9116 |  | 4900 | 162.67964141629636 | 450.45896884053946 | 40493507584 |
| Late | direct-B-r4 | 15610 | 11200 | 23100 | 1325.4485928602517 | 147.7249865140766 | 37140951040 |
| Late | direct-C-r4 | 15610 | 11200 | 23100 | 1330.1278514452279 | 147.78510098345578 | 39710020608 |
| Middle | native | 6616 |  |  |  | 440.040871610865 | 40493507584 |
| Middle | direct-B-r1 | 42890 | 33600 | 69300 | 4029.572242902592 | 446.64997493475676 | 37135570944 |
| Middle | direct-C-r2 | 42890 | 33600 | 69300 | 4002.469410955906 | 440.56307100877166 | 39704122368 |

Process total만 합산 가능하다. Child cumulative ledger는 같은 process 내부 차분이며 process total과 다시 더하지 않는다. FLOPs는 측정하지 않아 NOT_RECORDED로 둔다. Native z는 정확한 sealed cache hit이며 cold compute-z 비용을 포함한 속도 비교가 아니다. GPU allocation의 실제 비용은 job ledger를 따로 본다.

## 기록·실행 경계의 공개

- Middle Direct-B(r1)와 Direct-C(r2)는 같은 full logical objective지만 gradient accumulation의 FP32 reduction order가 다르다. 원래 유효 결과를 재실행하지 않았고 source identity를 분리한다.
- Middle의 NumPy C0 diagnostic promotion을 발견하여 Torch FP32 covariance readout으로 37개 저장 state의 algebra-only 구조 관측을 보충했다. Target/writer/optimizer/선택에는 영향이 없다. 원래 raw는 보존한다.
- Native/Early/Late W0 기준 평가는 cross-entry 일부 row가 중복 실행됐다. baseline-reuse receipt에 exact 수량/수치차를 기록한다. 성공분모 추가·평균·값 교체는 하지 않으며 중복 계산 비용을 숨기지 않는다.
- R1/R2 cached-generation token counter는 attention-visible history이고 실제 query token 수가 아니다. 이후 source는 query/visible/padded count를 분리했다. 과거 token counter를 FLOP으로 변환하지 않는다.

## 해석 한계와 다음 질문

세 entry는 서로 다른 Current cohort와 history를 가지므로 시간 경과만의 인과효과가 아니다. Fixed/Past의 exact subject/relation conflict 및 neighbor overlap은 metadata strata로 공개하되 제거하지 않는다. 주된 bootstrap은 request-cluster2000회, subject/relation cluster는 보조이며 optimizer seed/order population uncertainty가 아니다.

가장 중요한 후속 질문: 직접 최적화의 현재 품질과 실제 추가 write 크기를 함께 고려했을 때, 누적위험 방향의 refresh가 단순 contraction보다 재현 가능한 과거 편집 보존 이득을 남기는가? 이번 관측만으로 분리되지 않는 항목은 다음 실험의 사실로 가장하지 않는다.

scientific_promotion=false.

### A — 직접 최적화는 무엇을 바꿨는가

Middle의 training objective 마지막4step 평균만으로 B/C 모두 alpha0.02를 선택했다. Early/Late에서는 재선택하지 않았다. 모든 alpha의 유효 결과를 공개했고 PS/NS로 후보를 제외하지 않았다. Step32가 선택 점수 자체는 아니며 선택에는 step29–32의 post-update objective가 사용됐다.

선택된 Direct-C는 Early/Middle/Late 모두 Current RS100/100이었으나 Current NS는 Native 대비 각각 −6.7/−3.3/−5.8pp였다. Current PS도 Native195/197/193에서 C192/192/189(각200분모)로 낮아졌다. Middle에서는 rewrite new NLL이 Native0.0266691→C0.00406219로 줄면서 rephrase PS는98.5%→96%가 됐다. Current의 적합도·선호 성공·neighborhood 보존을 하나의 성공 지표로 합치면 안 된다.

Direct-B는 제한된 native support에서 더 작은 native-metric action을 사용했다. 선택된 B의 normalized native action은 Early/Middle/Late0.4172/0.4261/0.4200이고 C는1.2214/1.7479/1.7788이었다. 반면 C의 실제 batch-net Frobenius norm은4.7932/5.1280/4.9874로 Native8.7708/10.7440/11.4666보다 작았다. 즉 **작은 Frobenius write가 작은 native-metric 비용이라는 뜻은 아니다.** Support, metric anisotropy, fitting과 regularization이 함께 달라져 support 크기 하나만의 인과효과로 읽지 않는다.

Six-context training NLL과 canonical rewrite NLL 역시 같은 값이 아니다. Middle B step32의 training NLL0.07074는 Native0.04747보다 높지만 전체 training objective는0.13634로 Native0.15806보다 작았다. C training NLL은0.001729로 낮아도 normalized native action1.7479 때문에 objective0.19128이었다. 동일 RS100이 동일 train loss·NLL·regularization 비용을 뜻하지 않는 실제 예다.

Native scaling과 direct curve는 저장된 실제 endpoint만 비교했다. 같은 RS의 포화구간에서도 NLL/PS/NS가 달라 정확한 strength match를 주장하지 않는다. 보간 또는 가장 유리한 NS 지점 선택은 하지 않았다.

### B — 누적위험 방향과 단순 contraction을 얼마나 분리했는가

모든 B trial은 A의 동일 WN에서 독립 시작했다. 미리 정한 amplitude0.1의 full 평가에서 GF−와 GF+는 Current RS가 모든 entry에서100/100이었다. GF−−GF+의 Past NS 차이는 Early+0.9pp, Middle+1.4pp, Late+0.7pp였다. Request-cluster2000회의 탐색적95%구간은 각각[+0.3,+1.6], [+0.4,+2.5], [+0.1,+1.3]pp다. 이는 이번 request-panel의 부호 비교이지, 다중검정 보정된 승자 판정이나 safety gate가 아니다.

GF−는 Native 대비 Past NS가 Early79.2→79.8%, Middle68.6→69.3%, Late68.8→68.9%였다. GF−−LF−는 각각+0.3/+0.8/+0.3pp이고 구간은[−0.1,+0.8]/[+0.2,+1.5]/[−0.1,+0.8]pp다. 단순 batch-local contraction보다 항상 분명한 이득이라고 볼 수는 없다. Raw global/local risk coefficient-gradient cosine은0.3249/0.1592/0.1164였지만 이는 **filter 적용 전 gradient cosine**이며 최종 physical direction cosine으로 바꿔 부르지 않는다.

GF−가 모든 위험 surrogate 중 가장 좋지도 않았다. 같은0.1에서 OP− Past NS는80.5/71.1/69.9%로 GF−보다 높았다. 대신 OP−의 Current rewrite NLL은 각 Native보다 높았고, Middle/Late Past PS는 Native193/191에서191/190(각200분모)로 낮아졌다. COV− 역시 NS가 높아지는 반면 Middle/Late Fixed RS가100→99 및99→98로 줄었다. GF−에서도 Late Fixed RS가99→98로 감소했다. 위험 감소, NS 상승, 기존 edit 보존은 서로 동의어가 아니다.

같은0.1의 actual Frobenius extra norm은 entry별 약0.87708/1.07440/1.14666으로 방향마다 거의 같았다. GF−의 global squared-Frobenius risk 감소와 GF+의 증가, random의 주로 양의2차항은 actual FP32 delta의1차+2차 분해로 기록했다. 이 항등식은 성능/로컬리티 증명은 아니다. .03/.1/.3 전체 curve와 GF sign-pair NLL/margin 중심차분·curvature도 별도 CSV로 공개하며 단일 amplitude를 미분의 정확값으로 간주하지 않는다.

### C — 고정/갱신 및 amplitude 경로

상위 C의 다섯 arm은 동일 Middle WN에서 8step을 끝냈다. 모두 Current RS100/100·PS197/200, Past RS100/100·PS193/200이었다. NS는 Continue의 Current/Fixed/Past710/697/686에서 FrozenGlobal과 RefreshedGlobal 모두714/702/691로 변했다(각1000분모). SoftGlobal도714/702/691, RefreshedLocal은711/698/686이었다. Global 계열의 Past NS+0.5pp를 과거 edit RS 보존 향상이라고 바꿔 부르지 않는다. Fixed RS는 다섯 arm 모두99/100이고, Fixed PS는 Continue/RefreshedLocal195/200, 나머지194/200이었다.

Global 계열의 Past NS 변화는 Continue-success686개 중1개 loss와 Continue-failure314개 중6개 recovery를 합친 순+5/1000이다. Request-cluster2000회95%구간은+0.5pp에 대해[0,+1.1]pp였으며, 별도 성능 PASS 또는 일반화된 효과로 판정하지 않았다. 이 loss/recovery는 Continue reference이고, We reference의 inherited/additional 손실표와 혼합하지 않는다.

핵심 frozen↔refresh 비교에서 **Full3900쌍의 성공 판정이 바뀐 행은0**이었다. 단, endpoint weight SHA와 NLL은 같지 않다. Current rewrite new NLL은 Frozen0.00525272→Refreshed0.00523900, Past rewrite는0.02444763→0.02444616이었다. 전체 RS300/PS600/NS3000행의 new-NLL 차이 최대절댓값은 각각0.0046673/0.0256467/0.0207338이다. 따라서 bitwise equivalence나 모든 request의 동일 출력은 주장하지 않지만, 이번8step에서는 반복 갱신의 추가 성공률 이득이 관측되지 않았다. 이는 refresh 불필요성의 보편적 증명이 아니라 한 Middle state에서의 음성·혼합 진단이다.

RefreshedGlobal은 FrozenGlobal보다 group-J를7회 더 만들고 model forward2100회·group backward2100회를 추가했으며 group-J 구성에259.811초가 들었다. 각 arm의 공통 nominal backward는2800회다. 전체 arm forward는 Frozen6604회, Refreshed8704회로 차이는31.8%이지만 이를 전체 GPU wall-time 비율이나 FLOP 비율로 치환하지 않는다. 초기 J1회는 공통 준비이며 이후7회와 구분한다. RefreshedLocal도7회 추가 구성·259.787초였다. 비용을 더 썼다는 것과 더 유용한 direction을 얻었다는 것은 분리한다.

실제 correction 누적 길이는 Frozen1.07440166, RefreshedGlobal1.07440163, RefreshedLocal1.07440162로 거의 같았고 SoftGlobal은1.06693349였다. 실제 전체 path 길이/추가WN-net norm은 각각 Frozen1.092292/1.091388, RefreshedGlobal1.092096/1.091171, RefreshedLocal1.097993/1.096805, SoftGlobal1.084591/1.083672였다. Continue는0.199849/0.197294다. Path 합을 net norm으로 바꾸지 않았고, Soft의 감소하는 correction 크기는 최초 norm 보정 뒤 재보정하지 않은 결과로 기록했다.

Global risk 감소도 local/native 비용 감소와 같지 않았다. Continue의 global squared-Frobenius4530.633이 Frozen4387.452·RefreshedGlobal4387.452로 낮아지는 동안 native action은138.160→144.281/144.266으로 늘었다. Local L2는115.398→112.880으로 줄어도 history action이22.762→31.401/31.386으로 증가했기 때문이다. RefreshedLocal은 native action112.460과 batch-net norm9.6693으로 더 작지만 global risk4508.794이고 Past NS는Continue와 같은68.6%였다. Risk reference, metric anisotropy와 성능을 하나의 보존 지표로 합치지 않는다.

Current rewrite NLL은 Continue0.00501909, Frozen0.00525272, RefreshedGlobal0.00523900, RefreshedLocal0.00670210, SoftGlobal0.00533978이며, Current rephrase new NLL은 각각1.13669/1.15571/1.15563/1.21818/1.15675였다. 같은 RS/PS라고 같은 target 확률·tail·generation이라는 뜻은 아니다. Full/curve별 paired CSV와 NLL 분포를 함께 공개했으며, 작은 차이·위험 증가·비단조 궤적 때문에 arm을 제외하지 않았다. Frozen↔RefreshedGlobal은 risk gradient와 Current J를 공동 갱신하므로 J 단독의 인과효과로는 분리되지 않는다.

### 공통 한계와 해석 경계

Current/Fixed/Past의 loss/recovery는 We 또는 명시된 공통N reference와 all-panel 분모를 분리한다. 세 entry는 history와 Current cohort가 함께 달라지는 조건부 진단으로 age-only 인과효과가 아니다. Exact metadata 기반 overwrite/conflict/neighbor-overlap strata는 공개하지만 semantic conflict를 완전히 찾았다는 뜻은 아니다. Request/subject-relation bootstrap은 학습 seed·edit-order 모집단 불확실성이 아니다.

A의 Middle B와 C에는 FP32 gradient reduction order 차이가 있고, Middle C0 diagnostic의 NumPy promotion은 저장 state37개의 algebra-only 보충으로 교정했다. 학습/target/선택 결과는 바꾸지 않았다. W0 기준 평가의 cross-entry 중복2717쌍은 원래 관측값과 비용을 그대로 보존했으며 성공 판정 차이는0이었다. 이 중복을 추가 독립 분모로 합산하지 않는다.

Raw normalization/hash/snapshot은 원래 namespace에 보존했다. B plotting의 trial-vs-step schema 오류는 CPU publication만 수리했으며 GPU 재실행/metric 변경은0이었다. Native z는 sealed cache hit이므로 cold compute-z를 포함한 online speed 우위를 주장하지 않는다. FLOPs는 측정하지 않아 NOT_RECORDED다. 각 단계의 model forwards, nominal/group/small-penalty backwards, evaluator/generation 및 allocation 시간은 별도 표로 읽는다.

### 가장 중요한 후속 질문 한 개

**현재 edit 품질과 실제 추가 write 크기를 함께 맞췄을 때, 공동 refresh는 고정된 global 위험 방향보다 재현 가능한 과거 편집 보존 이득을 주는가?** 이번 Middle8step에서 성공 판정 추가 이득은0이고 비용은 증가했으므로, 이를 이미 입증된 feedback 기전 또는 자동 promotion의 근거로 삼지 않는다. 이 질문은 해석상의 미분리 요인을 명시한 것이며 새 실험 제출·설정 변경을 뜻하지 않는다.

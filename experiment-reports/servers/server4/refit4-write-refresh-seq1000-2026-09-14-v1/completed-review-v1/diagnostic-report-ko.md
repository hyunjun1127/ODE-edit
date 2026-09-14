# REFIT4 write-refresh Middle SEQ1000 사실 보고

Instruction: ODEEDIT-S06-REFIT4-WRITE-REFRESH-SEQ1000-SH4-V1. Scientific promotion=false; 최종 claim/후보 선택은 GH 소유다.

## 완료 범위와 비교 기준

공통 L4 W50/M50에서 B51–B60, 고정 ordinal [5000,6000)의 신규 unique1000을 각 정책이 자기 trajectory로 처리한다. 전체표는6정책60logicalbatch이며 N4/REFIT4의 기존20batch를 재사용한다. 신규 main은4정책40batch다. fullseen6000은 과거5000+신규1000이며 신규6000/full10k 실험이 아니다. 기술 native/I1 B100은 별도 분모·비용이다.

| policy | caps | gammas | target_mode | reuse |
|---|---|---|---|---|
| N4 | [24] | [1.0] | native_fresh_batch_entry | True |
| REFIT4 | [24, 24] | [0.75, 1.0] | fresh_current_stage | True |
| FROZEN2 | [24, 0] | [0.75, 1.0] | freeze_own_batch_first_absolute_target | False |
| I2 | [12, 12] | [0.75, 1.0] | carry_absolute_target_with_entry_offset_leaf | False |
| FROZEN4 | [24, 0, 0, 0] | [0.75, 0.75, 0.75, 1.0] | freeze_own_batch_first_absolute_target | False |
| I4 | [6, 6, 6, 6] | [0.75, 0.75, 0.75, 1.0] | carry_absolute_target_with_entry_offset_leaf | False |

여기서 N4는 BLUE-style AlphaEdit singleton L4/L2=1의 fresh native fit이다. blue=False 다섯 layer/L2=10의 원본 AlphaEdit baseline과 다르며 MEMIT 비교도 아니다. 모든 신규 정책은 L4 down_proj만 쓰고 P physical4→asset0→singleton0을 사용한다. L8 write 또는 M8 재구축은 없다.

comparison-capsule.json은 실행 source/archive/config와 기존 reference provenance를 분리한다. Immutable lock에 남은 이전 seq10의 expected90solve/9000z·config8·M8·start_main 항목은 상속된 reference metadata이며 신규 runner가 실행에 사용하지 않는다. 이번 실제 계약은 신규120solve/40historyappend, 기존reference 포함150solve/60append다. 원 lock bytes를 사후 수정하지 않았다.

같은 sample/host/config/entry에서 시작했으나 B52 이후 각자의 W/M/target이 달라진다. 교차 정책 차이는 end-to-end trajectory 비교이며 동일 state에서 한 요소의 인과 기여율이 아니다. 원 reference execution5e96dcb와 새 execution 8a061ea21661d9480acfc740fb888e7bcbdb4216는 구분한다.

## 최종 actual W60 fullseen6000

| policy | RS_n | RS_d | RS_percent | PS_n | PS_d | PS_percent | NS_n | NS_d | NS_percent | RS_delta_n_N4 | PS_delta_n_N4 | NS_delta_n_N4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N4 | 5984 | 6000 | 99.73333333333333 | 11596 | 12000 | 96.63333333333334 | 41621 | 60000 | 69.36833333333333 | 0 | 0 | 0 |
| REFIT4 | 5979 | 6000 | 99.65 | 11599 | 12000 | 96.65833333333333 | 41901 | 60000 | 69.83500000000001 | -5 | 3 | 280 |
| FROZEN2 | 5981 | 6000 | 99.68333333333334 | 11607 | 12000 | 96.72500000000001 | 41466 | 60000 | 69.11 | -3 | 11 | -155 |
| I2 | 5981 | 6000 | 99.68333333333334 | 11613 | 12000 | 96.775 | 41595 | 60000 | 69.325 | -3 | 17 | -26 |
| FROZEN4 | 5977 | 6000 | 99.61666666666666 | 11613 | 12000 | 96.775 | 41431 | 60000 | 69.05166666666666 | -7 | 17 | -190 |
| I4 | 5975 | 6000 | 99.58333333333333 | 11612 | 12000 | 96.76666666666667 | 41473 | 60000 | 69.12166666666667 | -9 | 16 | -148 |

RS/PS는 target-new 평균 token NLL<target-true, NS는 true<new이며 tie는실패다. TFstrict/tokenaccuracy는 보조이며 자유생성 의미정확도가 아니다. NS를 true-token preservation으로 부르지 않는다.

## 신규1000, 과거5000, 현재B60 분리

### suffix

| policy | metric | numerator | prompt_denominator | rate | new_strict_numerator | new_token_accuracy | two_P_request_strict_n | two_P_request_strict_d |
|---|---|---|---|---|---|---|---|---|
| N4 | RS | 1000 | 1000 | 1.0 | 993 | 0.9931102362204725 |  |  |
| N4 | PS | 1938 | 2000 | 0.969 | 1423 | 0.7155511811023622 | 569 | 1000 |
| N4 | NS | 7108 | 10000 | 0.7108 | 444 | 0.0593503937007874 |  |  |
| REFIT4 | RS | 999 | 1000 | 0.999 | 998 | 0.9980314960629921 |  |  |
| REFIT4 | PS | 1950 | 2000 | 0.975 | 1405 | 0.7062007874015748 | 542 | 1000 |
| REFIT4 | NS | 7175 | 10000 | 0.7175 | 452 | 0.05984251968503937 |  |  |
| FROZEN2 | RS | 1000 | 1000 | 1.0 | 997 | 0.9970472440944882 |  |  |
| FROZEN2 | PS | 1958 | 2000 | 0.979 | 1467 | 0.7376968503937008 | 588 | 1000 |
| FROZEN2 | NS | 7070 | 10000 | 0.707 | 456 | 0.06053149606299212 |  |  |
| I2 | RS | 1000 | 1000 | 1.0 | 997 | 0.9970472440944882 |  |  |
| I2 | PS | 1961 | 2000 | 0.9805 | 1456 | 0.7322834645669292 | 578 | 1000 |
| I2 | NS | 7086 | 10000 | 0.7086 | 450 | 0.05984251968503937 |  |  |
| FROZEN4 | RS | 999 | 1000 | 0.999 | 997 | 0.9970472440944882 |  |  |
| FROZEN4 | PS | 1969 | 2000 | 0.9845 | 1487 | 0.7470472440944882 | 599 | 1000 |
| FROZEN4 | NS | 7056 | 10000 | 0.7056 | 461 | 0.060728346456692915 |  |  |
| I4 | RS | 999 | 1000 | 0.999 | 998 | 0.9980314960629921 |  |  |
| I4 | PS | 1966 | 2000 | 0.983 | 1489 | 0.7480314960629921 | 594 | 1000 |
| I4 | NS | 7056 | 10000 | 0.7056 | 457 | 0.06033464566929134 |  |  |

### entry_old

| policy | metric | numerator | prompt_denominator | rate | new_strict_numerator | new_token_accuracy | two_P_request_strict_n | two_P_request_strict_d |
|---|---|---|---|---|---|---|---|---|
| N4 | RS | 4984 | 5000 | 0.9968 | 4867 | 0.9737620832511343 |  |  |
| N4 | PS | 9658 | 10000 | 0.9658 | 6863 | 0.690274215821661 | 2675 | 5000 |
| N4 | NS | 34513 | 50000 | 0.69026 | 2323 | 0.058216610771355294 |  |  |
| REFIT4 | RS | 4980 | 5000 | 0.996 | 4861 | 0.9725784178338923 |  |  |
| REFIT4 | PS | 9649 | 10000 | 0.9649 | 6884 | 0.6923456303018347 | 2673 | 5000 |
| REFIT4 | NS | 34726 | 50000 | 0.69452 | 2316 | 0.05819688301440126 |  |  |
| FROZEN2 | RS | 4981 | 5000 | 0.9962 | 4858 | 0.9719865851252713 |  |  |
| FROZEN2 | PS | 9649 | 10000 | 0.9649 | 6822 | 0.686230025646084 | 2652 | 5000 |
| FROZEN2 | NS | 34396 | 50000 | 0.68792 | 2312 | 0.05803906095876899 |  |  |
| I2 | RS | 4981 | 5000 | 0.9962 | 4861 | 0.9725784178338923 |  |  |
| I2 | PS | 9652 | 10000 | 0.9652 | 6827 | 0.6867232195699349 | 2656 | 5000 |
| I2 | NS | 34509 | 50000 | 0.69018 | 2287 | 0.05746695600710199 |  |  |
| FROZEN4 | RS | 4978 | 5000 | 0.9956 | 4850 | 0.9704083645689485 |  |  |
| FROZEN4 | PS | 9644 | 10000 | 0.9644 | 6807 | 0.6846518050897613 | 2634 | 5000 |
| FROZEN4 | NS | 34375 | 50000 | 0.6875 | 2308 | 0.057940422173998814 |  |  |
| I4 | RS | 4976 | 5000 | 0.9952 | 4845 | 0.9694219767212467 |  |  |
| I4 | PS | 9646 | 10000 | 0.9646 | 6796 | 0.6836654172420595 | 2629 | 5000 |
| I4 | NS | 34417 | 50000 | 0.68834 | 2323 | 0.05819688301440126 |  |  |

### current

| policy | metric | numerator | prompt_denominator | rate | new_strict_numerator | new_token_accuracy | two_P_request_strict_n | two_P_request_strict_d |
|---|---|---|---|---|---|---|---|---|
| N4 | RS | 100 | 100 | 1.0 | 100 | 1.0 |  |  |
| N4 | PS | 195 | 200 | 0.975 | 155 | 0.775 | 68 | 100 |
| N4 | NS | 734 | 1000 | 0.734 | 50 | 0.05 |  |  |
| REFIT4 | RS | 100 | 100 | 1.0 | 100 | 1.0 |  |  |
| REFIT4 | PS | 200 | 200 | 1.0 | 153 | 0.765 | 65 | 100 |
| REFIT4 | NS | 738 | 1000 | 0.738 | 49 | 0.049 |  |  |
| FROZEN2 | RS | 100 | 100 | 1.0 | 100 | 1.0 |  |  |
| FROZEN2 | PS | 199 | 200 | 0.995 | 158 | 0.79 | 68 | 100 |
| FROZEN2 | NS | 718 | 1000 | 0.718 | 53 | 0.053 |  |  |
| I2 | RS | 100 | 100 | 1.0 | 100 | 1.0 |  |  |
| I2 | PS | 198 | 200 | 0.99 | 156 | 0.78 | 66 | 100 |
| I2 | NS | 724 | 1000 | 0.724 | 50 | 0.05 |  |  |
| FROZEN4 | RS | 100 | 100 | 1.0 | 100 | 1.0 |  |  |
| FROZEN4 | PS | 198 | 200 | 0.99 | 158 | 0.79 | 68 | 100 |
| FROZEN4 | NS | 720 | 1000 | 0.72 | 52 | 0.052 |  |  |
| I4 | RS | 100 | 100 | 1.0 | 100 | 1.0 |  |  |
| I4 | PS | 198 | 200 | 0.99 | 157 | 0.785 | 66 | 100 |
| I4 | NS | 727 | 1000 | 0.727 | 51 | 0.051 |  |  |

### first_suffix500

| policy | metric | numerator | prompt_denominator | rate | new_strict_numerator | new_token_accuracy | two_P_request_strict_n | two_P_request_strict_d |
|---|---|---|---|---|---|---|---|---|
| N4 | RS | 500 | 500 | 1.0 | 496 | 0.9921104536489151 |  |  |
| N4 | PS | 970 | 1000 | 0.97 | 703 | 0.7061143984220908 | 279 | 500 |
| N4 | NS | 3578 | 5000 | 0.7156 | 225 | 0.05818540433925049 |  |  |
| REFIT4 | RS | 499 | 500 | 0.998 | 498 | 0.9960552268244576 |  |  |
| REFIT4 | PS | 978 | 1000 | 0.978 | 704 | 0.7071005917159763 | 272 | 500 |
| REFIT4 | NS | 3626 | 5000 | 0.7252 | 226 | 0.05818540433925049 |  |  |
| FROZEN2 | RS | 500 | 500 | 1.0 | 498 | 0.9960552268244576 |  |  |
| FROZEN2 | PS | 979 | 1000 | 0.979 | 728 | 0.7317554240631163 | 290 | 500 |
| FROZEN2 | NS | 3566 | 5000 | 0.7132 | 227 | 0.058579881656804736 |  |  |
| I2 | RS | 500 | 500 | 1.0 | 497 | 0.9940828402366864 |  |  |
| I2 | PS | 979 | 1000 | 0.979 | 729 | 0.7327416173570019 | 292 | 500 |
| I2 | NS | 3576 | 5000 | 0.7152 | 227 | 0.058579881656804736 |  |  |
| FROZEN4 | RS | 499 | 500 | 0.998 | 497 | 0.9940828402366864 |  |  |
| FROZEN4 | PS | 983 | 1000 | 0.983 | 735 | 0.73767258382643 | 296 | 500 |
| FROZEN4 | NS | 3556 | 5000 | 0.7112 | 245 | 0.06173570019723866 |  |  |
| I4 | RS | 499 | 500 | 0.998 | 498 | 0.9960552268244576 |  |  |
| I4 | PS | 982 | 1000 | 0.982 | 735 | 0.73767258382643 | 290 | 500 |
| I4 | NS | 3549 | 5000 | 0.7098 | 236 | 0.05996055226824457 |  |  |

Fullseen의 Current/suffix/old/Historical/first500 rows는 같은 endpoint identity에서 재사용했다. 별도 평가나 추가 독립 분모로 더하지 않는다. Online-at-write1000은 서로 다른10개state의 합이며 finalretention이 아니다.

## 누적 retention·전이

| before | metric | before_numerator | after_numerator | lost | gained | retained | failed_both | conditional_loss_rate | prompt_denominator |
|---|---|---|---|---|---|---|---|---|---|
| N4 | RS | 1000 | 1000 | 0 | 0 | 1000 | 0 | 0.0 | 1000 |
| N4 | PS | 1941 | 1938 | 8 | 5 | 1933 | 54 | 0.004121586810922205 | 2000 |
| N4 | NS | 7164 | 7108 | 249 | 193 | 6915 | 2643 | 0.0347571189279732 | 10000 |
| REFIT4 | RS | 1000 | 999 | 1 | 0 | 999 | 0 | 0.001 | 1000 |
| REFIT4 | PS | 1952 | 1950 | 5 | 3 | 1947 | 45 | 0.0025614754098360654 | 2000 |
| REFIT4 | NS | 7192 | 7175 | 189 | 172 | 7003 | 2636 | 0.02627919911012236 | 10000 |
| FROZEN2 | RS | 1000 | 1000 | 0 | 0 | 1000 | 0 | 0.0 | 1000 |
| FROZEN2 | PS | 1960 | 1958 | 5 | 3 | 1955 | 37 | 0.002551020408163265 | 2000 |
| FROZEN2 | NS | 7125 | 7070 | 266 | 211 | 6859 | 2664 | 0.037333333333333336 | 10000 |
| I2 | RS | 1000 | 1000 | 0 | 0 | 1000 | 0 | 0.0 | 1000 |
| I2 | PS | 1962 | 1961 | 4 | 3 | 1958 | 35 | 0.0020387359836901123 | 2000 |
| I2 | NS | 7126 | 7086 | 264 | 224 | 6862 | 2650 | 0.03704743193937693 | 10000 |
| FROZEN4 | RS | 1000 | 999 | 1 | 0 | 999 | 0 | 0.001 | 1000 |
| FROZEN4 | PS | 1969 | 1969 | 6 | 6 | 1963 | 25 | 0.0030472320975114273 | 2000 |
| FROZEN4 | NS | 7104 | 7056 | 286 | 238 | 6818 | 2658 | 0.04025900900900901 | 10000 |
| I4 | RS | 1000 | 999 | 1 | 0 | 999 | 0 | 0.001 | 1000 |
| I4 | PS | 1968 | 1966 | 8 | 6 | 1960 | 26 | 0.0040650406504065045 | 2000 |
| I4 | NS | 7136 | 7056 | 306 | 226 | 6830 | 2638 | 0.04288116591928251 | 10000 |

first-suffix500의 W55→W60은 같은 문항의 시간 전이다. 공통성공/공통실패 strata는 W55의 사후 상태에 조건을 건 것이므로 인과 식별이 아니다. B60 cohort는 후속 노출0으로 별도 표기했다. oldentry 평가가 없는 모집단의 교차정책 W60 차이를 W50→W60 forgetting으로 부르지 않는다.

ACTIVE_TARGET는 observed prefix의 exact(subject,relation) 최신target문자열과 같은 이벤트이며 같은target 재발행을 포함한다. SUPERSEDED/UNKNOWN을 버리지 않는다. 의미상 충돌의 완전 판별은 아니다. 저장되지 않은 중간 첫failure 시점은 추정하지 않는다.

## NLL·strict·반대 방향 근거

nll-distributions.csv는 new/true/desiredmargin의 mean/median/p90/p95/p99/min/max를 prompt와 request-cluster mean으로 분리한다. paired-transitions.csv의 new/true NLL harm tail은 동일 identity 교차정책 차이다. 성공률 증가를 모든 문항의 손실0 또는 strict 개선으로 바꾸지 않는다. P2/N10을 독립 request로 늘린 유의성 주장은 없다. 이번 표는 단일 고정 순서이며 별도 random-seed 반복 또는 full10k 안정성을 입증하지 않는다.

request-cluster-uncertainty.csv/JSON은 동일 case의 P2/N10을 함께 resample한 paired request-cluster percentile 95% CI다. NumPy PCG64 seed20260914, 1000회이며 저장된 표본 내 불확실성만 나타낸다. 독립 편집순서 반복이나 checkpoint 독립성을 가정한 결과가 아니고 다중 탐색에 대한 확증적 유의성 선언도 아니다. CI가0을포함하는지여부는성능gate가아니다. 양의 preference Δpp는 gain, 양의 desired-target NLL Δ는 harm이다.

## 작은 general 패널

| policy | wiki_nll | wiki_tokens | wiki_sequences | mmlu_correct | mmlu_d | mmlu_invalid |
|---|---|---|---|---|---|---|
| N4 | 2.3415367626585066 | 24999 | 128 | 18 | 32 | 0 |
| REFIT4 | 2.340970307122916 | 24999 | 128 | 19 | 32 | 0 |
| FROZEN2 | 2.354561456479132 | 24999 | 128 | 18 | 32 | 0 |
| I2 | 2.3523292653262615 | 24999 | 128 | 17 | 32 | 0 |
| FROZEN4 | 2.3639503247104585 | 24999 | 128 | 17 | 32 | 0 |
| I4 | 2.360537697095424 | 24999 | 128 | 17 | 32 | 0 |

Wiki128 mask/입력과 MMLUdev32 identity를 재사용했다. MMLU는 alternative integercorrect/invalid이고 generation parser/F1/full57subject 점수와 다르다. Audit128/MMLU68/FutureN은 DEFERRED_NOT_EVALUATED, 신규평가·정책feedback0.

## 실행·상태·기술 검증

첫 기술 array46990 native/I1 비교는 target maxabs1.273453e-4, stored weight3.423542e-6로 exact 기준에 실패하여 보존했다. 원인은 zero-offset 공유 AddBackward와 native leaf의 FP32 gradient 합산 경로 차이로 좁혀졌다. r2는 a0와aj가 torch.equal일 때 동일 leafu를 직접 사용하고 nonzero offset은 원계약 u+(a0-aj)를 유지한다. 수정 I1 47014는 보존 native B100과 target/loss/weight/history/.75partial/context/RNG 모두 exact이며 fixed-W pause/resume u/m/v/t도 exact였다. 원native/optimizerbudget/teacher/clamp/tolerance 변경0. 이 기술 비교는 정책 품질 PASS가 아니다. 기술 비용1418GPU-sec(실패942포함)는 본신규40batch 비용과 분리한다.

state-review-receipt.json 및 state-links/subwrites/checkpoints/target-counters CSV에 실제검산을 기록한다. innerhistoryappend0, native endpoint finalizer append1/batch; request100 target barrier 뒤 batchwrite1이다. u/Adam m/v/t는 request내chunk사이에만 유지하며 anchor/teacher/clamp는batchentry고정이다. Frozen후속chunk도 현재Y로residual을새로읽는다. oldreference optimizer내부미저장항목은 NOT_RECORDED이다.

I2/I4의 aj는 각 target chunk에서 현재 unhooked clean activation으로 측정된다. Frozen 후속 chunk는 target forward를 실행하지 않으므로 저장된 aj/teacher/u/Adam은 최초 target snapshot의 재사용이지 현재 clean-aj 재측정이 아니다. Frozen의 현재 canonical writer Y와 residual은 각 subwrite에 별도로 실제 기록됐다. 이 두 readout을 같은 관측값으로 바꾸어 부르지 않는다.

CPU weights_only/tensorSHA와 실제 GPU continuation/replay는 다르다. 신규 CP51/55/60은 selectedW4/M4/context/RNG를 보존하지만 전체모델checkpoint가 아니다. Base model/P/config/source/order closure가필요하다. Actualdelta journal의 exacttrajectory replay는 NOT_TESTED다. 원native및NativeSingletonFitter.fit 수정0.

## 비용과 저장

| policy | new_spending | online_seconds | evaluation_seconds | target_including_nested_IO_seconds | native_writer_seconds | history_seconds | materialization_seconds | snapshot_seconds |
|---|---|---|---|---|---|---|---|---|
| N4 | False | 3048.9109 | 1667.1237 | NOT_RECORDED | NOT_RECORDED | 133.0029 | 6.9196 | NOT_RECORDED |
| REFIT4 | False | 3727.5258 | 1694.2563 | NOT_RECORDED | NOT_RECORDED | 147.9162 | 8.8727 | NOT_RECORDED |
| FROZEN2 | True | 3254.3336 | 1376.2037 | 2807.246 | 293.7156 | 137.794 | 15.578 | 12.5381 |
| I2 | True | 3480.9635 | 1384.3906 | 2891.0494 | 381.166 | 186.4481 | 22.3 | 16.9453 |
| FROZEN4 | True | 3636.1753 | 1378.8845 | 2813.6348 | 638.3415 | 147.6279 | 36.5712 | 15.6843 |
| I4 | True | 3891.8415 | 1395.5275 | 3061.1968 | 637.9249 | 158.2248 | 34.495 | 14.124 |

compute-ledger.csv는 reused reference와 신규spending을 분리한다. 신규targettimer의 requeststate I/O는 nested이므로 총합에 다시더하지 않는다. 기존reference는 매batchgeneral, 신규는terminalgeneral이므로 평가포함wall을 같은정책online비용으로 치환하지 않는다. Instrumentedonline은purewriter가 아니다. Prepared/M8과기존N4/REFIT4의과거비용을새연구비로중복계상하지않는다. Scheduler allocation/queue/concurrency는별도receipt근거이며 측정되지않은component는NOT_RECORDED.

## 재현·산출물·제한

원시tensor/request/teacher/prompt/fullstdout은 local-only이고 Git에 포함하지 않는다. NO_BROADCAST_NOT_REQUIRED. 원본checkpoint/공유asset/다른run삭제0. Main에는 본namespace CPU분석코드와raw-free표/PNG/manifest만통합한다.

재현 명령(저장된 terminal 자료만 CPU):

```bash
python -m project.run_scripts.low_cost_write_donor_pilot.refresh_review --attempt /data/janghj/ODE-edit/local/refit4-write-refresh-seq1000/20260914-v1/attempt-r2 --out experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1
python -m project.run_scripts.low_cost_write_donor_pilot.refresh_state_review --attempt /data/janghj/ODE-edit/local/refit4-write-refresh-seq1000/20260914-v1/attempt-r2 --out experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1
python -m project.run_scripts.low_cost_write_donor_pilot.refresh_report --attempt /data/janghj/ODE-edit/local/refit4-write-refresh-seq1000/20260914-v1/attempt-r2 --out experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1
```

PNG는 이 코드가 CSV를 읽어 생성한다: final-six-policy.png, current-batch-curves.png, fixed-first500-retention.png, desired-target-nll-tails.png, suffix-preference-contrasts.png, cohort-retention-transitions.png, layer-update-magnitude.png, recorded-compute-cost.png. AI 이미지생성/수동그림수정0. 각PNG SHA는manifest와plot-reproduction.json에결속한다.

Middle 결과 후 최대1후보/Late policy는GH가결정한다. 본보고서가후보선정/새Late제출/full10k/추가alpha·layer실험을승인하지않는다. finitepoor결과를제외하거나성능ANDgate로완료를판정하지않는다.


## 완료 recall 후 추가 검산과 상세 대조

이번 지시 ODEEDIT-S06-REFIT4-WRITE-REFRESH-SEQ1000-COMPLETED-REVIEW-SH4-V1는 완료 검증/CPU 분석만 재개했다. 네 신규 scheduler COMPLETED0와 scientific terminal40batch/기존reference20batch를 별도로 확인했다. 이전 partial-final-table-v2 및 pause receipt는 불변이며 아래 완료표로 덮어쓰지 않았다.

### 주요 대조: after−before (동일 문항, 다른 누적 경로)

| before | after | population | metric | d | delta | delta_pp | lost | gained | strict_delta | desired_NLL_mean | desired_NLL_p95 | desired_NLL_p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FROZEN2 | REFIT4 | suffix | RS | 1000 | -1 | -0.1 | 1 | 0 | 1 | 0.014582953790282772 | 0.044790423824451836 | 0.10826680661644784 |
| FROZEN2 | REFIT4 | suffix | PS | 2000 | -8 | -0.4 | 23 | 15 | -62 | 0.12906479982487146 | 1.7944646120071404 | 3.5299408566951747 |
| FROZEN2 | REFIT4 | suffix | NS | 10000 | 105 | 1.05 | 182 | 287 | -4 | -0.07041459574532928 | 1.0935574293136592 | 2.2156152200698878 |
| FROZEN2 | REFIT4 | entry_old | RS | 5000 | -1 | -0.02 | 7 | 6 | 3 | 0.008890529235888244 | 0.01068838261999195 | 0.39388629555702415 |
| FROZEN2 | REFIT4 | entry_old | PS | 10000 | 0 | 0.0 | 49 | 49 | 62 | -0.009132141694460734 | 0.5360600531101215 | 1.3479792779684083 |
| FROZEN2 | REFIT4 | entry_old | NS | 50000 | 330 | 0.66 | 1131 | 1461 | 4 | -0.07339343392210547 | 1.1155468225479097 | 2.0677349829673783 |
| FROZEN2 | REFIT4 | fullseen | RS | 6000 | -2 | -0.03333333333333333 | 8 | 6 | 4 | 0.009839266661620666 | 0.026850369479507228 | 0.3584289240837101 |
| FROZEN2 | REFIT4 | fullseen | PS | 12000 | -8 | -0.06666666666666667 | 72 | 64 | 0 | 0.013900681892094629 | 0.7312289595603929 | 2.104018493294719 |
| FROZEN2 | REFIT4 | fullseen | NS | 60000 | 435 | 0.725 | 1313 | 1748 | 0 | -0.07289696089264278 | 1.1146729111671447 | 2.0861482810974175 |
| FROZEN2 | I2 | suffix | RS | 1000 | 0 | 0.0 | 0 | 0 | 0 | 0.0004615485692938819 | 0.0020161122083663923 | 0.013497504480183093 |
| FROZEN2 | I2 | suffix | PS | 2000 | 3 | 0.15 | 5 | 8 | -11 | -0.015635209700772975 | 0.5632550477981566 | 1.372012205123901 |
| FROZEN2 | I2 | suffix | NS | 10000 | 16 | 0.16 | 96 | 112 | -6 | -0.024819039438583424 | 0.49627996683120634 | 0.934579210281372 |
| FROZEN2 | I2 | entry_old | RS | 5000 | 0 | 0.0 | 1 | 1 | 3 | -0.001285602050946909 | 0.008717234805226346 | 0.1620483735203763 |
| FROZEN2 | I2 | entry_old | PS | 10000 | 3 | 0.03 | 21 | 24 | 5 | 0.000115366666270711 | 0.2585818827152251 | 0.5690238428115845 |
| FROZEN2 | I2 | entry_old | NS | 50000 | 113 | 0.226 | 486 | 599 | -25 | -0.013176223620728124 | 0.5254875540733333 | 0.946247577667238 |
| FROZEN2 | I2 | fullseen | RS | 6000 | 0 | 0.0 | 1 | 1 | 3 | -0.0009944102809067773 | 0.007008344773203144 | 0.12533738732337993 |
| FROZEN2 | I2 | fullseen | PS | 12000 | 6 | 0.05 | 26 | 32 | -6 | -0.002509729394903237 | 0.2951487064361571 | 0.7420050555467611 |
| FROZEN2 | I2 | fullseen | NS | 60000 | 129 | 0.215 | 582 | 711 | -31 | -0.015116692923704007 | 0.5209503650665277 | 0.9430223751068134 |
| FROZEN4 | I4 | suffix | RS | 1000 | 0 | 0.0 | 0 | 0 | 1 | 0.00018932499006587024 | 0.0018914491345640252 | 0.00793710027122869 |
| FROZEN4 | I4 | suffix | PS | 2000 | -3 | -0.15 | 7 | 4 | 2 | -0.017324626264908147 | 0.7228709287941456 | 1.8555274033546445 |
| FROZEN4 | I4 | suffix | NS | 10000 | 0 | 0.0 | 140 | 140 | -4 | -0.01056010238721501 | 0.7042887210845914 | 1.2090623378753673 |
| FROZEN4 | I4 | entry_old | RS | 5000 | -2 | -0.04 | 3 | 1 | -5 | 0.0040730615026805025 | 0.01998351905494931 | 0.2436111430078747 |
| FROZEN4 | I4 | entry_old | PS | 10000 | 2 | 0.02 | 27 | 29 | -11 | 0.0032899621220458357 | 0.359537565708159 | 0.7873749876022359 |
| FROZEN4 | I4 | entry_old | NS | 50000 | 42 | 0.084 | 727 | 769 | 15 | -0.007314597373588476 | 0.6913759469985954 | 1.2476867151260396 |
| FROZEN4 | I4 | fullseen | RS | 6000 | -2 | -0.03333333333333333 | 3 | 1 | -4 | 0.003425772083911397 | 0.014628867246210632 | 0.22883974194526688 |
| FROZEN4 | I4 | fullseen | PS | 12000 | -1 | -0.008333333333333333 | 34 | 33 | -9 | -0.00014580260911316146 | 0.40108689069747877 | 0.9962893486022951 |
| FROZEN4 | I4 | fullseen | NS | 60000 | 42 | 0.07 | 867 | 909 | 11 | -0.007855514875859565 | 0.6935866117477417 | 1.2368481349945084 |
| I2 | I4 | suffix | RS | 1000 | -1 | -0.1 | 1 | 0 | 1 | -0.00031134777112856683 | 0.001178918458754195 | 0.005587305364315379 |
| I2 | I4 | suffix | PS | 2000 | 5 | 0.25 | 7 | 12 | 33 | -0.038265289497066986 | 0.7582576833665368 | 2.112929302901029 |
| I2 | I4 | suffix | NS | 10000 | -30 | -0.3 | 156 | 126 | 7 | 0.039880955999705474 | 0.8558589220046992 | 1.5125069618225098 |
| I2 | I4 | entry_old | RS | 5000 | -5 | -0.1 | 6 | 1 | -16 | 0.02355895264849132 | 0.043655676022171984 | 0.6628893184661903 |
| I2 | I4 | entry_old | PS | 10000 | -6 | -0.06 | 32 | 26 | -31 | 0.02748393749966417 | 0.43851992189884137 | 1.1245893907547022 |
| I2 | I4 | entry_old | NS | 50000 | -92 | -0.184 | 773 | 681 | 36 | 0.03389015451954678 | 0.8126833915710449 | 1.4348607444763184 |
| I2 | I4 | fullseen | RS | 6000 | -6 | -0.1 | 7 | 1 | -15 | 0.019580569245221342 | 0.02806194648146635 | 0.5409270470589415 |
| I2 | I4 | fullseen | PS | 12000 | -1 | -0.008333333333333333 | 39 | 38 | 2 | 0.01652573300020898 | 0.4792787820100784 | 1.371128840446474 |
| I2 | I4 | fullseen | NS | 60000 | -122 | -0.20333333333333334 | 929 | 807 | 43 | 0.03488862143290656 | 0.8181158304214455 | 1.4512753963470506 |

양의 desired NLL 차이는 정답 target 악화, 양의 성공 수 차이는 선호 개선이다. NS의 desired target은 true이며 표의 strict_delta는 new strict로 별도 정의한다. NS 총점 증가만으로 true NLL가 개선됐다고 결론내리지 않는다. competing_NLL_mean 및 양쪽 NLL는 contrast-summary/terminal-contrasts에 함께 제공한다.

### 과거 유효 target과 신규 target

| policy | population | metric | numerator | prompt_denominator | rate | new_strict_numerator |
|---|---|---|---|---|---|---|
| N4 | suffix | RS | 994 | 994 | 1.0 | 989 |
| N4 | suffix | PS | 1932 | 1988 | 0.971830985915493 | 1422 |
| N4 | suffix | NS | 7059 | 9940 | 0.7101609657947686 | 443 |
| N4 | entry_old | RS | 4901 | 4910 | 0.9981670061099797 | 4833 |
| N4 | entry_old | PS | 9529 | 9820 | 0.970366598778004 | 6849 |
| N4 | entry_old | NS | 33939 | 49100 | 0.6912219959266802 | 2305 |
| REFIT4 | suffix | RS | 994 | 994 | 1.0 | 993 |
| REFIT4 | suffix | PS | 1943 | 1988 | 0.977364185110664 | 1404 |
| REFIT4 | suffix | NS | 7127 | 9940 | 0.7170020120724346 | 451 |
| REFIT4 | entry_old | RS | 4900 | 4910 | 0.9979633401221996 | 4830 |
| REFIT4 | entry_old | PS | 9523 | 9820 | 0.9697556008146639 | 6862 |
| REFIT4 | entry_old | NS | 34150 | 49100 | 0.6955193482688391 | 2298 |
| FROZEN2 | suffix | RS | 994 | 994 | 1.0 | 993 |
| FROZEN2 | suffix | PS | 1950 | 1988 | 0.9808853118712274 | 1466 |
| FROZEN2 | suffix | NS | 7020 | 9940 | 0.7062374245472837 | 455 |
| FROZEN2 | entry_old | RS | 4900 | 4910 | 0.9979633401221996 | 4828 |
| FROZEN2 | entry_old | PS | 9520 | 9820 | 0.9694501018329938 | 6811 |
| FROZEN2 | entry_old | NS | 33821 | 49100 | 0.6888187372708757 | 2296 |
| I2 | suffix | RS | 994 | 994 | 1.0 | 993 |
| I2 | suffix | PS | 1953 | 1988 | 0.9823943661971831 | 1455 |
| I2 | suffix | NS | 7037 | 9940 | 0.7079476861167002 | 449 |
| I2 | entry_old | RS | 4900 | 4910 | 0.9979633401221996 | 4830 |
| I2 | entry_old | PS | 9520 | 9820 | 0.9694501018329938 | 6814 |
| I2 | entry_old | NS | 33933 | 49100 | 0.6910997963340122 | 2269 |
| FROZEN4 | suffix | RS | 994 | 994 | 1.0 | 993 |
| FROZEN4 | suffix | PS | 1962 | 1988 | 0.9869215291750503 | 1486 |
| FROZEN4 | suffix | NS | 7008 | 9940 | 0.7050301810865192 | 460 |
| FROZEN4 | entry_old | RS | 4897 | 4910 | 0.9973523421588595 | 4822 |
| FROZEN4 | entry_old | PS | 9517 | 9820 | 0.9691446028513239 | 6789 |
| FROZEN4 | entry_old | NS | 33798 | 49100 | 0.6883503054989817 | 2287 |
| I4 | suffix | RS | 994 | 994 | 1.0 | 994 |
| I4 | suffix | PS | 1959 | 1988 | 0.9854124748490946 | 1488 |
| I4 | suffix | NS | 7007 | 9940 | 0.7049295774647887 | 456 |
| I4 | entry_old | RS | 4896 | 4910 | 0.9971486761710794 | 4816 |
| I4 | entry_old | PS | 9516 | 9820 | 0.9690427698574338 | 6780 |
| I4 | entry_old | NS | 33838 | 49100 | 0.6891649694501019 | 2300 |

### 요청당 optimization과 실측 비용

| policy | reused | Adam | loss | recorded_request_chunks | zero_Adam_chunks | frozen_reuse_chunks | online_seconds | online_ratio_N4 | evaluation_seconds | allocated_GPU_seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| N4 | True | REFERENCE_LOG_RECONSTRUCTION_24000 | REFERENCE_LOG_RECONSTRUCTION_25000 | 0 | NOT_RECORDED_STRUCTURED | NOT_APPLICABLE | 3048.9109355434775 | 1.0 | 1667.1237003495917 | 5237 |
| REFIT4 | True | REFERENCE_LOG_RECONSTRUCTION_27525 | REFERENCE_LOG_RECONSTRUCTION_29525 | 0 | NOT_RECORDED_STRUCTURED | NOT_APPLICABLE | 3727.52581872046 | 1.2225761583475108 | 1694.2562718959525 | 6027 |
| FROZEN2 | False | 24000 | 25000 | 2000 | 1000 | 1000 | 3254.333564837463 | 1.0673757396122063 | 1376.203738739714 | 5178 |
| I2 | False | 24000 | 26000 | 2000 | 0 | 0 | 3480.963488657959 | 1.1417071742167721 | 1384.3905565161258 | 5444 |
| FROZEN4 | False | 24000 | 25000 | 4000 | 3000 | 3000 | 3636.1753174467012 | 1.1926144758959782 | 1378.8844836298376 | 5813 |
| I4 | False | 24000 | 28000 | 4000 | 0 | 0 | 3891.8415133673698 | 1.2764694002691939 | 1395.5275497669354 | 6065 |

| policy | program_wall_seconds | load_seconds | peak_GPU_allocated_bytes | peak_GPU_reserved_bytes | host_batch_MaxRSS |
|---|---|---|---|---|---|
| N4 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| REFIT4 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| FROZEN2 | 5112.5896340748295 | 13.131150630302727 | 35770822656 | 39298531328 | 33519920K |
| I2 | 5382.144452076405 | 16.31167613901198 | 35770822656 | 39254491136 | 34114100K |
| FROZEN4 | 5751.6439860844985 | 14.62578375916928 | 35770822656 | 39298531328 | 38599892K |
| I4 | 6007.421680117957 | 14.476625954732299 | 35770822656 | 39252393984 | 33522500K |

신규 main 22500 GPU-sec = 6.250000 GPUh; 기술 실패942+수정검증476=1418 GPU-sec. 이번 신규 합계 23918 GPU-sec = 6.643889 GPUh. 재사용 N4/REFIT4 할당11264 GPU-sec는 새 비용0으로 처리한다.

동시 allocation 최대 2 GPU, 2GPU 겹친 구간 10701.0초. 이는 GPU utilization 실측이 아니다. 새 raw rehash bytes=81633255100이며 referenced input이 포함된 검산량과 output의 실제 disk allocation은 동일하지 않다.

Primary+supplement full SHA/size 12432 files / 81633386526 bytes. 신규 output 12388개 전부 검사, 미참조 미검사0. Supplemental restore4개도 확인했다. 비L4 불변은 frozen runtime의 full nonselected guard와 성공 receipt 결속이며 독립적인 전체 모델 tensor 재감사/새 GPU replay가 아니다.

### Target refresh와 실제 write 관측

| policy | chunk | requests | zero_step | early_stop | u_chunk_displacement_norm_mean | Z_chunk_displacement_norm_mean | stored_aj_minus_a0_norm_mean | current_residual_norm_mean | anchor_scope |
|---|---|---|---|---|---|---|---|---|---|
| FROZEN2 | 0 | 1000 | 0 | 0 | 2.8605515977082647 | 2.860551597594055 | 0.0 | 2.860551616456601 | CURRENT_CHUNK_UNHOOKED_ANCHOR |
| FROZEN2 | 1 | 1000 | 1000 | 0 | 0.0 | 0.0 | 0.0 | 1.1749553369698376 | STALE_FROZEN_SNAPSHOT_NOT_CURRENT_READOUT |
| I2 | 0 | 1000 | 0 | 0 | 2.8604611988669353 | 2.8604611988623945 | 0.0 | 2.860461217962999 | CURRENT_CHUNK_UNHOOKED_ANCHOR |
| I2 | 1 | 1000 | 0 | 0 | 0.357246504080954 | 0.3572465041043548 | 1.688033945852352 | 1.212296733182616 | CURRENT_CHUNK_UNHOOKED_ANCHOR |
| FROZEN4 | 0 | 1000 | 0 | 0 | 2.86928415801466 | 2.8692841579416 | 0.0 | 2.8692841745769444 | CURRENT_CHUNK_UNHOOKED_ANCHOR |
| FROZEN4 | 1 | 1000 | 1000 | 0 | 0.0 | 0.0 | 0.0 | 1.178622556203287 | STALE_FROZEN_SNAPSHOT_NOT_CURRENT_READOUT |
| FROZEN4 | 2 | 1000 | 1000 | 0 | 0.0 | 0.0 | 0.0 | 0.49977638857017265 | STALE_FROZEN_SNAPSHOT_NOT_CURRENT_READOUT |
| FROZEN4 | 3 | 1000 | 1000 | 0 | 0.0 | 0.0 | 0.0 | 0.22163997204363386 | STALE_FROZEN_SNAPSHOT_NOT_CURRENT_READOUT |
| I4 | 0 | 1000 | 0 | 0 | 2.869111001671531 | 2.8691110016504875 | 0.0 | 2.869111017442541 | CURRENT_CHUNK_UNHOOKED_ANCHOR |
| I4 | 1 | 1000 | 0 | 0 | 0.31320314844010344 | 0.31320314838468755 | 1.6930542175769951 | 1.2094100387905637 | CURRENT_CHUNK_UNHOOKED_ANCHOR |
| I4 | 2 | 1000 | 0 | 0 | 0.18792660110691262 | 0.18792660107542006 | 2.369191874647883 | 0.5604873495333547 | CURRENT_CHUNK_UNHOOKED_ANCHOR |
| I4 | 3 | 1000 | 0 | 0 | 0.19162014524875487 | 0.19162014527269516 | 2.647135725013779 | 0.35298423732540823 | CURRENT_CHUNK_UNHOOKED_ANCHOR |

| policy | path_length_sum_actual_delta_frobenius | summed_stored_delta_net_frobenius | checkpoint_W_minus_W50_frobenius | path_to_checkpoint_net_ratio | delta_sum_minus_checkpoint_displacement_frobenius |
|---|---|---|---|---|---|
| FROZEN2 | 127.1356080701192 | 40.09123115379434 | 40.091231153805985 | 3.1711574928286987 | 1.6026478324834825e-07 |
| I2 | 128.516855530139 | 40.04606291025553 | 40.04606291023358 | 3.2092257313339316 | 1.6123932771565546e-07 |
| FROZEN4 | 141.01244122973577 | 43.976193200393006 | 43.97619320043003 | 3.206563164461376 | 1.6620161703623113e-07 |
| I4 | 148.0497083494013 | 43.920343206424384 | 43.92034320646269 | 3.370868657684269 | 1.6512490311140006e-07 |

위 norm은 실제 저장 FP32 tensor의 Frobenius norm을 FP64로 집계한 값이며 native history/C_reg metric이 아니다. Subwrite path 합과 W60−W50 net은 다르다. FP64 delta 합과 checkpoint 차이를 제시해도 FP32 순차 replay 성공을 뜻하지 않는다. subwrites.csv의 다음 chunk currentY−이전Y는 직전 write의 관측 response이며, 마지막 subwrite 이후 별도 canonicalY가 없어 그 response는 NOT_MEASURED다. Canonical Kc 미저장으로 predicted DKc realization은 NOT_TESTED이며 새 forward로 채우지 않았다. Frozen 후속 aj는 최초 snapshot이지 새 clean readout이 아니다.

### 현재 비교가 구분하는 것과 구분하지 못하는 것

FROZEN2와 REFIT4는 같은 두 write/계수이나 second fresh target·teacher/clamp reset·추가 Adam 사용이 함께 다르다. REFIT4−FROZEN2의 관측을 budget 하나의 독립 효과라고 해석할 수 없다. I2−FROZEN2와 I4−FROZEN4는 write 수와 최대 Adam budget을 맞춘 정책 대조지만 실제 target 경로·loss 횟수·자기 W/M trajectory가 다르다. 단일 동일-state 원인 기여율은 식별되지 않았다.

전체6000 관측 I2−FROZEN2: RS+0/PS+6/NS+129; I4−FROZEN4: RS-2/PS-1/NS+42; I4−I2: RS-6/PS-1/NS-122. 전체 평균은 신규1000의 strict/retention과 과거5000의 변화를 대체하지 않으므로 위 모집단별 반대 방향과 tails를 함께 읽어야 한다. 방법의 최종 허용 claim과 후속 후보 선택은 하지 않았다.

I1/fixed-W exactPASS는 zero-offset native 연결과 고정W optimizer carry에 한정된다. changed-W/nonzero-offset I2/I4는 저장 상태와 수식/호출 순서를 감사한 것이며 별도 native 동등성 정답이 있는 검증이 아니다. 모델 backend 수치, 자기경로와 teacher/reset 차이를 현재 관측만으로 모두 분리할 수 없다. 0Adam/frozen재사용도 residual solve와 actual write가 있으므로 0write라고 표기하지 않는다.

### 원설계 산출물 coverage

| item | status | evidence |
|---|---|---|
| Final six policies/full6000 old+new | PASS | metric-reduction-receipt.json; final-populations.csv |
| Current/Historical all60; strict/token/two-P | PASS | batchmetrics.csv |
| W55/W60 same first500; at-write; B60 exposure0 | PASS | paired-transitions.csv |
| Active/superseded/unknown, both NLL tails | PASS | terminal-contrasts.csv; nll-distributions.csv |
| Request-cluster CI | PASS_DESCRIPTIVE_SINGLE_ORDER | request-cluster-uncertainty.json |
| New output fullhash/checkpoints/carry/history | SEE_STATE_RECEIPT | state-review-receipt.json |
| Native reference internal u/m/v/teacher | NOT_RECORDED | Reference source/CP audit reused; no invented moments |
| Intermediate actual tensor cross-hash bridge | PARTIAL | Stored CP/prepared bridged; other states dual ledgers only |
| GPU continuation / full nonzero-offset parity | NOT_TESTED | I1/fixedW exact gate does not prove all I2/I4 trajectories |
| Pure writer / target F-B separately | NOT_RECORDED_WHERE_NESTED | compute-ledger.csv; do not add nested IO twice |
| Quality/cost reference lines | DESCRIPTIVE_NO_AND_GATE | quality-frontier.csv; no policy pruning |
| Oldentry whole5000 temporal forgetting | NOT_MEASURED | Final cross-policy difference only |
| Audit128/MMLU68/FutureN/Late/selection | DEFERRED_NOT_EVALUATED | No new evaluation or policy selection |

추가 집계 재현 순서: 위 primary state reducer 뒤 `refresh_state_review --supplement-only`를 같은 --attempt/--out으로 실행하고, `python -m project.run_scripts.low_cost_write_donor_pilot.refresh_completed_report --attempt <attempt-r2> --out <publication> --control <completed-review-v1 local control>`로 추가 표와 본문을 생성한다. 마지막 refresh_report.seal()로 package manifest를 봉인한다. 모두 새 output namespace에서 실행한다. 원본 partial/report/raw는 변경하지 않는다. 원자료 누락을 새 evaluator 또는 인접 state로 보간하지 않았다.

# Native-response v3.1 B10 short-history mechanism pilot

상태: PRIMARY_FOUR_CELL_TECHNICAL_PASS. scientific_promotion=false. 본 자료는 단일 Official D10A(B10) warm history 뒤 독립 D10B(B10) 비교다. B9 재현, lifelong 결과 또는 locality 보장으로 해석하지 않는다. Server4 retention rerun과 별도 source/process/표본이며 진행 중인 다른 실험 결과는 사용하지 않았다.

## 지표와 분모

RS/PS는 각각 rewrite/rephrase에서 target-new NLL < target-true NLL인 prompt 수/전체 prompt 수다. NS는 반대로 neighborhood target-true NLL < target-new NLL이다. tie는 실패다. NLL은 token 평균이며 낮을수록 해당 continuation의 likelihood가 높다. strict teacher-forced coverage는 target token 전부 argmax와 일치하는 prompt 수다. old loss는 D10A의 warm-entry 성공 요청 중 D10B endpoint에서 실패한 요청만 센다. initially failed는 별도다.

CSV cell mapping: 0=Llama-MEMIT, 1=Llama-AlphaEdit, 2=Qwen-MEMIT, 3=Qwen-AlphaEdit. `PRE_EDIT_WARM`은 pristine cold W0가 아니라 Official D10A 한 batch를 적용한 공통 warm entry다.

| cell | arm | RS | PS | NS | old_RS | old_new_failure | V_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-MEMIT | PRE_EDIT_WARM | 1/10 (10.00%) | 4/20 (20.00%) | 90/100 (90.00%) | —→— | —/— | — |
| Llama-MEMIT | O_NATIVE | 9/10 (90.00%) | 17/20 (85.00%) | 90/100 (90.00%) | 10→10 | 0/10 | 0.1129406385880372 |
| Llama-MEMIT | ORBFH_HIST | 10/10 (100.00%) | 18/20 (90.00%) | 90/100 (90.00%) | 10→10 | 0/10 | 0.13455534377961262 |
| Llama-MEMIT | JV_NATIVE | 10/10 (100.00%) | 18/20 (90.00%) | 90/100 (90.00%) | 10→10 | 0/10 | 0.008644859078070584 |
| Llama-MEMIT | ORB_RAY_N | 10/10 (100.00%) | 18/20 (90.00%) | 90/100 (90.00%) | 10→10 | 0/10 | 0.05920142463147326 |
| Llama-AlphaEdit | PRE_EDIT_WARM | 1/10 (10.00%) | 4/20 (20.00%) | 90/100 (90.00%) | —→— | —/— | — |
| Llama-AlphaEdit | O_NATIVE | 10/10 (100.00%) | 18/20 (90.00%) | 86/100 (86.00%) | 10→10 | 0/10 | 0.0010871945557260078 |
| Llama-AlphaEdit | ORBFH_HIST | 10/10 (100.00%) | 18/20 (90.00%) | 86/100 (86.00%) | 10→10 | 0/10 | 0.045307662718103034 |
| Llama-AlphaEdit | JV_NATIVE | 10/10 (100.00%) | 20/20 (100.00%) | 85/100 (85.00%) | 10→10 | 0/10 | 0.004608298341466849 |
| Llama-AlphaEdit | ORB_RAY_N | 10/10 (100.00%) | 18/20 (90.00%) | 85/100 (85.00%) | 10→10 | 0/10 | 0.021263247980957516 |
| Qwen-MEMIT | PRE_EDIT_WARM | 1/10 (10.00%) | 4/20 (20.00%) | 89/100 (89.00%) | —→— | —/— | — |
| Qwen-MEMIT | O_NATIVE | 10/10 (100.00%) | 17/20 (85.00%) | 88/100 (88.00%) | 10→10 | 0/10 | 0.018687143931283 |
| Qwen-MEMIT | ORBFH_HIST | 10/10 (100.00%) | 18/20 (90.00%) | 88/100 (88.00%) | 10→10 | 0/10 | 0.02642050877884341 |
| Qwen-MEMIT | JV_NATIVE | 10/10 (100.00%) | 17/20 (85.00%) | 88/100 (88.00%) | 10→10 | 0/10 | 0.007006493670237749 |
| Qwen-MEMIT | ORB_RAY_N | 10/10 (100.00%) | 18/20 (90.00%) | 88/100 (88.00%) | 10→10 | 0/10 | 0.034250788721230496 |
| Qwen-AlphaEdit | PRE_EDIT_WARM | 1/10 (10.00%) | 4/20 (20.00%) | 89/100 (89.00%) | —→— | —/— | — |
| Qwen-AlphaEdit | O_NATIVE | 10/10 (100.00%) | 17/20 (85.00%) | 89/100 (89.00%) | 10→10 | 0/10 | 0.00010052582510455886 |
| Qwen-AlphaEdit | ORBFH_HIST | 10/10 (100.00%) | 17/20 (85.00%) | 88/100 (88.00%) | 10→10 | 0/10 | 0.01151337980747895 |
| Qwen-AlphaEdit | JV_NATIVE | 10/10 (100.00%) | 17/20 (85.00%) | 90/100 (90.00%) | 10→10 | 0/10 | 0.004370283359631236 |
| Qwen-AlphaEdit | ORB_RAY_N | 10/10 (100.00%) | 17/20 (85.00%) | 87/100 (87.00%) | 10→10 | 0/10 | 0.02039467567507225 |

### Rewrite NLL

| cell | arm | target | n | mean | median | p90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-MEMIT | PRE_EDIT_WARM | new | 10 | 10.487653577327729 | 10.473434448242188 | 15.93746395111084 | 18.268747329711914 |
| Llama-MEMIT | PRE_EDIT_WARM | true | 10 | 3.7672600939869882 | 2.6233330965042114 | 7.98519821166992 | 12.576455116271973 |
| Llama-MEMIT | O_NATIVE | new | 10 | 0.2828978858393384 | 0.022170167416334152 | 0.5057377189397805 | 2.2496743202209473 |
| Llama-MEMIT | O_NATIVE | true | 10 | 8.38160805106163 | 8.911693572998047 | 11.958709144592284 | 13.900571823120117 |
| Llama-MEMIT | ORBFH_HIST | new | 10 | 0.10024637755705043 | 0.022955799475312233 | 0.28148200511932364 | 0.5072330236434937 |
| Llama-MEMIT | ORBFH_HIST | true | 10 | 8.57820667028427 | 9.035438060760498 | 11.950099945068358 | 14.342613220214844 |
| Llama-MEMIT | JV_NATIVE | new | 10 | 0.007166834984673187 | 0.001989668991882354 | 0.0226367050781846 | 0.0342121385037899 |
| Llama-MEMIT | JV_NATIVE | true | 10 | 11.626633071899414 | 11.14310073852539 | 18.188159370422362 | 18.598630905151367 |
| Llama-MEMIT | ORB_RAY_N | new | 10 | 0.010499974258709698 | 0.0030471974750980735 | 0.034688260406255715 | 0.04816719889640808 |
| Llama-MEMIT | ORB_RAY_N | true | 10 | 11.218339109420777 | 10.831472873687744 | 16.920194625854492 | 17.722814559936523 |
| Llama-AlphaEdit | PRE_EDIT_WARM | new | 10 | 10.433673936128617 | 9.883934020996094 | 16.128209686279295 | 18.259124755859375 |
| Llama-AlphaEdit | PRE_EDIT_WARM | true | 10 | 3.665742865204811 | 2.5773656368255615 | 7.334599018096922 | 12.72107982635498 |
| Llama-AlphaEdit | O_NATIVE | new | 10 | 0.0008299750235892134 | 0.0004883405927103013 | 0.002157094026915729 | 0.0027315232437103987 |
| Llama-AlphaEdit | O_NATIVE | true | 10 | 13.450815582275391 | 13.78626537322998 | 16.576036834716795 | 16.79387855529785 |
| Llama-AlphaEdit | ORBFH_HIST | new | 10 | 0.00717485586865223 | 0.0013114495086483657 | 0.019126143306493752 | 0.041642047464847565 |
| Llama-AlphaEdit | ORBFH_HIST | true | 10 | 10.847871017456054 | 11.24983024597168 | 12.916317367553711 | 13.990835189819336 |
| Llama-AlphaEdit | JV_NATIVE | new | 10 | 0.0017379082077241038 | 0.0008087434689514339 | 0.0037516211858019213 | 0.008466540835797787 |
| Llama-AlphaEdit | JV_NATIVE | true | 10 | 12.145489692687988 | 12.196006298065186 | 14.546349143981933 | 15.130844116210938 |
| Llama-AlphaEdit | ORB_RAY_N | new | 10 | 0.0019747731814277357 | 0.0008133321534842253 | 0.004012065776623783 | 0.010535219684243202 |
| Llama-AlphaEdit | ORB_RAY_N | true | 10 | 12.240053081512452 | 12.410650730133057 | 14.665355491638183 | 15.165214538574219 |
| Qwen-MEMIT | PRE_EDIT_WARM | new | 10 | 10.281250154972076 | 11.14622163772583 | 15.599754047393798 | 16.298507690429688 |
| Qwen-MEMIT | PRE_EDIT_WARM | true | 10 | 4.442137086391449 | 3.4870256185531616 | 8.121063709259031 | 10.830296516418457 |
| Qwen-MEMIT | O_NATIVE | new | 10 | 0.02273912318632938 | 0.008312877966091037 | 0.07104732990264892 | 0.08537805080413818 |
| Qwen-MEMIT | O_NATIVE | true | 10 | 12.17908353805542 | 11.120818614959717 | 18.43181457519531 | 19.909873962402344 |
| Qwen-MEMIT | ORBFH_HIST | new | 10 | 0.02207605162402615 | 0.0087292673997581 | 0.06589551791548728 | 0.08886975795030594 |
| Qwen-MEMIT | ORBFH_HIST | true | 10 | 12.403430223464966 | 11.172930717468262 | 18.280702972412108 | 20.055753707885742 |
| Qwen-MEMIT | JV_NATIVE | new | 10 | 0.019593577925115825 | 0.011243402026593685 | 0.04880053848028182 | 0.06892229616641998 |
| Qwen-MEMIT | JV_NATIVE | true | 10 | 11.651789808273316 | 10.327197551727295 | 18.204209518432616 | 18.42952537536621 |
| Qwen-MEMIT | ORB_RAY_N | new | 10 | 0.02230460859136656 | 0.005345453973859549 | 0.07343218550086021 | 0.09105068445205688 |
| Qwen-MEMIT | ORB_RAY_N | true | 10 | 12.338445138931274 | 10.851202487945557 | 18.718949127197266 | 19.62421226501465 |
| Qwen-AlphaEdit | PRE_EDIT_WARM | new | 10 | 10.27273126244545 | 11.065819263458252 | 15.766256618499755 | 16.34977912902832 |
| Qwen-AlphaEdit | PRE_EDIT_WARM | true | 10 | 4.408887434005737 | 3.4681296348571777 | 8.017069911956787 | 10.87627124786377 |
| Qwen-AlphaEdit | O_NATIVE | new | 10 | 0.024586987151997163 | 0.015345499385148287 | 0.05493420660495756 | 0.10342944413423538 |
| Qwen-AlphaEdit | O_NATIVE | true | 10 | 11.894517612457275 | 11.124853134155273 | 17.320849800109862 | 18.085905075073242 |
| Qwen-AlphaEdit | ORBFH_HIST | new | 10 | 0.02518622565548867 | 0.0174638070166111 | 0.0544111482799053 | 0.10967838019132614 |
| Qwen-AlphaEdit | ORBFH_HIST | true | 10 | 11.988763904571533 | 11.944864749908447 | 17.120030403137207 | 18.934499740600586 |
| Qwen-AlphaEdit | JV_NATIVE | new | 10 | 0.027896269515622407 | 0.01593772368505597 | 0.05474599227309225 | 0.10170555114746094 |
| Qwen-AlphaEdit | JV_NATIVE | true | 10 | 11.526485919952393 | 11.755055904388428 | 17.115016746520997 | 17.264497756958008 |
| Qwen-AlphaEdit | ORB_RAY_N | new | 10 | 0.026387392182368786 | 0.017640945967286825 | 0.055317191407084444 | 0.11033514887094498 |
| Qwen-AlphaEdit | ORB_RAY_N | true | 10 | 11.893744969367981 | 11.77948808670044 | 16.85926761627197 | 18.373624801635742 |

### Rephrase NLL

| cell | arm | target | n | mean | median | p90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-MEMIT | PRE_EDIT_WARM | new | 20 | 10.903824067115783 | 10.937126159667969 | 14.200730895996095 | 16.33244514465332 |
| Llama-MEMIT | PRE_EDIT_WARM | true | 20 | 5.830285365134477 | 4.807765483856201 | 12.722092056274416 | 14.794351577758789 |
| Llama-MEMIT | O_NATIVE | new | 20 | 4.662789693474769 | 3.1092811822891235 | 11.147288322448732 | 14.150289535522461 |
| Llama-MEMIT | O_NATIVE | true | 20 | 7.138239073753357 | 5.96640944480896 | 12.859624958038331 | 15.878514289855957 |
| Llama-MEMIT | ORBFH_HIST | new | 20 | 4.518885692209006 | 2.257608413696289 | 11.053616428375246 | 14.232908248901367 |
| Llama-MEMIT | ORBFH_HIST | true | 20 | 7.238162410259247 | 5.939977169036865 | 12.855417728424074 | 15.921192169189453 |
| Llama-MEMIT | JV_NATIVE | new | 20 | 4.185014405474067 | 2.2557289600372314 | 10.820630264282228 | 13.977691650390625 |
| Llama-MEMIT | JV_NATIVE | true | 20 | 8.614836764335632 | 8.109369277954102 | 13.162505626678469 | 15.674517631530762 |
| Llama-MEMIT | ORB_RAY_N | new | 20 | 4.066931946203113 | 1.844930112361908 | 10.781090831756593 | 14.47419548034668 |
| Llama-MEMIT | ORB_RAY_N | true | 20 | 8.83728199005127 | 8.639747142791748 | 13.386882972717286 | 16.058738708496094 |
| Llama-AlphaEdit | PRE_EDIT_WARM | new | 20 | 10.872598814964295 | 10.880557537078857 | 14.06918525695801 | 16.429279327392578 |
| Llama-AlphaEdit | PRE_EDIT_WARM | true | 20 | 5.761847653239966 | 4.670449733734131 | 12.452589130401613 | 14.865983009338379 |
| Llama-AlphaEdit | O_NATIVE | new | 20 | 3.4907079760712802 | 1.6313911080360413 | 8.654107856750489 | 14.304123878479004 |
| Llama-AlphaEdit | O_NATIVE | true | 20 | 9.744598460197448 | 10.159447193145752 | 13.8880729675293 | 15.841861724853516 |
| Llama-AlphaEdit | ORBFH_HIST | new | 20 | 3.906646784581244 | 2.558674693107605 | 8.867727756500244 | 14.128313064575195 |
| Llama-AlphaEdit | ORBFH_HIST | true | 20 | 8.390787017345428 | 8.093183994293213 | 12.585331153869634 | 15.758607864379883 |
| Llama-AlphaEdit | JV_NATIVE | new | 20 | 3.6655573994881707 | 2.0586937069892883 | 8.42460069656372 | 13.778962135314941 |
| Llama-AlphaEdit | JV_NATIVE | true | 20 | 8.812066078186035 | 8.983735084533691 | 13.422956371307377 | 15.852181434631348 |
| Llama-AlphaEdit | ORB_RAY_N | new | 20 | 3.6081478256266566 | 2.180185317993164 | 8.505961418151855 | 14.168261528015137 |
| Llama-AlphaEdit | ORB_RAY_N | true | 20 | 9.060776424407958 | 9.190951824188232 | 13.145081138610845 | 15.771109580993652 |
| Qwen-MEMIT | PRE_EDIT_WARM | new | 20 | 11.139257335662842 | 11.252816200256348 | 15.51852493286133 | 17.673789978027344 |
| Qwen-MEMIT | PRE_EDIT_WARM | true | 20 | 6.10471478253603 | 5.400911092758179 | 12.584426307678225 | 17.024417877197266 |
| Qwen-MEMIT | O_NATIVE | new | 20 | 3.8421625872142613 | 2.537485957145691 | 7.6628279685974166 | 14.3486328125 |
| Qwen-MEMIT | O_NATIVE | true | 20 | 10.3727285861969 | 10.415910720825195 | 15.644338512420655 | 16.370222091674805 |
| Qwen-MEMIT | ORBFH_HIST | new | 20 | 3.4846631885739043 | 2.062923789024353 | 7.761762142181399 | 12.669469833374023 |
| Qwen-MEMIT | ORBFH_HIST | true | 20 | 10.562798428535462 | 10.744746685028076 | 16.009951782226562 | 16.225021362304688 |
| Qwen-MEMIT | JV_NATIVE | new | 20 | 3.4189405342563988 | 2.2073928117752075 | 8.083009958267214 | 9.22010326385498 |
| Qwen-MEMIT | JV_NATIVE | true | 20 | 10.35631034374237 | 10.530705451965332 | 15.107198429107667 | 16.324308395385742 |
| Qwen-MEMIT | ORB_RAY_N | new | 20 | 3.3340684955241158 | 2.187924385070801 | 7.5132306098937995 | 11.18818473815918 |
| Qwen-MEMIT | ORB_RAY_N | true | 20 | 10.531993770599366 | 10.78128719329834 | 15.976736450195313 | 16.13017463684082 |
| Qwen-AlphaEdit | PRE_EDIT_WARM | new | 20 | 11.153654885292053 | 11.223045349121094 | 15.432604026794436 | 17.601951599121094 |
| Qwen-AlphaEdit | PRE_EDIT_WARM | true | 20 | 6.10173893943429 | 5.422556638717651 | 12.46306257247925 | 17.012889862060547 |
| Qwen-AlphaEdit | O_NATIVE | new | 20 | 3.732687901845202 | 1.8243155479431152 | 8.304220199584964 | 13.969404220581055 |
| Qwen-AlphaEdit | O_NATIVE | true | 20 | 10.912854075431824 | 10.45271348953247 | 14.806382465362551 | 16.43844985961914 |
| Qwen-AlphaEdit | ORBFH_HIST | new | 20 | 3.7714913566596806 | 2.0845569372177124 | 8.093170070648197 | 14.256863594055176 |
| Qwen-AlphaEdit | ORBFH_HIST | true | 20 | 10.964833664894105 | 10.562414646148682 | 14.796477127075198 | 16.65963363647461 |
| Qwen-AlphaEdit | JV_NATIVE | new | 20 | 3.8334230839740484 | 2.5753633975982666 | 7.973394250869752 | 14.758538246154785 |
| Qwen-AlphaEdit | JV_NATIVE | true | 20 | 10.902023100852967 | 10.810670375823975 | 14.254963111877444 | 16.454374313354492 |
| Qwen-AlphaEdit | ORB_RAY_N | new | 20 | 3.7435637368820607 | 2.0498303174972534 | 8.079122352600098 | 14.427062034606934 |
| Qwen-AlphaEdit | ORB_RAY_N | true | 20 | 10.804666018486023 | 10.556317329406738 | 14.832003974914553 | 16.592845916748047 |

## 1. 수학 및 fidelity

CPU algebra/overlay 체크와 GPU raw-direction FD, RHS scaling, joint-state materialized parity는 각각 cpu_algebra_checks.json, gpu_fidelity_checks.json에 실제 결과를 결속했다. 주 controller는 FP64 nonnegative active-set reference solver이며 model/forward/overlay는 FP32다. MEMIT의 native ephemeral FP64 solve는 보존한다. Fixed qN_ref와 entry-normalization을 node 또는 comparator별로 다시 정하지 않는다.

## 2. Historical anomaly

LM-ORBFH B9, LM-JAC B6, LA-JAC B5: historical report ref=f2dcfd4ab6fcb2917ae0a29cbc384cf95a82eb3c. 그 보고서의 execution sources는 별도 source.lock.json에 있다. exact historical entry W/method-state/z가 이 pilot 입력에 없으므로 HISTORICAL_STATE_UNAVAILABLE. 과거 B1–B8 replay는 하지 않았다.

## 3. Same-state physical turning

same_state_fields.csv는 같은 raw dictionary/JVP로 native joint, ORB ray, diagonal ray, raw ORB snapshot 및 실제 Frobenius metric을 비교한다. gᵀc의 차이, native Rturn>0, endpoint usefulness는 서로 다른 판정이다. 영벡터 angle은 N/A이며 primary 선택에 shadow λ 또는 normalization을 사용하지 않았다.

| cell | recorded | nonzero_angle_rows | Rturn_positive | Rturn_mean | Rturn_median | Rturn_p90 | Rturn_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8 | 8 | 8 | 0.5283784076839181 | 0.6058723659610953 | 0.6289533763346673 | 0.6367029321756249 |
| 1 | 8 | 8 | 8 | 0.36912289565104417 | 0.4605416458004704 | 0.46975113728530715 | 0.46975113728530715 |
| 2 | 8 | 8 | 8 | 0.5314592937258023 | 0.531614538492633 | 0.5533451418061833 | 0.5533451418061833 |
| 3 | 8 | 8 | 8 | 0.476314322748041 | 0.4942766420329331 | 0.5261065812845956 | 0.5359679303876207 |

## 4. Actual path

trajectory_nodes.csv는 node entry/exit V, raw/native normalized action, integrated work E, path length L, Euler model error, response velocity mismatch 및 barrier defect를 분리한다. 바뀌는 dictionary의 계수 합을 physical displacement로 쓰지 않는다. 실제 materialized block ΔW의 native/Frobenius action은 pilot_main_table.csv에 별도 기록한다. finite Euler defect/near-stall/낮은 성능은 제외 사유가 아니다.

## 5. Refinement

공통 four-cell primary 표가 먼저다. D2 fixed T2 N2/4/8은 별도 후속 budget/공통 gate를 통과한 경우에만 수행한다. 이 primary package에서 미실행 refinement를 convergence PASS로 주장하지 않는다.

## 6. Matched-progress usefulness

사전 값 V/V0={.75,.5,.25}, saved node의 절대 오차 .02 이내만 비교 가능하다. interpolation은 금지한다. 같은 V는 같은 efficacy가 아니다. Primary terminal이 서로 다른 progress이면 단순 endpoint 성능 차이를 matched-progress 이득으로 부르지 않는다. 중간 상태의 evaluation이 없으면 NOT_EVALUATED이며 추정하지 않는다.

## 7. D10A loss

old_edit_metrics.csv는 request별 before/after NLL advantage, entry-success/post-failure 및 initially-failed를 분리한다. old prompts/margins는 평가 callback에서만 읽으며 controller·normalization·stopping에 입력하지 않는다. 파생 first-hit 평가는 주 endpoint의 old-edit 관측값을 대체하지 않는다.

## 8. Native와 Frobenius

QN은 native S에 의한 action을 entry qN_ref로 나눈 것이다. 다른 weight block의 native-whitened G=I는 Frobenius metric을 뜻하지 않는다. GF는 독립 계산한다. absolute cross-model action을 같은 척도라고 해석하지 않는다.

## 9. Compute와 미실행

compute_accounting.csv의 실제 model.forward/JVP 수와 CUDA-synchronized wall을 분리한다. FLOPs 추정으로 대체하지 않는다. target/setup/endpoint observation의 비용 범위를 구분하며, native dense endpoint action의 관측 비용은 evaluator 범위다. refinement/audit는 남은 2GPUh/cell 및 8GPUh total 예산으로만 진행한다. 예산 부족은 NOT_RUN_BUDGET이며 outcome 기반 탈락은 없다.

## 10. 독립성 및 한계

Source 29884f208bca5afc2367c67510779b4a674cbbb1 / tree 5c7310d74357461f3040cf22342282b69cda7baf. 표본 38개 hash-rank seal과 reserved 1000 overlap0은 sample.lock.json에 결속한다. 4 cells×4 primary arms×10 requests가 완성 분모다. 현재 실제 분모는 16/16 arm endpoints, 160/160 request endpoints다. raw 입력은 수정하지 않았고 보고서에는 prompts/targets/token IDs/weight/cache tensors를 복사하지 않는다. automatic promotion·rescue·budget expansion=0.

Analysis source f50e2e84cfa344ec57c63791660a83fa9cb8312a / tree c9eb6eefac4a0ddf41614c13a4414ff872b992d2는 실행 source와 분리했다. `integrity.csv`는 source/entry/fixed-z/restore를 독립 재검산하며, `paired_endpoint_deltas.csv`는 cell·request·prompt 단위 method−Official 차이다. 다른 model/family 사이 absolute metric 크기를 직접 비교하지 않는다. native scalar 기하와 Frobenius 기하의 동등성은 가정하지 않는다.

### 누락/관측 경계

| field | status | impact |
| --- | --- | --- |
| Frobenius_cosine | NOT_RECORDED_SCHEMA_GAP | QN angle is not Frobenius angle; no equivalence claim |
| intermediate_matched_progress_endpoint_evaluation | NOT_EVALUATED | saved residual alone is not matched-quality/retention evidence |
| Official_ORBFH_continuous_native_path | NOT_RECORDED_SCHEMA_GAP | actual endpoint native action present; do not infer integrated work |
| model_forward_setup_breakdown | NOT_RECORDED_SCHEMA_GAP | scoped arm counts and setup seconds remain factual |

### 계산량과 setup

| cell | arm | write_wall_seconds | eval_wall_seconds | model_forward_calls | main_JVP_calls | diagnostic_JVP_calls | peak_gpu_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | O_NATIVE | 41.92959468066692 | 43.852470023557544 | 53 | 0 | 0 | 42359669248 |
| 0 | ORBFH_HIST | 172.11152659915388 | 44.295591281726956 | 145 | 20 | 0 | 42359669248 |
| 0 | JV_NATIVE | 197.12174249999225 | 39.31276427023113 | 93 | 20 | 0 | 42359669248 |
| 0 | ORB_RAY_N | 208.10002791509032 | 49.47781445644796 | 93 | 20 | 0 | 42359669248 |
| 1 | O_NATIVE | 33.36494185589254 | 43.14062718115747 | 58 | 0 | 0 | 40242872832 |
| 1 | ORBFH_HIST | 228.3239911980927 | 78.26092990487814 | 191 | 20 | 0 | 40242872832 |
| 1 | JV_NATIVE | 144.77473674714565 | 42.07142850756645 | 98 | 20 | 0 | 40242872832 |
| 1 | ORB_RAY_N | 154.80512097850442 | 40.11251107417047 | 98 | 20 | 0 | 40242872832 |
| 2 | O_NATIVE | 71.51967075653374 | 62.18168779462576 | 53 | 0 | 0 | 45709838848 |
| 2 | ORBFH_HIST | 253.10079197771847 | 166.83791914768517 | 188 | 20 | 0 | 45709838848 |
| 2 | JV_NATIVE | 353.82796165905893 | 65.57990050315857 | 93 | 20 | 0 | 45709838848 |
| 2 | ORB_RAY_N | 392.8418708052486 | 71.20127020962536 | 93 | 20 | 0 | 45709838848 |
| 3 | O_NATIVE | 38.40893394686282 | 52.62232484854758 | 58 | 0 | 0 | 42296334848 |
| 3 | ORBFH_HIST | 270.8146279975772 | 104.29782945290208 | 193 | 20 | 0 | 42296334848 |
| 3 | JV_NATIVE | 199.20371014997363 | 51.91825980320573 | 98 | 20 | 0 | 42296334848 |
| 3 | ORB_RAY_N | 208.0587241537869 | 52.36280737258494 | 98 | 20 | 0 | 42296334848 |

`write_wall_seconds`는 전체 arm 시간에서 공통 endpoint evaluator callback 시간을 제외한 관측값이며, dictionary/solve/shadow algebra/transaction 비용을 포함한다. setup/target/model load는 `setup_accounting.csv`에 별도로 기록한다. 동시 GPU contention 및 CPU metric 관측 비용이 wall에 영향을 줄 수 있으므로 FLOPs 또는 순수 GPU kernel 비용이라고 부르지 않는다.

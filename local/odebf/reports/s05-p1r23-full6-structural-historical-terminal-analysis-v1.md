# P1R23 Full-6 Structural Historical: 터미널 분석

## 범위와 판독 규칙

- **FACT**는 지정된 10개 유효 root의 raw-free `manifest.json`, `terminal.json`, `raw/round-00..10-endpoint.json` 및 ODE의 raw-free accepted-field receipt에서 직접 확인한 값이다.
- **INFERENCE**는 FACT로부터 제한적으로 해석한 결과다. 인과·일반화 주장은 별도로 표시한다.
- **NOT_RECORDED**는 이 허가된 수령증 집합에 없는 값이다. raw prompt/target/tensor/private log는 열지 않았다.
- 유효 비교축은 Llama-3-8B-Instruct와 Qwen2.5-7B-Instruct의 `BG/RS × NoSoft/Soft` 완전 2×2이며, AlphaEdit은 각 모델의 공통 native 비교군이다. 100 logical request는 10개의 joint B10 commit으로 처리되었다.

## 결론

**FACT.** 10/10 root에서 manifest→terminal SHA, 10개 round receipt SHA, W0 endpoint SHA가 실제 파일과 모두 일치했다. 각 root는 10 persistent commit, 10 history append, 최종 history 100, 누적 B10 endpoint evaluation 55, retry/backtracking 0, 최종 W0 restore=true, promotion=false를 기록한다.

**FACT.** 8개 ODE root의 640개 accepted field는 모두 `FULL_SIX_FIXED`, context 수 6, K=8, `h=1/8`, `tau=1`, certificate passed였다. 최대 |a^T v-q residual|은 3.53e-13, 최대 simplex residual은 1.82e-13이고, energy certificate는 허용오차 내에서 통과했다.

**INFERENCE.** full-6 controller와 short sequential(10 B10/100 edit) 운용의 기계적 증거는 충분하지만, Soft의 보존 이득은 모델·allocation에 따라 일관되지 않았다. 따라서 본 결과는 short-sequential mechanism evidence이며 lifelong/main-table 우월성 주장이 아니다.

## t=10 공통 비교

`누적 E/G/L`은 100/200/1000 분모, `현재`는 마지막 B10의 10/20/100, `이력`은 그 이전 9 B10의 90/180/900이다. KL은 teacher-KL mean이다.

| 모델 | arm | Tech | 누적 E/G/L | 현재 E/G/L | 이력 E/G/L | KL@10 |
|---|---|---:|---:|---:|---:|---:|
| Llama | AlphaEdit | v1 | 100/190/843 | 10/20/72 | 90/170/771 | 0.4239 |
| Llama | BG-NoSoft | R1 | 98/179/834 | 10/19/75 | 88/160/759 | 1.9147 |
| Llama | BG-Soft | R1 | 98/181/825 | 10/18/74 | 88/163/751 | 2.0810 |
| Llama | RS-NoSoft | R1 | 100/187/822 | 10/18/72 | 90/169/750 | 4.5151 |
| Llama | RS-Soft | R1 | 100/183/823 | 10/16/70 | 90/167/753 | 4.2688 |
| Qwen | AlphaEdit | v1 | 100/192/814 | 10/20/74 | 90/172/740 | 0.0994 |
| Qwen | BG-NoSoft | R2 | 96/163/841 | 9/16/75 | 87/147/766 | 0.0091 |
| Qwen | BG-Soft | R2 | 96/161/841 | 9/15/75 | 87/146/766 | 0.0077 |
| Qwen | RS-NoSoft | R2 | 98/170/840 | 10/17/75 | 88/153/765 | 0.0064 |
| Qwen | RS-Soft | R3 | 97/172/845 | 9/15/76 | 88/157/769 | 0.0062 |

**FACT.** Qwen의 final KL은 네 ODE arm 모두 AlphaEdit보다 낮지만, Llama의 ODE KL은 AlphaEdit보다 높다. 같은 allocation에서 Soft의 final KL은 Qwen BG/RS에서 각각 낮고, Llama는 BG에서 높고 RS에서 낮다. 효능·일반화·locality도 Soft의 균일한 우위를 보이지 않는다.

## 정확한 checkpoint cadence

표의 각 셀은 해당 outer checkpoint의 누적 `E/G/L`이다. t=2의 KL은 의도적으로 미측정(`NOT_EVALUATED_OUTER_CHECKPOINT_SCHEDULE`); KL은 t={1,5,10}에서만 수집되었다.

| 모델·arm | t=1 (10/20/100) | t=2 (20/40/200) | t=5 (50/100/500) | t=10 (100/200/1000) |
|---|---:|---:|---:|---:|
| Llama AlphaEdit | 10/19/92 | 20/39/174 | 50/92/435 | 100/190/843 |
| Llama BG-NoSoft | 10/18/92 | 20/35/176 | 50/88/440 | 98/179/834 |
| Llama BG-Soft | 10/18/92 | 20/35/178 | 50/88/441 | 98/181/825 |
| Llama RS-NoSoft | 10/19/92 | 20/36/176 | 50/91/439 | 100/187/822 |
| Llama RS-Soft | 10/19/92 | 20/36/176 | 50/91/440 | 100/183/823 |
| Qwen AlphaEdit | 10/20/92 | 20/40/166 | 50/97/408 | 100/192/814 |
| Qwen BG-NoSoft | 10/18/92 | 20/34/168 | 49/83/425 | 96/163/841 |
| Qwen BG-Soft | 10/17/91 | 20/32/170 | 49/84/426 | 96/161/841 |
| Qwen RS-NoSoft | 10/18/92 | 20/35/169 | 49/85/424 | 98/170/840 |
| Qwen RS-Soft | 10/17/92 | 20/35/169 | 49/86/424 | 97/172/845 |

KL mean `(t=1, t=5, t=10)`은 Llama에서 Alpha `(0.0125, 0.0986, 0.4239)`, BG-No `(0.0064, 0.1786, 1.9147)`, BG-Soft `(0.0071, 0.2127, 2.0810)`, RS-No `(0.0117, 0.8103, 4.5151)`, RS-Soft `(0.0109, 0.6284, 4.2688)`이고, Qwen에서 Alpha `(0.0051, 0.0697, 0.0994)`, BG-No `(0.00125, 0.00758, 0.00910)`, BG-Soft `(0.00119, 0.00551, 0.00769)`, RS-No `(0.00204, 0.00362, 0.00638)`, RS-Soft `(0.00118, 0.00507, 0.00617)`이다.

## Full-6, transaction, history 검증

**FACT.** 수식 계약은 `context_ordinals=(0,1,2,3,4,5)`, `L_full6=sum(loss)/global_denominator`, 그리고 full-6 physical signed slope `a_{k,l}=-d L_new^full6(W_k+εB_{k,l})/dε|_{ε=0}`이다. 모든 ODE accepted receipt에서 full-six fixed mode와 six routing contexts가 확인되었다. NoSoft/Soft는 같은 progress-simplex q를 쓰고 Soft는 structural P/H 목적 아래 energy가 Neutral 이하인 allocation을 선택한다.

**FACT.** 각 ODE cell은 80 accepted state(10×K8)를 남겼고, physical context=6, tau-after=`1/8..1`, reject/retry/backtracking=0, functional routing influence=0, inner-K functional probe=0이다. endpoint held-out/first-hit evaluator도 0이다.

**FACT.** 각 root의 round-00 W0 receipt, round-01..10 endpoint receipt, terminal, manifest chain이 닫힌다. terminal의 `round_receipt_sha256` 배열은 10개의 실제 endpoint SHA와, manifest의 같은 배열과 동일하며, `final_w0_restored=true`다. history는 commit 성공 뒤 정확히 1회 append되어 entry count `0,10,...,90`, after count `10,20,...,100`을 따른다. 중복 append=0이고 future-batch controller access=0이다.

## Historical H와 structural/functional 관측

| 모델·allocation | NoSoft H input / influence | Soft H input (t=1..10) | Soft influence counts (t=1..10) |
|---|---:|---:|---:|
| Llama BG | 0 / 0 | 0,8,8,8,8,8,8,8,8,8 | 0,8,8,7,8,7,8,8,7,8 |
| Llama RS | 0 / 0 | 0,8,8,8,8,8,8,8,8,8 | 0,8,7,7,7,7,7,7,7,7 |
| Qwen BG | 0 / 0 | 0,8,8,8,8,8,8,8,8,8 | 0,8,6,3,6,8,7,8,7,8 |
| Qwen RS | 0 / 0 | 0,8,8,8,8,8,8,8,8,8 | 0,8,6,7,6,8,7,8,8,7 |

**FACT.** Soft는 모든 cell에서 t=2..10마다 H를 8개 K-state router input으로 받았고, 각 outer round에서 양의 decision influence를 가졌다(누적 69, 64, 61, 65 state). NoSoft는 H input/influence 모두 0이다. `historical_h_persistent_inactivity=false`다.

**FACT.** historical sketch는 ODE 전 cell에서 rank=100, history item=100, layer=4..8, raw-history replay=0이다. layer별 row count=100이며 width는 Llama 14,336, Qwen 18,944다. 따라서 dense d×d replay가 아닌 fixed-rank receipt로만 historical state를 보존했다.

**FACT.** functional P/H의 online decision influence와 inner-K functional probe는 모두 0이다. functional 관측은 outer endpoint의 current/all-history E/G/L과 t={1,5,10} teacher-KL에 한정된다. final terminal functional barrier의 historical sample count는 이 raw-free receipt에서 0이므로, 그 값으로 functional-history 보존을 주장할 수 없다.

## Allocation, capacity, actual/predicted progress

아래 `load/max-share`는 t=10 누적 layer load 합과 최대 layer 비율, `actual/predicted`는 마지막 trajectory의 refreshed-field delayed progress 합 비율이다.

| 모델·allocation | NoSoft load/max-share/ratio | Soft load/max-share/ratio | FACT 판독 |
|---|---:|---:|---|
| Llama BG | 29.48 / .269 / .529 | 30.44 / .283 / .504 | Soft concentration·realization 모두 개선 아님 |
| Llama RS | 31.11 / .268 / .458 | 30.73 / .281 / .510 | realization은 개선, concentration은 증가 |
| Qwen BG | 251.61 / .355 / .595 | 149.06 / .343 / .625 | load·concentration 감소, realization 개선 |
| Qwen RS | 233.06 / .346 / .628 | 158.21 / .274 / .659 | load·concentration 감소, realization 개선 |

**FACT.** Soft accepted-field의 same-state `risk(Soft)-risk(Neutral)` mean은 Llama BG -0.269, Llama RS -0.196, Qwen BG -0.568, Qwen RS -0.564이며 모두 음수 범위였다. 단, structural H/P의 절대 단위는 모델별 scale가 다르므로 모델 간 절대값 비교는 하지 않는다.

**INFERENCE.** Qwen에서 Soft는 두 allocation 모두 t=10 load 및 max-share를 낮추고 realization ratio를 높였지만, Llama에서는 uniform하지 않다. 따라서 capacity concentration 감소는 Qwen의 조건부 mechanism evidence이지 전 모델 일반화가 아니다.

## Runtime, F/B, tokens, memory, scaling

ODE에는 pure edit-core wall이 직접 기록되었고, AlphaEdit에는 동등한 exact pure-core wall 분해가 기록되지 않았다. ODE의 `F/B`는 full-6 edit core model-forward/backward이며 evaluator는 별도다.

| 모델·arm | total wall s | pure core s | evaluator s | F/B | processed/padded tokens | peak GPU alloc/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| Llama AlphaEdit | 801.3 | NOT_RECORDED | 307.9 | 3212/2359 | 533946 / NOT_RECORDED | 19.57 / 21.30 |
| Llama BG-No/Soft | 1850.1 / 1803.0 | 1215.0 / 1167.0 | 308.0 / 309.2 | 260/160 each | 270042 / 354960 | 22.56 / 23.50 each |
| Llama RS-No/Soft | 1807.3 / 1897.2 | 1172.6 / 1254.3 | 308.0 / 308.0 | 260/160 each | 270042 / 354960 | 22.56 / 23.50 each |
| Qwen AlphaEdit | 774.2 | NOT_RECORDED | 303.9 | 2131/1278 | 383263 / NOT_RECORDED | 22.24 / 24.54 |
| Qwen BG-No/Soft | 2219.0 / 2117.0 | 1509.5 / 1405.3 | 304.6 / 304.7 | 260/160 each | 247604 / 358080 | 21.62 / 23.90 each |
| Qwen RS-No/Soft | 2110.3 / 2305.4 | 1402.3 / 1585.9 | 305.1 / 306.1 | 260/160 each | 247604 / 358080 | 21.62 / 23.90 each |

**FACT.** evaluator는 모든 root에서 553 forward이며 Llama 162,349 tokens, Qwen 151,369 tokens다. ODE/Alpha total-wall ratio는 Llama 2.25–2.37×, Qwen 2.73–2.98×다. Qwen RS-Soft(R3)의 2305.4 s는 해당 모델·method matrix의 최대 wall이다.

**NOT_RECORDED.** AlphaEdit의 exact pure edit-core wall과 padded-token count는 terminal receipt에 별도 분해가 없다. 따라서 Alpha와 ODE의 pure-core-only ratio는 산출하지 않았다.

## TECH 경계와 비교 한계

| namespace | 유효 ODE cell | source head |
|---|---|---|
| TECH-R1 | Llama BG-No, BG-Soft, RS-No, RS-Soft | `1b8ab5659c922d36e324abe9f055f71b641129cc` |
| TECH-R2 | Qwen BG-No, RS-No | `48ac36262f58b67ce1dfaf1f36daabaa10d8724e` |
| TECH-R2 | Qwen BG-Soft | `94d126876d51514b198a0311683c2b0849eaca8f` |
| TECH-R3 | Qwen RS-Soft | `7ffd70169f19cfcd3de798053b4f8f418cabd1a9` |

AlphaEdit 양 모델은 v1/source `bfbb7ad9da6b8825852bbb7939ee2e7a1624e88a`다. **FACT:** 위 namespace/source boundary는 receipt provenance다. **NOT_RECORDED:** 이 authorized terminal set만으로 R1/R2/R3 source-head 차이를 과학적 treatment 효과로 분해할 수 없다. 따라서 Qwen RS Soft는 R3의 유효 terminal로만, matched soft/no-soft causal 쌍은 동일 allocation/model의 최종 수치와 함께 제한적으로 읽는다.

P1R21은 이 보고서의 재계산·새 atomic 보고서 대상이 아니다. 필요할 때만 short-sequential 맥락의 서술적 선행 항목으로 연결할 수 있으며, 여기서는 P1R23 terminal evidence를 대체하지 않는다.

## Attachment의 다섯 질문에 대한 답

1. **full-6 physical slope가 Qwen route/objective mismatch를 줄였는가?** **FACT:** 80/80 Qwen trajectory state가 six fixed context로 실행되어 objective/slope context 불일치는 발생하지 않았다. **NOT_RECORDED:** 유효 root에 matched rotating-2-of-6 control이 없으므로 "줄였다"는 인과 차이는 산출 불가다.
2. **t>=2 H가 allocation을 실제로 바꾸었는가?** **FACT:** Soft의 매 t=2..10에서 H input=8, influence>0; NoSoft=0이다. 답은 Soft에 대해 예다.
3. **Soft가 early/history retention collapse를 늦췄는가?** **INFERENCE:** Qwen RS는 t=10 historical G/L이 157/180, 769/900으로 NoSoft 153/180, 765/900보다 높지만, Qwen BG와 Llama 두 allocation에서 균일한 개선은 없다. 일반적 "늦춤"은 지지되지 않는다.
4. **같은 new-edit quality에서 concentration 또는 structural H/P가 줄었는가?** **FACT:** Qwen은 Soft가 load/max-share를 두 allocation 모두 낮췄고 same-state structural-risk difference도 음수다. Llama는 max-share가 상승했다. **INFERENCE:** Qwen 조건부 evidence는 있으나 보편적 결과는 아니다.
5. **추가 preservation signal이 wall-time 비용에 상응하는가?** **FACT:** Soft의 total wall 변화는 Llama BG -47 s, Llama RS +90 s, Qwen BG -102 s, Qwen RS +195 s로 일률적이지 않다. **NOT_RECORDED:** 사전 효용/비용 threshold가 없고 promotion=false이므로 상응 여부의 최종 판정은 하지 않는다.

## 경계

이 결과는 `promotion=false`를 유지한다. 10 outer B10의 sealed stream에서 terminal/checkpoint observation을 한 short-sequential pilot이며, lifelong retention, broad main-table superiority, 또는 causal benefit of TECH namespace는 주장하지 않는다.

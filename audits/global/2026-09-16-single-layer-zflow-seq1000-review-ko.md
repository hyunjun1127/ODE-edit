# SL-ZFlow 실제 SEQ1000 실험 상세 리뷰

작성일: 2026-09-16. 사용자 요청: “z flow부분 실험 자세히 리뷰하라”.

## 목표 해석 정정 — 사용자 후속 설명 반영

사용자가 명시한 주목표는 **필요한 만큼 최적화하여 더 좋은 편집·보존 균형에 도달하는 것**이다. 더 적은 step 또는 조기 종료는 필수 성공 조건이 아니다. 추가 최적화와 write가 유리할 수 있다는 REFIT4 등의 관측 역시 설계 동기에 포함된다. 아래 비용 분석은 계산 낭비 진단이며 방법의 목표를 조기 종료로 한정하지 않는다.

따라서 10/10 RESOURCE_STOP은 **현재 예산에서 1차 종료 조건에 도달하지 않았으며 추가 진행의 결과를 측정하지 않았음**을 뜻한다. 그 사실만으로 품질 목표의 실패나 더 많은 최적화의 무용성을 판정하지 않는다. 현재 설정의 PS·NS 열세는 별도의 실제 성능 관측이다.

V1은 각 후보 W_entry+XB의 실제 all-token 효과를 반영한 fresh suffix loss/gradient로 계속 최적화하므로, 변경된 모델 상태에서 추가 최적화하는 feedback은 포함한다. Edit loss가 충분히 줄면 C가 증가하는 step도 수용할 수 있다. 그러나 고정 λ=1, batch-entry teacher, 최대25oracle, terminal-only materialization 정책이며, 이득에 따른 계산 연장이나 REFIT4의 단계별 native target/optimizer/reference reset은 구현하지 않았다.

고정 teacher-forced 입력과 upstream, P/M을 유지하는 single L4에서는 K와 B가 변하지 않는다. 이 조건의 반복 write는 W_entry+X1 B+X2 B=W_entry+(X1+X2)B로 표현된다. 따라서 terminal 물리 write가 한 번이라는 이유만으로 반복 write가 도달할 수 있는 endpoint를 배제했다고 볼 수 없다. 중요한 차이는 fresh target fitting·regularizer/teacher 기준·update rule·예산 배분이며, 그 신호를 좋은 trajectory로 연결하는 정책은 V1에 충분히 구체화되지 않았다.

## 1. 종합 판정

**이번 실행은 single-layer actual-write trajectory의 구현과 실행을 보여준다. 그러나 현재 설정에서 N4보다 편집·보존 trade-off가 좋거나, 적은 fresh 계산으로 유리한 종료점을 찾았다는 증거는 없다.**

판정은 세 부분으로 나뉜다.

1. **구현:** single L4, 실제 write와 동등한 all-token cache, 고정 entry teacher, native key binding, terminal actual FP32 write, history 한 번 갱신이 구현되어 있다. 공개 기술 검증과 source에서 성능 열세를 설명할 뚜렷한 구현 오류는 찾지 못했다.
2. **성능:** N4 대비 RS +0.20%p, PS −2.95%p, NS −8.42%p다. 동시에 P 정답과 N 원래 정답의 평균 NLL 및 teacher-forced strict 정확도는 개선됐다. 따라서 결과의 핵심은 **정답 강화와 문맥별 선택성 보존의 불일치**다.
3. **종료·효율:** 10개 batch 모두 25 oracle에서 RESOURCE_STOP이다. 초기 overshoot와 rejected full forward/backward가 비용을 크게 소비했다. optimal endpoint나 계산 절약을 입증하지 못했다.

사용자가 제시한 MPES·L4-only·REFIT4 관측은 여전히 trajectory와 endpoint를 연구할 동기를 제공한다. 이번 한 설정은 그 동기를 부정하지 않지만, 우수한 trade-off의 method evidence로 추가할 수도 없다. **현재 결과에 맞게 solver의 비용 낭비, 목적함수와 locality의 연결, 추가 진행 및 종료 정책을 각각 검증해야 한다.**

## 2. 검토 대상과 독립 확인 범위

### 2.1 CPU prototype과 실제 실행을 구별한다

이번 리뷰는 이전 첨부의 CPU toy 결과가 아니라 SH2의 실제 Llama W0→SEQ1000 완료 실행을 대상으로 한다.

- 실제 실행 HEAD: `5d149fec254a7a53b6d91790d186880f248676e6`.
- 실행 tree: `0202db9aee30ba5103c52e21d0647c66e1428567`.
- input lock SHA256: `79e69eee91a7c60e57351f9253c5ee6768dbea4557c71f8957f3b4020017d911`.
- 완료 package snapshot: `a178bf4c`, 분석 실행 HEAD: `83913514eabc9a7325b032a3f551f52eff792418`.
- 신규 MAIN: 1 chain, B100×10, unique/attempted requests 1,000개.
- 신규 same-objective Adam, barrier, 다른 λ, 다른 order 실험: 0.
- N4: 동일 case/prompt/target/order를 갖는 과거 job38997 관측 재사용.

고정 설정은 Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, FP32/eager, seed20260907, `model.layers.4.mlp.down_proj.weight` 하나, λ_write=1, λ_flow=1, β=.0625, barrier off, η_initial=η_max=1, max_oracle_calls=25다. λ_flow=1은 미튜닝 초기값이다.

### 2.2 독립 검산과 SH 보고를 구별한다

이번 리뷰에서 직접 확인한 것은 다음과 같다.

- package에 열거된 17개 파일의 SHA256·크기.
- 실제 execution과 snapshot 사이의 runtime/native_binding/llama_adapter/technical/durable/flow_core/transaction 7개 파일 byte 동일성.
- metrics/paired 산술과 identity, node/batch accounting·trajectory, compute 합계.
- source와 명세의 대응, 기술 receipt와 결과 보고의 범위.

[재검산 스크립트](2026-09-16-single-layer-zflow-seq1000-review/recompute.py)는 총 **420개 검사**를 통과했고, 결과는 [derived-summary.json](2026-09-16-single-layer-zflow-seq1000-review/derived-summary.json)에 저장했다. 검사 개수는 독립 실험 수가 아니다.

현재 리뷰 호스트에는 MAIN raw/checkpoint와 technical calibration 원본 JSON이 없다. 따라서 모델을 새로 실행하거나 item-level raw를 재집계하지 않았다. “모든 checkpoint를 CPU 검증했다”, “별도 process resume이 동일했다”는 아래 설명은 SH가 공개한 검증 결과이며, 이번 리뷰에서 직접 checkpoint를 다시 읽은 결과로 표현하지 않는다.

N4와 SL의 관측 identity/order 및 token 분모는 일치한다. 다만 N4의 cudnn TF32=True, SL=False와 host 차이가 있고, historical RNG도 완전히 보존되지 않았다. 비교표를 무효화할 근거는 없지만 bitwise-equivalent 실행 또는 optimizer 하나의 순수 인과 효과로 부르지 않는다.

## 3. 편집 성능: ranking과 절대 정답 품질이 갈라졌다

### 3.1 W10 동일 분모 비교

| 지표 | N4 | SL-ZFlow | 차이 |
| --- | ---: | ---: | ---: |
| Rewrite 선호 RS | 998/1000, 99.80% | 1000/1000, 100.00% | +0.20%p |
| Paraphrase 선호 PS | 1943/2000, 97.15% | 1884/2000, 94.20% | **−2.95%p** |
| Neighborhood 선호 NS | 8072/10000, 80.72% | 7230/10000, 72.30% | **−8.42%p** |
| Rewrite new TF-strict | 995/1000, 99.50% | 995/1000, 99.50% | 동일 |
| Paraphrase new TF-strict | 1347/2000, 67.35% | 1447/2000, 72.35% | **+5.00%p** |
| Neighborhood true TF-strict | 1813/10000, 18.13% | 2406/10000, 24.06% | **+5.93%p** |

RS/PS는 mean-token NLL(new)<NLL(true), NS는 반대이며 tie는 실패다. TF-strict는 teacher forcing에서 모든 target token이 top-1인지를 검사한다. 자유 생성 정확도와 같지 않다. 근거: `metrics.csv` 35–37, 41–43행.

RS 두 문항 개선은 확인된다. 그러나 이를 PS 순손실59개와 NS 순손실842개를 상쇄하는 종합 우위로 해석할 근거는 없다. 반대로 “정답 출력 전체가 악화했다”는 결론도 TF-strict와 NLL에 맞지 않는다.

- 두 P 문항이 모두 성공한 요청: 954→907/1000.
- 열 N 문항이 모두 성공한 요청: 413→301/1000.
- N 문맥에서 competing-new TF-strict도 246→667/10000으로 늘었다.

### 3.2 무엇이 강해졌는가

NLL은 낮을수록 해당 답에 더 높은 확률을 부여한다.

| 평가 | 답변 | N4 평균 NLL | SL 평균 NLL | SL−N4 |
| --- | --- | ---: | ---: | ---: |
| R | 원하는 new | .031905 | .033733 | +.001829 |
| R | 경쟁 true | 14.528767 | 12.319518 | −2.209249 |
| P | 원하는 new | 1.543660 | 1.281628 | **−.262032** |
| P | 경쟁 true | 10.181154 | 8.746427 | −1.434727 |
| N | 원하는 true | 5.302156 | 4.693866 | **−.608290** |
| N | 경쟁 new | 9.272972 | 7.621593 | −1.651380 |

P에서는 new가 좋아졌지만 true가 더 크게 강해졌다. N에서는 true가 좋아졌지만 competing-new가 더 크게 강해졌다. 성공 방향 margin은 P에서 `true−new`, N에서 `new−true`다.

- P margin 평균: 8.637494→7.464799, −1.172695.
- N margin 평균: 3.970816→2.927727, −1.043089.
- P margin p05: +1.293938→−.488660.
- N margin p05: −3.856557→−6.099108.

**따라서 이번 locality 문제를 “원래 답의 평균 NLL이 N4보다 높아졌다”로 설명하면 잘못이다.** 문맥별 경쟁 답변과의 구분이 약해졌다. 위 분해는 전체 문항 평균이며, 실패한 각 문항의 인과 기여율을 알아낸 것은 아니다. 평균 NLL 감소를 산술평균 확률 증가와 동일시하지 않는다.

또한 R new NLL은 N4가 평균·median·tail에서 더 낮다. “SL이 canonical target confidence를 N4보다 더 끝까지 밀었기 때문에 NS가 나빠졌다”는 설명은 이 비교로 지지되지 않는다.

### 3.3 평균 개선이 itemwise 보존을 뜻하지 않는다

동일 문항의 desired NLL 변화량 `SL−N4`에서 양수는 손상이다.

| Desired NLL 변화 | 평균 | p95 | p99 |
| --- | ---: | ---: | ---: |
| R new | +.001829 | +.042972 | +.439563 |
| P new | −.262032 | **+3.264078** | **+6.253232** |
| N true | −.608290 | **+4.696822** | **+9.007049** |

N true 평균이 개선됐어도 일부 문항은 크게 나빠졌다. Endpoint NLL p99와 paired ΔNLL p99는 다른 통계다. 예를 들어 N true endpoint p99는 15.139921→15.323682지만 paired 변화 p99는 +9.007049다.

| N4 W10→SL W10 | 유지 성공 | 성공→실패 | 실패→성공 | 둘 다 실패 |
| --- | ---: | ---: | ---: | ---: |
| R /1000 | 998 | 0 | 2 | 0 |
| P /2000 | 1857 | **86** | 27 | 30 |
| N /10000 | 6820 | **1252** | 410 | 1518 |

N4에서 성공했던 N 문항 중 15.51%가 SL에서 실패한다. NS가 모든 문항에서 균일하게 조금 낮아진 형태로 읽으면 안 된다. 근거: `paired.csv` 20–22행.

## 4. 순차 편집: locality 저하가 누적된다

### 4.1 자기 batch 직후와 W10

| 지표 | 자기 batch 직후 합계 | W10 | lost/gained |
| --- | ---: | ---: | ---: |
| R | 1000/1000 | 1000/1000 | 0/0 |
| P | 1882/2000, 94.10% | 1884/2000, 94.20% | 18/20 |
| N | 7637/10000, 76.37% | 7230/10000, 72.30% | **660/253** |

P는 자기 write 직후부터 모은 값이94.10%다. 최종 P 열세를 모두 이후 forgetting으로 설명할 수 없다. N은 이후 추가로 −4.07%p, 순손실407개가 생긴다.

단 at-write는 서로 다른 열 모델의 관측을 합친 것이다. 각 write 전 동일 entry의 관측이 없으므로 이것을 해당 write의 즉시 손상량이나 W0 대비 손상량으로 바꾸지 않는다.

### 4.2 동일 first500을 W5와 W10에 고정

| First500 | W5 | W10 | lost/gained |
| --- | ---: | ---: | ---: |
| R /500 | 500 | 500 | 0/0 |
| P /1000 | 951, 95.10% | 949, 94.90% | 10/8 |
| N /5000 | 3801, 76.02% | 3556, 71.12% | **400/155** |

동일 cohort의 NS가 **−4.90%p**다. N true NLL 평균은4.629172→4.911464, competing-new는8.140362→7.597248로 움직여 양쪽 모두 margin을 악화시킨다. N true paired ΔNLL p99는+8.970468이다.

이 비교는 at-write pooling 문제를 피한다. 다만 같은 cohort의 N4 W5→W10 결과가 package에 없으므로 N4보다 손상이 누적되는 속도가 더 빠르다고 직접 비교할 수는 없다.

### 4.3 명시적 superseded 한 건으로 주열세를 설명할 수 없다

Input-only 분류에서 active999, later-batch different-target superseded 후보1, within-batch conflict0, unresolved0이다. Active-only N4→SL에서도 P는1943→1882/1998, **−3.0531%p**, N은8065→7227/9990, **−8.3884%p**다. Superseded 한 건을 제외해도 결론이 유지된다. 이 분류는 exact identity 규칙이며 의미적 모순을 완전히 판별한 것은 아니다.

## 5. 실제 trajectory: target fitting 이후 비용을 줄인다

이번 목적함수는 batch entry를 기준으로

\[
W(X)=W_e+XB,\qquad
F(X)=L_{edit}(X)+.0625\,KL(p_X\Vert p_e)+C(X),
\]

\[
C(X)=\tfrac12\operatorname{tr}(XSX^T),\qquad
S=\frac{BM_eB^T+BB^T}{m}
\]

다. Native 비대칭 시스템을 한 번 풀어 B를 만들고, H=S+εI에 대한 IMEX proposal과 line search를 수행한다. 이는 native Adam z의 동일 경로에서 중간 checkpoint만 고르는 실험과 다르다. **실제 write 좌표 X에서 목적함수와 경로를 함께 바꾼 실험이다.**

### 5.1 관측한 accepted 경로

- 154 accepted state에서 F는 모두 감소했다.
- 모든 batch에서 oracle11–13 사이에 accepted edit NLL<.1에 도달했다.
- Terminal 평균 edit NLL=.0143396, essence KL=.266744, L=.0310111, C=.401684, F=.432695.
- Terminal F 중 C 비중은86.3–96.7%다.
- Accepted state 중127회에서 C가 감소했다.
- 22회는 edit NLL이 증가하면서 F가 감소했다.
- 10개 중7개 batch는 terminal보다 이전 accepted node에서 edit NLL이 더 낮았다.
- Accepted 경로의 최대 C와 비교하면 terminal C는58.4–85.3% 낮다.

따라서 흐름은 **초기에 target을 맞추고, 이후 비용을 줄이며 일부 confidence를 내주는 정규화 최적화**로 해석할 수 있다. 단순히 target confidence만 끝까지 높인 실행은 아니다. 이 부분은 edit 이득과 write 비용을 함께 고려한다는 설계의 작동 증거다.

그러나 training CE와 C만으로 어떤 node의 PS·NS가 가장 좋은지는 알 수 없다. 비용이 F의 대부분이라는 사실도 λ가 과도하다는 증명은 아니다. 필요한 것은 별도 trade-off 비교다.

### 5.2 Optimal endpoint는 찾지 못했다

| 항목 | 결과 |
| --- | ---: |
| Initial oracle | 10 |
| Accepted candidate | 154 |
| Rejected candidate | 86 |
| 전체 oracle | 250 |
| FIRST_ORDER_STATIONARY | 0/10 batches |
| RESOURCE_STOP | **10/10 batches** |

각 batch가25회를 모두 썼다. B002는 oracle22, B008은oracle23의 accepted state가 최종 저장되고 이후 후보는 reject됐다. 나머지8개는oracle25가 accepted였다. 최종 state는 이 실행에서 관측한 accepted F 중 최저지만, 전체 공간의 optimum이나 knowledge-editing metric의 최선은 아니다.

초기 residual r0는53.14–117.27, threshold는 `1e-9+1e-5*r0`, 약.000531–.001173이다. 공개 CSV의 `stationarity_before`는 **후보를 만들기 전 state**의 값이고, `gradient_norm`은 **∇L의 Euclidean norm**이다. 둘 다 일반적으로 terminal residual과 같지 않다.

마지막 query가 reject인 두 batch만 terminal residual을 정확히 복원할 수 있다.

| Batch | terminal residual | threshold 대비 |
| --- | ---: | ---: |
| B002 | 5.378267 | 약5443배 |
| B008 | 6.076682 | 약5681배 |

나머지8개 terminal residual은 runtime 원본에는 있지만 published batch export에 없다. 모든 batch가 roundoff 부근에서 사실상 수렴했다는 주장을 뒷받침하지 않는다. 마지막 accepted F 감소량도.00544–.22473으로 남아 있다.

ODE 형식 자체가 유한 계산 내 optimal endpoint나 낮은 NFE를 보장하지 않는다. 현재 step controller는 objective descent를 검사하며, 지식 편집 품질의 최적 시점을 직접 측정하지 않는다. 설령 FIRST_ORDER_STATIONARY에 도달해도 reduced-space 목적함수의 1차 조건이지 PS·NS 최적성 인증이 아니다.

## 6. Solver에서 가장 명확한 문제: 초기 step scale과 rejected oracle

### 6.1 모든 batch가 η=1에서 overshoot한다

| Batch | 첫 accept 전 reject | 첫 accepted η | 첫 trial C / 초기 F |
| --- | ---: | ---: | ---: |
| B001 | 5 | .03125 | 31.7배 |
| B002 | 5 | .03125 | 129.4배 |
| B003 | 6 | .015625 | 172.9배 |
| B004 | 6 | .015625 | 195.9배 |
| B005 | 6 | .015625 | 168.9배 |
| B006 | 6 | .015625 | 193.3배 |
| B007 | 6 | .015625 | 201.6배 |
| B008 | 6 | .015625 | 176.2배 |
| B009 | 6 | .015625 | 204.6배 |
| B010 | 6 | .015625 | 199.2배 |

처음 η=1의 C는352.94–1719.06인데 초기 F는7.875–11.124다. 초기 backtracking만58회, 전체 oracle의23.2%, reject의67.4%다. 남은28회 reject는 후반에 발생했다.

두 번 clean accept 후η×1.5, reject 후÷2인 회복 정책도 재차 큰 후보를 만든다. B002는η=.17798까지 회복한 뒤 마지막.26697/.13348/.06674가 모두 reject됐다.

H는 write geometry의 metric이며 nonlinear language-model loss curvature를 추정한 Hessian이 아니다. 따라서 η=1이 자연스럽거나 보편적으로 적절한 scale이라는 근거는 없다. 이번 trajectory는 이 초기값이 맞지 않았다는 직접 증거다.

86개 reject는 전부 F_trial>F_before이고,154개 accepted의 roundoff_limited는 모두0이다. Barrier도 off다. 이번 문제를 barrier 또는 tiny-budget KKT·FP32 roundoff guard 탓으로 돌릴 근거는 없다. Rejected row의 roundoff_limited=1은 감소량이 음수여도 설정되는 진단 flag이며, 그 flag가 reject의 원인이었다는 뜻이 아니다.

### 6.2 61개 후보는 모델 호출 전에 기각할 수 있었다

이번 설정에서 CE와 KL은 수학적으로 비음수이고 λ_flow=1이므로

\[
F(Y)=L(Y)+C(Y)\ge C(Y).
\]

따라서 C(Y)가 이미 현재 acceptance 상한을 넘으면 새 suffix forward/backward가 필요 없다. 구현에서는 Armijo 감소항, FP32 rounding, KL 수치오차에 대한 보수적 여유를 함께 반영해야 한다.

공개 경로에서 **86개 rejected 중61개가 C_trial>F_before**이며, 현재 FP32 rounding과 추가1e-6 여유를 넣어도61개가 유지된다. 최소 차이도.0145859다. 이61개는 초기49개와 후반12개다.

- 해당 full oracle 시간: **2225.89초=37.10분**.
- 전체 oracle 시간의 **24.40%**.
- 현재 core는 이 사전 검사를 하지 않고 모든 trial에 full F+B를 호출한다.

이는 저장된 동일 경로의 후보를 동일하게 처리한다는 조건에서 사전에 기각할 수 있었던 작업량이다. 이를 생략하고25 full-oracle까지 더 진행하면 경로와 endpoint가 달라지므로 **검증된 end-to-end speedup 또는 동일 성능 보장치가 아니다.**

이 조치는 stale gradient 근사와 다르다. 수용 가능성이 없는 후보를 비용의 수학적 하계로 제거하는 것이다. Accepted candidate의 fresh gradient와 기존 rejected candidate 평가 규칙을 바꾸는 작업은 별도 계약 변경으로 기록해야 한다.

## 7. 계산 비용: single-layer cache의 장점은 구현됐지만 suffix가 지배한다

| 구간 | 합계 | Batch 총시간 대비 |
| --- | ---: | ---: |
| 준비 | 281.68초, 4.69분 | 2.52% |
| Flow | **9123.87초, 152.06분** | **81.69%** |
| Terminal 검증·commit | 662.24초, 11.04분 | 5.93% |
| 평가 | 1056.11초, 17.60분 | 9.46% |
| Batch 총시간 | 11169.40초, 186.16분 | 100% 기준 |

표의 개별 구간과 batch 총시간 사이에는 소량의 기타 bookkeeping 시간이 있다. 중첩 timer를 중복 합산하지 않았다. MAIN process wall은11181.09초,3.106시간이다.

| Oracle 종류 | 횟수 | 시간 |
| --- | ---: | ---: |
| Initial | 10 | 364.18초 |
| Accepted | 154 | 5620.33초 |
| Rejected | **86** | **3138.19초,52.30분** |
| 전체 | 250 | 9122.71초 |

Rejected 계산이 oracle 시간의34.40%다. Reject의 backward만1494.39초,24.91분이다. 다만 forward-only screening을 실제로 도입하면 accepted graph 유지·재forward·메모리 비용이 달라지므로 이 숫자를 그대로 순절약량으로 쓰지 않는다. 우선순위는 위 cheap algebraic check다.

실제 cache는 각 batch에서 L0..L4와 L4 down_proj 입력을 고정하고 모든 token의 affine response를 저장한다. 이후 **L5..31의27개 block은 매 oracle마다 fresh F+B**한다. Batch100의 edit600+essence100 sequences를 microbatch2로 처리하므로 whole-batch oracle 하나가350 suffix microbatches다.

- Prefix forward microbatch: 전체3500회.
- Suffix forward microbatch: **87500회**.
- Suffix backward microbatch: **87500회**.
- Suffix valid token 누계:2,701,475; padded token 누계:3,026,100. 같은 입력을25회 평가한 계산량이며 독립 데이터 수가 아니다.
- Native keys 준비184.72초, geometry9.37초, prefix+teacher 준비86.14초.

Single-layer의 구조적 이득은 분명히 구현됐다. 하지만 L4가 앞쪽이므로 모델의 상당 부분은 매번 다시 계산된다. **현재 병목은 작은 geometry solve나 cache 전송보다 반복 suffix와 reject다.** Fresh “z compute”를 논의할 때 본 실행의25 whole-batch F+B와 native Adam의 요청별 max25 forward/24 backward·early-stop을 같은25step으로 비교하면 안 된다.

추가 비용:

- 기술 실패 job48294:31 GPU-sec, 기술 성공48297:1064 GPU-sec, MAIN48303:11191 GPU-sec. 신규 할당 합계12286 GPU-sec.
- N4 historical process3654.88초. 관측된 전체 process 시간은 SL이 더 길지만 observer 일정·host/backend·계측 구간이 달라 pure optimizer 속도 비율을 식별하지 않는다.
- Torch process 누적 peak allocated33.764GiB, reserved38.283GiB. Batch마다 reset한 값이 아니다.
- Checkpoint10개 합계10,701,640,900bytes, 약10.70GB. Pure filesystem I/O는 별도로 계측되지 않았다. Commit에서 parity·cost·prepare를 뺀 잔여를 pure I/O로 부르지 않는다.

## 8. Method fidelity: 무엇이 검증됐고 무엇은 아닌가

### 8.1 명세와 맞는 주요 부분

- Native N은 비대칭을 유지하고 LU로 푼다. S만 대칭화한다.
- Native key는 clean1/2·generated각1/10, edit loss는6contexts각1/6이다. 두 평균을 혼동하지 않았다.
- Edit loss token weight는1/(m·6·T), essence weight는1/m이다.
- KL은 fixed batch-entry teacher에 대한 KL(current||entry), β=.0625다.
- All-token cache로 H(X)=H_entry+A X^T를 구성하고 nonlinear suffix를 새로 계산한다.
- Native endpoint warm start나 j_native 의존성이 없다.
- 실제 FP32 Parameter candidate를 적용한 terminal full decoder와 cache 경로를 비교하고 entry bytes를 복원한다.
- History는 terminal durable publication에서 한 번 갱신한다.

### 8.2 정합성 검사에서 얻은 근거

공개 MAIN10개 terminal 결과에서:

- 최대 logit 절대오차1.0681e-4, sealed 허용값2.2125e-4.
- 실제 저장 Δ cost와 predicted C의 상대오차 최대4.7988e-5, sealed 허용값3.8832e-4.

Cached terminal과 실제 physical write가 크게 달라져 NS가8.42%p 하락했다는 설명은 이 근거와 잘 맞지 않는다.

SH는10개 saved endpoint의 W/M 및 chain hash, 실제 W 차이의 cost, M_prev+CPU FP32 K K^T 한 번의 결과를 검증했다. 기술 checkpoint의 별도 process resume에서 next-entry logits가 exact same, max-abs0이었다고 보고했다.

단 dense X-gradient parity는 첫2개 요청의14sequences/7microbatches, 초기 부근σ=.001/.002 probe 범위다. 전체 B100 terminal 검증은 logits/NLL/KL이며 terminal gradient 전수 검증은 아니다. Cache와 full decoder의 parity는 loss/head helper를 공유하므로 label/tokenizer의 독립 타당성 검증까지 대신하지 않는다. 그것은 native source binding 검사에 의존한다.

### 8.3 기술 PASS의 의미

Barrier off인 MAIN의 actual-cost PASS는 **optimizer의 비용과 실제 저장 write 비용이 일치한다**는 뜻이다. 비용 절대상한이나 NS 보존을 통과했다는 뜻이 아니다. 따라서 이 PASS와 NS 열세는 모순되지 않는다.

기술 R1은 optional tokenizer metadata attribute 읽기 실패로 oracle/edit/checkpoint0에서 중단됐다. Postrun analysis R1은 pinned HF blob symlink 처리 문제다. 이후 수정과 threshold binding 강화는 분석·보고 수정이며, actual runtime/core를 사후 변경하거나 tolerance를 완화했다는 근거는 찾지 못했다.

## 9. 왜 C 감소를 locality 개선이라고 부를 수 없는가

현재 C는 실제 write에 대해

\[
C(\Delta W)=\frac{\operatorname{tr}(\Delta W M_e\Delta W^T)+\|\Delta W\|_F^2}{2m}
\]

를 측정하는 geometric surrogate다. 여기서 M_e는 batch entry의 history다. 이는 해당 key 방향의 선형 출력 변화와 weight 크기에 대한 비용이지, 평가 neighborhood에서의 NLL 또는 new/true margin이 아니다. Nonlinear suffix의 민감도, 평가 문맥 차이, 경쟁 답변 확률 이동을 전부 나타내지 않는다.

또한 비용은 매 batch의 W_entry를 기준으로 하며 M_e도 달라진다. Essence teacher 역시 그 batch entry다. 따라서 작은 batch별 C 또는 KL은 W0로부터의 누적 변화가 작다는 보장이 아니다. 서로 다른 batch C 감소 추세를 직접 “long-term locality가 개선된다”로 읽으면 안 된다.

이번 package에는 N4의 동일 entry/M/정의에서 비교한 actual C 표가 없다. SL의 C가 줄었다는 사실만으로 N4보다 모델을 덜 변경했다고 결론내릴 수 없다. NS가 낮다는 사실도 모든 pretrained capability가 더 나빠졌다는 증거는 아니다. General/MMLU/wiki 등은 이번 package에 없다.

현재의 가장 적절한 가설은 **C와 제한된 essence KL이 실제 문맥별 선택성 보존을 충분히 대리하지 못했을 가능성**이다. 인과 원인 확정은 아니며 λ/optimizer/barrier/endpoint를 맞춘 비교가 필요하다.

## 10. 사용자 motivation에 대한 판정

| 주장 | 이번 실행이 제공한 근거 |
| --- | --- |
| Single layer에서 actual write 경로를 직접 최적화할 수 있다 | 실제 구현·terminal parity·완료 실행으로 뒷받침 |
| Confidence를 계속 높이는 것과 좋은 edit endpoint는 같지 않을 수 있다 | 비용 감소 중 일부 confidence를 내주는 실제 경로는 관측. 중간 node의 평가 우위는 미측정 |
| Edit 이득과 변경 비용을 함께 반영할 수 있다 | F 구성과22개 edit-NLL-increasing accepted step으로 작동 확인 |
| 그 결과 N4보다 locality가 좋아진다 | 현재 λ=1/off/25 설정에서는 NS가열세 |
| ODE가 적은 fresh compute로 종료점을 찾는다 | 10/10 RESOURCE_STOP,250oracle이므로 입증되지 않음 |
| Barrier가 trajectory를 유리하게 안내한다 | NOT_RUN |
| ODE integrator 자체가 Adam보다 좋다 | Same-objective Adam이없어 식별 불가 |

MPES·L4-only·REFIT4의 관측은 후속 검증 동기로 유지할 수 있다. 그러나 이번 실험의 actual-write joint optimization과 native z early stopping은 다른 intervention이다. 향후 논문에서는 둘을 동일한 주장으로 묶지 말고 연결되는 가설로 설명하는 편이 정확하다.

## 11. 다음 검증의 우선순위

이번 리뷰에서 아래 실험을 새로 실행하지 않았다. 후속 설계의 우선순위다.

### 11.1 우선 공개 reporting을 보완한다

1. Raw flow에 이미 있는 terminal residual·threshold·stop ratio·terminal oracle index를 batch export에 추가한다.
2. Cheap algebraic reject와 full-oracle reject, suffix forward/backward, accepted update를 별도로 계수한다.
3. R/P/N ranking과 desired/competing NLL·strict·paired tail을 함께 주표에 넣는다.
4. Actual C를 보호항과 weight-norm항으로 분해하고, 비교 시에는 동일 entry/M 기준을 명시한다.

이 단계는 기존 raw가 있는 소유 호스트에서 모델 재실행 없이 가능한 부분부터 처리할 수 있다.

### 11.2 계산 낭비부터 분리해 검증한다

- λ·B·L·C를 고정한 상태에서 cost lower-bound 사전 기각을 추가한다.
- 초기 step scale을 현재 관측에 맞춰 재설계한다. .015625/.03125는 진단 출발점이지 보편적 최적값이 아니다. 이전 batch scale을 재사용할 때도 새 entry 검증이 필요하다.
- 현재 two-accept×1.5 회복이 만드는 후반 reject를 별도로 측정한다.
- 같은25 full-oracle budget과 같은 wall-time budget을 구별해 비교한다. Algebraic reject는 full-oracle budget을 차감하지 않되 별도 proposal 수와 wall cap이 필요하다.

목표는 solver 낭비가 줄어드는지 확인하는 것이다. 더 많은 accepted step이 PS·NS를 개선할 것이라고 미리 가정하지 않는다.

### 11.3 “중간 endpoint가 더 낫다”를 직접 검사한다

Accepted node별 edit loss, essence KL, C와 별도 개발용 R/P/N 관측을 연결해 경로상의 trade-off를 본다. 현재 node.csv에는 scalar만 있고 durable checkpoint는 terminal이므로, 중간 X가 따로 보존되지 않았다면 현재 표만으로 중간 품질을 복원할 수 없다. 작은 재현 실행에서 필요한 X만 저장하거나 검증된 replay가 필요하다.

최소한 첫 충분한 edit 지점, 비용이 줄어드는 중간 지점, terminal을 비교한다. Final test R/P/N을 보고 endpoint 또는 λ를 고른 뒤 같은 test를 최종 성적으로 보고하지 않는다. 개발 split에서 stopping rule을 정하고 고정된 held-out chain에서 평가해야 한다.

### 11.4 Optimizer와 목적함수 효과를 구별한다

| 비교 | 고정할 것 | 확인할 질문 |
| --- | --- | --- |
| 현재 flow vs 개선 step controller | 동일 X0/B/L/C/λ/entry | Overshoot·reject 비용이 줄어드는가 |
| 개선 flow vs actual-write Adam | 동일 X0/B/L/C/λ/entry, F/B·wall 예산 | Integrator/metric의 독자적 이점이 있는가 |
| λ trade-off | 동일 optimizer·entry·data | C와 실제 PS/NS 사이에 유용한 관계가 있는가 |
| Barrier off vs fixed/동적 barrier | 동일 loss·solver, 개발용으로 정한 비용 설정 | 제약이 유리한 경로·종료점을 만드는가 |
| Native Adam 중간 z write vs native endpoint write | 동일 native trajectory·write 규칙 | 원래 motivation인 z early stopping이 성립하는가 |

모든 조합을 즉시 긴 chain으로 실행할 필요는 없다. 작은 공통 entry 진단으로 원인을 좁힌 뒤, 설정을 고정해 SEQ1000에서 재검증하는 것이 정보 효율적이다. 순서에 강건한 장기 보존 우위를 주장하려면 여러 순서·seed와 더 긴 chain에서의 재현이 필요하다. P/N prompt 수가 각각2000/10000이어도1000개의 request에 묶여 있으므로 독립 표본으로 간주한 신뢰구간을 만들지 않는다.

## 12. 근거 경로와 재현

Pinned snapshot root: `local/reviews/sl-zflow-seq1000-review-2026-09-16/source/`.

- [SH 완료 보고](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/diagnostic-report-ko.md).
- [Metrics](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/aggregates/metrics.csv), [paired](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/aggregates/paired.csv).
- [Node](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/aggregates/node.csv), [batch](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/aggregates/batch.csv), [compute](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/aggregates/compute.csv).
- [Flow core](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/project/run_scripts/single_layer_zflow/flow_core.py), [adapter](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/project/run_scripts/single_layer_zflow/llama_adapter.py), [runner](../../local/reviews/sl-zflow-seq1000-review-2026-09-16/source/project/run_scripts/single_layer_zflow/runner.py).

집계 독립 검산:

```bash
python3 audits/global/2026-09-16-single-layer-zflow-seq1000-review/recompute.py
```

**최종 권고:** 현재 설정을 성능·효율 우위로 승격하지 않는다. Actual-write 구현과 경로가 작동한 근거로 보존하고, cheap cost rejection·step scale을 우선 정리한 뒤 endpoint별 품질과 same-objective Adam 비교로 원래 가설을 직접 검증한다.

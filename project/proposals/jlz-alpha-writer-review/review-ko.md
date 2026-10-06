# PRICE + AlphaEdit writer: 두 모델 × 세 arm 검토

2026-10-07 KST. 검토 기준 `fe7c92f6cb85f0659b01f5e0c8468c144a4679d5`. 상태: **실험 제안 검토 완료 / 미구현 / 미전달 / 미제출**. 이 문서는 기존에 승인된 MEMIT 계열 soft-ridge writer의 여섯 run을 변경하지 않는다.

## 판단과 기대효과

검토할 가치가 있다. 주가설은 **PRICE의 공유 배분 정책과 실현 응답 목적함수를 유지하면서, AlphaEdit의 쓰기 공간 제약으로 초기 paraphrase 획득과 누적 보존의 관계를 개선할 수 있는가**다. AlphaEdit를 exact writer처럼 요청량을 강제로 맞추는 수단으로 해석하지 않는다. 원 논문의 핵심은 보존 key의 null-space 방향으로 update를 제한하는 것이며, history와 L2 정규화가 있는 soft solve다.

현재 근거는 낙관적인 NS 개선을 단정하기 어렵게 한다. 최신 게시 W20은 다음과 같다.

| 기존 Llama 결과 | PS | NS | 비교 성격 |
|---|---:|---:|---|
| 현재 PRICE, 이전 source 0415 | 91.275 | 85.115 | 2k raw 검산 |
| AlphaEdit | 93.225 | 68.590 | 역사 참고; 다른 runtime, L2=10 |
| AlphaEdit-BLUE | 97.150 | 76.585 | 역사 참고; L4+L8, L2=1 |

이는 writer 단독의 인과 비교가 아니다. 가설의 초점은 기존 AlphaEdit의 높은 PS를 그대로 가져온다는 것이 아니라, **현재 PRICE의 높은 NS와 retention을 유지하면서 PS 획득을 개선할 여지가 있는지**다. 기존 PRICE의 편집 직후 PS는 3682/4000=92.05%, W20 PS는 3651/4000=91.275%다. acquisition과 후속 망각을 분리해야 한다.

CAP075는 writer 변경의 기준, CAP100은 보호된 쓰기 공간에서 추가 요청 budget이 도움이 되는지, FREE100은 그 조건에서 층별 cap까지 풀었을 때의 이득·손실을 본다. CAP100의 가설이 가장 직접적이다. FREE100은 성능 상한을 시험하는 arm이지만 큰 write·history 간섭 위험도 가장 크다. 효과 크기나 우승 arm은 아직 예측하지 않는다.

## 실험 범위 제안

모델은 기존 승인 run과 같은 Meta-Llama-3-8B-Instruct 및 Qwen2.5-7B-Instruct다. 같은 model/tokenizer revision, CounterFact first2000와 순서·seed, L4–L8/anchor8를 사용한다. Qwen readout27/차원과 Llama readout31/차원을 구분한다.

| AlphaEdit writer arm | beta_base | local cap | 모델별 edits |
|---|---:|---|---:|
| AE-CAP075 | 0.75 | 0.75 a | 2,000 |
| AE-CAP100 | 1.00 | 0.75 a | 2,000 |
| AE-FREE100 | 1.00 | 없음 | 2,000 |

추가 여섯 run이며, 기존 승인 여섯 run과 합치면 총 열두 model×writer×arm 비교 cell이다. 모든 cell은 독립 cold W0/H0에서 BS100×20을 진행한다. edited W/H, entry 가격, 실현량을 writer 간 재사용하지 않는다. 정확한 identity가 일치하는 cold W0 평가만 모델 내 재사용 가능하다. GPU cap 2는 기존 작업과 두 writer·두 모델 전체가 공유한다.

기본 예산·cap·최대 예산식 `max(beta_base,0.75*max(pi))`, grace12, 4단계, threshold .05, 25 evaluations/24 updates, LR .1, KL .0625, norm .5, 동일 optimizer와 tiny-block repair를 유지한다. FREE075·FLAT·REVERSE는 추가하지 않는다. toy/별도 pilot 없이 실제 B1에서 runtime 계약을 점검하고 같은 trajectory를 계속하는 설계다.

## writer 정의와 가격 연결

기호 혼동을 피하기 위해 고정 null-space projector를 `N`, writer의 얇은 우측 factor를 `Q`, batch mean key를 `K`, 누적 key covariance를 `H`로 둔다. 각 layer에서 다음을 사용한다.

```text
A_alpha = lambda_alpha I + N (H + K K^T)
Q_alpha = solve(A_alpha, N K)
DeltaW  = R Q_alpha^T
M_alpha = Q_alpha^T K
```

이는 official AlphaEdit solve의 factor 형태다. MEMIT update에 사후로 N만 곱하는 방식과 다르다. MEMIT의 `15000*C0`를 이 normal equation에 그대로 추가하면 또 다른 hybrid writer가 된다. 원래 AlphaEdit의 residual/remaining-layer divisor나 compute_z까지 도입하는 것은 이번 writer 교체 범위를 벗어난다.

제안 고정값은 저장소의 기존 `.02`-threshold projector와 **lambda_alpha=1**이다. projector 파일·모델 revision·층 index를 봉인하고 성능을 보고 threshold/L2를 조절하지 않는다. 원 논문은 1e-2 threshold를 서술하지만 이 repo의 기존 projector는 .02 자산이므로 둘을 혼동하지 않는다. 과거 AlphaEdit L2=10 run을 이번 L2=1의 동일조건 기준으로 쓰지 않는다. Qwen baseline YAML의 clamp4/LR.5/decay.001도 PRICE planner에 가져오지 않는다.

가격은 AlphaEdit own-entry에서 계산한 `M_alpha`로 갱신한다.

```text
X_lrj = (a_lr/a_lj) * M_alpha[r,j] / (1-M_alpha[j,j]), j != r
kappa_lr = RMS_j X_lrj
pi_lr = 기존 floor 후 kappa_lr / min_layer kappa_lr
```

고정 N/H에서 한 key를 제거하는 rank-one identity가 성립하므로 LOO 가격의 형태는 유지할 수 있다. 하지만 검산식은 바뀐다. `q_minus = Q_r + Q_j*M[r,j]/(1-M[j,j])`에 대해 확인할 잔차는

```text
(lambda_alpha I + N H) q_minus
  + N K_minus (K_minus^T q_minus) - N k_r
```

다. 현재 MEMIT용 `Aq + K_minus K_minus^Tq - k_r`를 재사용하면 잘못된 검산이다. `1-M_jj`가 작아질 수 있으므로 기존 denominator guard·분포 기록을 유지한다. 임의 clipping, 실현 역수 보상, exact forcing을 추가하지 않는다.

매 candidate에서 fresh upper keys와 Q를 다시 계산하고, 실제 FP32 materialized weight의 local response를 subject objective에 주입한다. 같은 Q로 raw-context response pullback을 계산한다. Mean M은 가격용, raw-context M은 pullback용이라는 구분을 유지한다. terminal에 평가한 payload를 그대로 commit하고 final rewrite mean key의 H를 한 번 갱신한다. H는 각 arm 자신의 상태이며 zero/미달 요청도 기존 계약대로 포함한다. 새 dK/dQ 또는 full builder reverse는 필요하지 않다.

## 특별히 확인할 실패 경로

1. **싸지만 잘 써지지 않는 층.** N이 현재 key를 거의 제거하면 자기 실현량과 다른 key로의 누설이 함께 작아질 수 있다. 현재 가격은 후자만 재므로 그런 층을 싸게 볼 수 있다. `||N k_r||/||k_r||`, `M_rr`, kappa/pi, price-floor 수, 실제 response를 같이 기록해야 한다. 낮은 누설만으로 좋은 allocation이라고 판정하지 않는다. 이를 이유로 실험 도중 가격이나 layer eligibility를 바꾸지 않는다.
2. **null-space가 모든 neighborhood를 보호하지는 않는다.** N은 고정 cold 통계에서 얻은 근사 projector다. 이후 upper key 변화, 이웃 key, paraphrase 방향, 누적 history에 대한 보존은 직접 평가해야 한다. H 제약도 exact constraint가 아니다. FREE100의 큰 요청이 자동으로 안전해지지 않는다.
3. **writer mismatch는 남는다.** AlphaEdit도 일반적으로 `M != I`다. 자기 응답, off-owner leakage와 context별 실현은 달라질 수 있다. 실현 응답을 목적함수에 쓰는 기존 접근을 유지해야 하며, 단순 R/v 비율을 1로 만드는 것이 목표는 아니다.
4. **수치 교정과 confound.** 새 AE 여섯 arm과 MEMIT 여섯 arm은 동일 tiny-block 교정·정의·optimizer를 공유해야 한다. 기존 source 0415의 PRICE 결과는 역사 참고다. writer별로 계산한 가격과 trajectory가 다르므로 대응 arm 비교는 writer를 포함한 전체 정책의 비교이며, 고정 R·고정 가격에서 actuator 하나만 바꾼 효과는 아니다.

## 계산량과 구현 변경점

현재 `jlz_realized_subject/geometry.py`의 MEMIT prior/ridge, `jlz_native_writer_aware/builder.py`의 ridge import, `jlz_v12r/entry.py`의 prior 생성은 writer별 backend로 연결해야 한다. commit만 AlphaEdit로 바꾸면 planner와 commit이 다시 어긋난다.

`A0_alpha=lambda_alpha I+N H`는 일반적으로 비대칭이다. 기존 Cholesky 경로를 그대로 사용하지 않는다. 같은 A0의 LU factorization을 layer/batch마다 한 번 만들고,

```text
Y = solve(A0_alpha, N K)
Q = Y (I + K^T Y)^(-1)
```

로 candidate마다 B×B solve를 수행하는 경로가 가능하다. 이 식은 저장된 N의 완전한 대칭·멱등성을 가정하지 않아도 성립한다. 기존 projector를 다른 orthogonal basis로 조용히 재구성하지 않는다. 원 A_alpha 잔차와 실제 effective payload를 검산한다. 새 dense d×d factorization이나 projector SVD를 candidate마다 반복하지 않는다. 공식 FP32 dense solve와의 bit identity가 아니라 같은 writer operator의 PRICE FP64 geometry 구현이라는 범위를 기록한다.

고정 projector만 FP32 기준 Llama 약 **3.83 GiB**, Qwen 약 **6.68 GiB**다. 이것이 모두 추가 VRAM 상주량이라는 뜻은 아니며 CPU/mmap과 layer별 전송을 사용할 수 있다. N H 구성·LU·전송·추가 buffer 비용이 있어 현재 MEMIT와 동일 실행시간이라고 예측할 수 없다. 실제 B1의 fit/solve/telemetry 비용과 peak RAM/VRAM을 분리 기록한다. V14처럼 full solve backward를 도입할 이유는 없다.

텔레메트리도 분리해야 한다. 현재 `_ideal_cost`는 solver prior A를 보존 에너지 행렬로 함께 사용한다. AlphaEdit의 비대칭 A0를 여기에 넣고 `Q_C0=total-Q_H`를 계산하면 의미가 틀린다. writer solver matrix와 공통 damage 기준 C0/H를 분리하고, `tr(DeltaW C0 DeltaW^T)`, `tr(DeltaW H DeltaW^T)`, update norm을 공통 정의로 보고한다. 기존 weighted C0 지표를 유지한다면 15000 배율을 명시한다. 이들은 진단이며 새 loss나 배분 가격 항이 아니다.

## 결과 판단과 claim

주 비교는 같은 모델의 MEMIT-CAP075 ↔ AE-CAP075, MEMIT-CAP100 ↔ AE-CAP100, MEMIT-FREE100 ↔ AE-FREE100이다. 각 writer 내부에서 CAP075→CAP100과 CAP100→FREE100을 함께 본다. 큰 budget에서 PS가 높아져도 NS/기존 cohort가 더 빨리 떨어지면 개선으로 단정하지 않는다.

필수 보고는 current/birth-cohort PS acquisition, W5/10/15/20 all-seen 및 고정 cohort PS/NS, TF-strict와 NLL, lost/gained, 가격·자기 실현·projection된 key 비율·cap/stage/spend, costs다. P/N 평가를 학습이나 controller에 넣지 않는다. 동일 requested budget은 같은 실현 강도가 아니다.

양쪽 writer에서 결과가 좋으면 **배분·실현 응답 최적화가 다른 writer에도 적용 가능하다**는 claim을 보강할 수 있다. 그러나 가격 대조군이 없는 이 여섯 추가 arm만으로 AlphaEdit 위에서 PRICE의 독립 인과 효과를 증명하거나 stock AlphaEdit보다 우월하다고 결론낼 수는 없다.

## 근거

- [AlphaEdit 원 논문, 방법식 12–14](https://arxiv.org/html/2410.02355v2#S3.SS3), [official writer 구현](https://github.com/jianghoucheng/AlphaEdit/blob/main/AlphaEdit/AlphaEdit_main.py).
- `project/run_scripts/ode_edit_motivation/alphaedit_factors.py`: genuine historical projected solve와 posthoc projection 구분.
- `project/run_scripts/fixed_z_nonuniqueness/config/alphaedit-llama3-8b.yaml`, `alphaedit-qwen2.5-7b.yaml`: 기존 projector threshold .02, L2=1과 모델별 baseline hparams.
- `audits/servers/server4/S4-M0-storage-execution-readiness.md`: 두 모델의 projector SHA/shape/size. 2026-08-22 관측이며 현재 자산·자원 상태는 이번 검토에서 새로 확인하지 않았다.
- [PRICE W20와 역사 baseline 비교](../../../experiment-reports/servers/server4/jlz-interference-priced-l1-2k/w20/report-ko.md).
- [앞서 승인된 MEMIT 계열 여섯 run 계약](../jlz-price-cap-budget-review/authorized-run-contract.json).

이번 작업은 source·문헌 검토와 설계 문서 작성이다. 모델 실행·toy·새 검증 실험·GH/SH4 메시지·Slurm 제출은 수행하지 않았다.

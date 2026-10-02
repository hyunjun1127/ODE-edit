# JLZ v9 구현과 비교 계약

본 문서는 변경 명세이며 production 코드 구현 보고가 아니다. [v8 구현 계약](../2026-10-03-jlz-native-writer-v8/implementation-ko.md)의 native mean-key writer, current-key gradient, q Adam, replay0, native history를 기반으로 한다. V8 source가 아직 구현되지 않았다면 그 항목도 함께 구현해야 한다. V7 namespace나 실행 결과를 덮어쓰지 않는다.

## 변경할 책임

| 모듈 또는 책임 | v9 계약 |
|---|---|
| Native adapter와 entry | 문장·subject·readout·norm·KL 방향 및 clean anchor를 보존한다. |
| Causal builder와 solve | Native 평균 κ, whole-B barrier, FP32 materialization, 모든 상층 κ/P의 total gradient를 유지한다. |
| Allocation | G/E 개별 root 손실을 c_l 한 항으로 교체한다. A=sum(c), B=vector_norm(c), λ=.1. G/E 자체는 telemetry에 남긴다. |
| Optimizer와 physical auxiliary | q 좌표, 24 update, terminal25, current pulse schedule/계수 유지. Past branch는 생성하지 않는다. |
| Realization recorder | M, DM, 실제 materialized weight action, context/상속 gap을 구분한다. 손실·후보 선택에 입력하지 않는다. |
| Exact probe | Fixed-K operator 비교와 full causal shadow 비교를 별도 함수로 둔다. Main writer나 optimizer 설정을 바꾸지 않는다. |
| Writer와 transaction | Ridge만 main commit/history를 갱신한다. Shadow 종료 후 W/H/RNG 및 main checkpoint identity를 검증한다. |
| Evaluator와 collector | 동일 case/token/metric의 endpoint 비교, null/unsupported 구분, W5 누적 분모를 보존한다. |

## Combined cost와 수치 의미

Reference는 `energy = tr(D @ G @ D.T) + norm(D @ (P.T @ kappa - I))**2`다. `c = sqrt(energy)/(sqrt(B)*sigma)`이며 σ²=mean(a²)다. Native norm은 별도로 유지한다. Norm0의 선택 subgradient는0이다. Arbitrary ε smoothing을 추가하지 않는다. Tiny negative roundoff만 사전 정의된 기계 오차 범위에서0으로 처리하며 값/횟수를 기록한다. 그 범위를 넘는 음수는 기술 실패다.

SPD-qualified dual에서는 F=L⁻¹κ, S=I+FᵀF, P=L⁻ᵀF S⁻¹를 사용한다. G+E=S⁻¹ 항등식으로 계산을 줄일 수 있으나, 기존 FP32 A/H의 의미를 바꾸기 위해 symmetrize/jitter를 추가하지 않는다. Dense same-A reference와 값/전체 D gradient parity를 먼저 확인한다. Policy gradient, direct κ와 solve adjoint를 포함한다. 기존 v8 solve VJP는 유지하며 새로운 비용의 adjoint로 검증한다.

Root 합 변경에 따라 λ_W/λ_E를 계속 읽어 무심코 .2를 곱하지 않는다. Config에는 `allocation_metric=ridge_optimal_value_root`, `lambda_allocation=.1`만 둔다. V8의 두 비용은 detached diagnostic 값으로만 계산한다.

## 실현량의 세 수준

1. FP64 수학 operator: M=Pᵀκ, Y64=D M=U64 κ. κᵀP를 대신 쓸 때는 symmetry qualification이 필요하다.
2. Materialized parameter action: ΔW_eff=double(W_trial32)−double(W_entry32), Y_weight=ΔW_eff κ. U64를 FP32 cast하고 entry에 더하는 roundoff를 포함한다.
3. 실제 forward에서의 own linear output difference. 동일 actual incoming key에서 `linear(W_trial,k)−linear(W_entry,k)`를 관측한다. FP32 GEMM roundoff 때문에 2번과도 bitwise 동일하지 않을 수 있다.

“Exact”의 이론식과 실제 committed 모델의 수치 오차를 구분한다. Layer profile이 down-projection 출력과 full-block output 사이에 비선형 변환을 포함한다면 별도 adapter 검증 없이 UK를 full-block δ 실현이라고 기록하지 않는다.

## 필수 telemetry

모든 candidate에서 이미 계산한 D/κ/P를 이용해 다음을 기록한다. 큰 tensor의 per-request `.item()` 동기화를 반복하지 않고 벡터로 수집한다.

- Planned `rho=norm(D_r)/a_r`, 실제 mean action의 `norm(Y_r)/a_r`, 각각의 layer share. Share의 분모0은 null이다.
- `diag(M)`, off-diagonal Frobenius norm, 요청별 self vector `D_r*M_rr`와 cross vector `sum_s!=r D_s*M_sr`, 그 norm 및 D 방향 투영.
- D_r≠0에서 `gamma=dot(D_r,Y_r)/norm(D_r)^2`, `norm_ratio=norm(Y_r)/norm(D_r)`, `orthogonal_error=norm(Y_r-gamma*D_r)/norm(D_r)`, `fit_error=norm(Y_r-D_r)/norm(D_r)`. Gamma는 음수나1초과도 허용한다. D=0이면 이 비율은 null이고, norm(Y)는 여전히 보고한다. 작은 rho는 별도 flag로 표시하며 평가에서 삭제하지 않는다.
- G/E의 원시 에너지와 g/e, combined c, native norm을 분리한다. Scalar capacity를 g/e로 역산하는 필드는 만들지 않는다.
- V8의 first-step bound, pre/post clamp, proposal radial component, physical D gradient/q gradient, current virtual NLL/KL과 pulse actual 항별 값.
- Main entry와 terminal에서 full B×B M 및 계획/실현 tensor를 별도 파일로 저장한다. 모든 candidate의 JSON에 큰 행렬을 반복 저장할 필요는 없다. Hash와 dtype/shape/owner를 결속한다.

Current pulse와 terminal의 기존 실제 forward에서 canonical·각 rewrite context의 own action, z_virtual−h_actual_pre와 method의 세 항 분해를 저장한다. **후보25의 기존 native virtual forward에서 전체 B의 모든 rewrite subject hidden을 capture**한다. 이전 후보 teacher를 대신 쓰거나 `teacher_requests=[]`로 terminal virtual capture를 생략하지 않는다. 이는 같은 pass의 hook 추가이며 별도 fit/forward를 요구하지 않는다. Context 각각의 gamma/norm/orthogonal/fit error도 산출하고 canonical과 평균 context를 분리한다. Gap norm뿐 아니라 세 벡터의 합이 전체 gap과 일치하는지, 교차 내적 때문에 cancellation이 있는지 확인한다. 추가 all-layer endpoint backward는 하지 않는다. Builder의 prefix state를 terminal hidden으로 재사용할 때 동일 candidate/input binding을 확인한다.

Terminal actual native NLL과 current‖entry KL을 반드시 저장한다. `virtual_native_kl`, `actual_native_kl`, `actual_target_kl_from_virtual`처럼 방향과 branch를 구분한다. Terminal에는 추가 Adam update나 gradient가 없으며 `gradient_measured=false`다.

## Exact probe의 수치 qualification

Fixed A/κ의 whitened X를 FP64로 계산하고 eigenspectrum, numerical rank, smallest/largest eigenvalue, rcond, requested B를 저장한다. 모든 eligible layer에 대해 SPD A와 full rank를 확인한다. Numerical rank tolerance는 `max(d_in,B)*eps64*lambda_max(X)`이고, exact 실행에는 추가로 `rcond(X)>=1e-10`을 요구한다. 이는 수치 적용 범위이며 layer 품질 gate가 아니다. 순위가 불분명하면 exact를 `numerically_unqualified`로 기록하고 native ridge는 그대로 진행한다.

Inverse를 명시 생성하지 않고 X에 대한 solve를 사용한다. FP64 `norm(Ueq@kappa-D)/max(norm(D),1e-30)<=1e-8`을 요구한다. D0은 절대 residual도 별도 확인한다. Materialized FP32 forward의 equality는 RMS(error)≤1e-5+1e-4 RMS(D)를 시작 기준으로 고정한다. 결과를 보고 tolerance를 늘리지 않는다. 기준 밖이면 `algebra_exact_but_FP32_unqualified`로 기록하고 모델 endpoint 성능을 qualified exact로 표기하지 않는다.

Singular X의 pseudoinverse는 CPU/분석의 compatibility 진단에만 사용한다. `norm(D-D@X_pinv@X)`와 key duplication을 보고하되 실제 exact endpoint로 commit하지 않는다. Compatible singular target도 이번 endpoint 구현 범위 밖임을 명시한다. Native ridge로 fallback하거나 εI를 추가해 성공으로 기록하지 않는다. Low-rank compatible exact 지원은 후속 method revision이다.

## 두 비교의 실행 순서

먼저 terminal ridge fit의 D, batch entry W/A/H, native context/RNG, actual κ를 고정한다.

**Fixed geometry probe:** 각 층에서 같은 D/κ/A로 Uridge/Ueq, mean/context action, preservation energy와 norm을 계산한다. 다른 실제 incoming key로 얻은 비교값을 같은-K 표에 섞지 않는다. 이 단계는 선형 operator 분석이며 hybrid frozen-K 모델의 RS/PS/NS를 causal 결과로 보고하지 않는다.

**Causal shadow endpoint:** 정확히 같은 D를 유지한 채 entry 모델에서 exact lower→upper builder를 새로 실행한다. Upper κ/P/X는 exact lower writes에 맞춰 새로 계산한다. 모든 층이 qualified일 때만 terminal actual NLL/KL 및 R/P/N을 관측한다. 중간 층 실패 시 exact endpoint 전체를 unsupported로 기록한다. 실패한 요청·층을 제외한 분모로 성공률을 만들지 않는다.

Probe는 no-grad이고 D 재학습, rescale, early stopping, 품질 기반 candidate 재선택이 없다. Frozen-K probe의 exact P를 causal endpoint에 재사용하지 않는다. **Ridge와 exact의 P/G/E 및 first_geometry cache를 writer kind별로 분리**한다. 첫층의 κ/A가 같아도 ridge P는 exact P가 아니다. 저장된 ridge terminal D/weights/geometry는 immutable하게 보관한다. **Frozen ridge upper K에서의 qualification 실패를 causal exact upper K에 복사하지 않는다.** Causal branch는 자기 incoming K에서 독립적으로 판정한다. 첫층은 K가 동일하므로 그 failure를 공유할 수 있다. 모든 shadow trace에 D hash, entry state hash, writer kind, geometry kind를 저장한다.

Main state에는 probe history update가 없다. Exact probe가 성공·실패·예외로 끝나도 `try/finally` rollback 후 W/H/RNG hash를 확인한다. 동일 실제 ridge weights를 main에 commit하고 H를 한 번만 갱신한다. Scope 바깥 layer의 weight 비변이도 검사한다. 공식 P/N 값은 collector에만 전달한다.

## 검증과 중단 조건

CPU algebra와 full causal finite difference를 통과한 뒤 실제 small-model/request GPU parity를 수행한다. 시작 기준은 FP64 atol1e-9/rtol1e-8, 전체 finite-difference gradient atol1e-6/rtol2e-4다. Actual LM gradient RMS(error)≤1e-6+2e-3 RMS(reference), output/component atol1e-4+rtol1e-5를 v8에서 상속한다. Selected-position head와 microbatch 변경은 동일 logical loss/gradient를 유지해야 한다.

Nonfinite, key/owner mismatch, missing eligible layer, incorrect gradient, illegal past forward, wrong commit/history/rollback은 기술 실패다. Exact branch의 solve 미지원이나 해당 scratch tensor의 nonfinite는 그 endpoint만 실패로 격리할 수 있지만, **main W/H/RNG 복구나 scope 외 weight 비변이 확인이 실패하면 본선도 진행하지 않는다.** 낮은 NS/PS, concentration, 이후 clipping, 높은 exact preservation energy 또는 fixed-budget 미수렴은 main 중단·재튜닝 조건이 아니다. CPU toy PASS를 production/GPU PASS로 복사하지 않는다.

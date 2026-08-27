# BGODE-R1 S1 Llama request000 — GH 독립 사실 분석

> **최종 판정:** 실행은 6-arm, FULL-FP32, event normalization, Official AlphaEdit adapter fidelity, atomic W0 복구를 통과한 유효한 기술 pilot이다. 그러나 최종 수학 설계가 요구한 node별 scalar exact-progress localization `rho_n`가 dynamic 실행 경로에 구현되지 않았다. 실제 action은 `beta=h*u`를 곧바로 적용했고 terminal progress residual이 arm별로 크게 달랐다. 따라서 이 결과는 `unlocalized Euler diagnostic`으로 보존하며, matched-progress 조건의 barrier 인과 효과·ODE 우위·scientific promotion을 주장하지 않는다.

- verdict: `TECHNICAL_PASS_SCIENTIFIC_HOLD_MISSING_EXACT_PROGRESS_LOCALIZATION`
- sample: Llama B1 request000, case `19795`, single request only
- raw terminal: `60261a5f5166ef645aee0550f977bbdf617d2badc9a7efc4386a51eb10605bcc`, 99755 bytes, mode 0600
- exact execution source: `7c41098cc56fbf4ad6e00a9a32bcf1948425d94e` / `edb50be2ced768833b3c4a84afc15afe6152d895`
- exact source blob SHA: `af20997eaf3c2442ffefc3a66e1f50a67fb68b5c26744052dbbeebb9bd105f23`
- final design SHA: `fd95be5f0edb4be74a8bad12aea0dbd3254b6e78a56bf3e2a562c8efa49579d9`
- scientific promotion: `false`

## 1. 가장 중요한 구현-수학 경계

최종 설계는 각 node에서 `D_n=h sum_j u_j B_j`를 만든 뒤 `r(W_n+rho_n D_n)=r(W_n)+h`를 만족하는 scalar root `rho_n`을 찾고 `beta_n=h rho_n u_n`을 적용하도록 요구한다. exact 실행 source의 `_run_dynamic_arm()`은 `coefficients=(velocity*step_size)`를 만들고 즉시 `trajectory.apply()`를 호출한다. dynamic block에는 `rho` 또는 root-localization 호출이 없고 terminal node에도 rho/root receipt가 없다.

|검사|관측|판정|
|---|---:|---|
|설계의 scalar exact localization 요구|PASS|required|
|dynamic source의 `h*u` direct apply|PASS|observed|
|dynamic rho/root receipt|0/17|absent|
|최대 abs nonlinear progress residual|12.559093|matched-progress FAIL|
|최대 event normalization log residual|0.00004695|PASS|

이 경계는 단순 telemetry 누락이 아니다. terminal `r`이 Native target `r_AE`와 크게 달라져 Full/Fisher/Plain이 같은 semantic progress에서 비교되지 않는다. barrier attribution H2와 dynamic-relinearization H3의 모델 증거는 이 pilot만으로 닫히지 않는다.

## 2. Arm endpoint

낮은 NLL/KL이 좋지만, progress가 일치하지 않으므로 arm 순위는 기술적 관측값이다.

|arm|steps|terminal r|r-rAE|Rewrite new/true NLL|Rephrase new/true NLL|Loc KL|energy|
|---|---:|---:|---:|---:|---:|---:|---:|
|official-native-alphaedit-bypass|1|12.970818|+0.000000|0.725594/16.124846|5.611300/12.695827|0.770053|0.000000|
|plain-dynamic|4|0.411724|-12.559093|14.253559/14.690209|13.976562/14.529531|10.897244|233355.229773|
|fisher-only-dynamic|4|13.306610|+0.335793|3.315482/13.534610|3.525201/17.287363|1.599856|1868.936343|
|full-moving-barrier-dynamic|4|7.529375|-5.441442|4.926925/14.496796|4.924151/12.627197|1.763355|2336.676024|
|one-step-full-barrier|1|3.550217|-9.420601|14.501599/17.608765|8.942920/11.481150|0.168221|38.641116|
|frozen-field-n4-split|4|3.550217|-9.420601|14.501599/17.608768|8.942913/11.481140|0.168221|9.660279|

### 사실 해석

- Native는 `r=12.970818`로 `r_AE`를 그대로 재현했다. adapter relative Frobenius는 `7.9446035e-06`로 tolerance `5e-05`를 통과했다.
- Fisher는 terminal residual `+0.335793`, Full은 `-5.441442`이다. Fisher가 Rewrite/Rephrase/Loc/energy 모두 Full보다 낮게 관측됐지만 progress mismatch 때문에 barrier가 해롭다는 결론은 금지한다.
- Plain은 node 3에서 coefficient abs max `113.333809`, total energy `233355.230`, Loc KL `10.897`로 붕괴했다. 이는 unlocalized first-order step의 불안정성 증거이지 Full barrier의 causal 승리 증거가 아니다.
- one-step과 frozen-N4 endpoint는 Rewrite new delta `0.000000000`, Rephrase new delta `0.000006676`, Loc KL delta `0.000000194`로 사실상 동일하다. 이는 frozen-field split identity를 지지하지만 dynamic ODE 우위를 지지하지 않는다.

## 3. Node progress residual

|arm|K1|K2|K3|K4|terminal abs|
|---|---:|---:|---:|---:|---:|
|plain-dynamic|-1.773177|-4.071566|-9.557557|-12.559093|12.559093|
|fisher-only-dynamic|-1.283622|-4.041796|-7.734624|0.335793|0.335793|
|full-moving-barrier-dynamic|-1.283027|-5.336546|-5.567918|-5.441442|5.441442|
|one-step-full-barrier|-9.420601|nan|nan|nan|9.420601|
|frozen-field-n4-split|-1.283027|-3.297220|-6.266450|-9.420601|9.420601|

`a^T u=1`의 solver equality residual과 위 nonlinear residual은 다른 양이다. 전자는 local tangent equation, 후자는 실제 nonlinear W update가 의도한 `h`만큼 진행했는지를 측정한다. 이번 pilot은 전자를 통과했지만 후자를 local root로 교정하지 않았다.

## 4. z→W와 single-request 성능

- W0 Rewrite/Rephrase new NLL: `12.262517` / `8.112165`.
- fixed native z*: `0.001026` / `3.405871`. Rewrite는 exact target을 만족했지만 Rephrase는 만족하지 않았다.
- Native W: `0.725594` / `5.611300`. fixed-z 대비 W gap은 `+0.724568` / `+2.205429`이다.
- 단일 request pilot이므로 success rate, 평균 우위, generalization/locality promotion으로 확장하지 않는다.

## 5. 기술 gate와 compute

- terminal status `BGODE_R1_S1_TERMINAL_PASS`; FULL-FP32 parameter tensors `291/291`; BF16/FP16/quantization/autocast/cast 0.
- total wall `1256.037s`; model load `4.364s`; peak allocated `37737603584` bytes; reserved `41655730176` bytes.
- fixed z compute/recompute `1/0`; Euler history append 0; heldout/locality controller influence 0; terminal W0 pointer+bytes rollback gate PASS.
- Plain first-node JVP-vs-FD max abs error `0.00387476` with configured allclose PASS.
- Native physical writes/layers `1/5`; each dynamic N4 arm `4/20`; one-step `1/5`. dictionary helper internal physical forward count는 `NOT_RESOLVED_SOURCE_LEVEL_HELPER_INTERNALS`로 보존한다.

## 6. 과학 판정

|hypothesis|판정|근거|
|---|---|---|
|H1 event validity|SUPPORTED_FOR_THIS_PILOT|normalization residual finite/small, tokenizer boundary fixed, JVP-vs-FD PASS|
|H2 barrier attribution|OPEN / INVALID_MATCHED_PROGRESS_PANEL|Full/Fisher/Plain terminal progress와 energy가 불일치|
|H3 ODE attribution|OPEN|one-step=frozen identity는 확인, dynamic benefit은 localization 없는 trajectory라 판정 불가|
|Native fidelity|PASS|Official endpoint relative Frobenius within tolerance|
|Scientific promotion|FALSE|B=1 단일 sample + missing localization|

다음 유효 실험은 현재 결과의 threshold 조정이나 imputation이 아니라, exact design대로 node별 monotonic bracket과 `rho_n` root를 구현하고 `rho`, actual/linear progress, subdivision count를 저장한 뒤 동일 sample·동일 terminal progress에서 Fisher/Full/Plain을 재비교하는 것이다. Qwen 결과는 SH1 소유의 별도 package로 유지한다.

## 7. 산출물

- `arm-endpoints.csv`: 6-arm endpoint NLL/event/KL/energy/compute
- `node-trajectory.csv`: dynamic node별 actual/expected progress, residual, energy, coefficient, rho absence
- `compute.csv`: dictionary/write/JVP/forward/메모리 ledger
- `analysis-summary.json`: 핵심 gate·verdict·비교 경계
- `analysis-manifest.json`, `rooted-analysis-receipt.json`: input/output byte identity

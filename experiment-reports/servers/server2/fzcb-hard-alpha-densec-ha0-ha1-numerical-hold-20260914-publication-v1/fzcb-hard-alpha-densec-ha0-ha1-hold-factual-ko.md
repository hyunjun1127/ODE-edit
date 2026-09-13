# FzCB Hard-Alpha Dense-C HA0/HA1 수치 HOLD 사실 보고서

## 한눈에 보는 결론

|단계|Llama|Qwen|다음 단계|
|---|---:|---:|---|
|HA0 model-free dense backend|13/13 PASS|공통|HA1 허용|
|HA1 Official replay B1|TERMINAL_VALID|SENSITIVITY_NUMERICALLY_INCONCLUSIVE|HA2/B10 미제출|

Llama는 zero/nonzero dense-history 전 layer에서 direct Official 식과 LU/Woodbury backend parity, FP32, padding/position, cache mutation0, W0 pointer+bytes restore를 통과했다. Qwen layer4 zero-history에서는 두 solve residual이 모두 작고 D 차이는 lock 이내였지만, residual과 곱한 최종 update 차이는 lock 밖으로 이동했다. 계약의 numerical-boundary 규칙에 따라 tolerance를 완화하거나 이를 구조적 hard-Alpha 실패로 부르지 않고 중단했다.

## 1. 권위 입력과 source

- contract: `/mnt/raid5/janghj/ODE-edit/local/state/fzcb-hard-alpha-densec-ha0-b10-v1/authoritative-contract.txt` / `65d1f80291d64a6cdf97f7e6072d7fe33cb1595f5fcd7462f42e8ad7ecd1b35a` / 27,954 bytes / 1,437 lines / mode0600.
- implementation: `codex/server2-fzcb-hard-alpha-densec-ha0-b10-v1` / HEAD `f4658c3f05d70299266c687f0df00e2c70294d25` / tree `885028d453c3e37e15f287e0431d745d114629c6` / parent `0ae42e079a77e50f1473e04b2fb348dd1bb1d81a`.
- Official source: `/mnt/raid5/janghj/ODE-edit/local/state/fzcb-hard-alpha-densec-ha0-b10-v1/easyedit-official-3488a66-v1` / HEAD `3488a66ee988d83ee7891a8abbbe6bcb24a77daf` / tree `1f6d5e9a4a95daa15a4b5a8963dcd133876e47b3` / clean read-only.
- pinned user EasyEdit dirty checkout edit/reset 영향0. Official `AlphaEdit_main.py` normal equation, RHS, shape adapter, post-terminal K append semantics는 preflight에서 독립 봉인했다.

## 2. HA0 backend 및 transaction

- nonsymmetric `A_H=lambda I+P C_H`: LU factorization/solve. Cholesky/CG/inverse0.
- `D=Q S^-1`: `solve(S.T,Q.T).T`; C_H=0/nonzero direct parity, P=I, hard-range/right leakage, incidence, orientation PASS.
- matrix-free LSQR/null projection/envelope FP64 FD/analytic alpha/terminal limit PASS; explicit Gram/pseudoinverse/null basis/Kronecker0.
- cache checkpoint+append-only WAL replay, rejected append0, successful append-once test PASS.
- pinned Official PyTorch hook signature는 clean source를 수정하지 않고 ODE thin adapter에서 `(module,args,kwargs,output)`으로만 교정했다.

## 3. HA1 Llama 결과

- job `31020_0`, case `21100`, result `/mnt/raid5/janghj/ODE-edit/local/results/fzcb-hard-alpha-densec-ha1-b1-tech-r2-v1/llama3-8b-inst/ha1-result.json`.
- update relative difference 범위 `5.415562e-06..1.019054e-05`; D direct/Woodbury relative 범위 `1.256610e-06..3.982386e-06`.
- target compute1/recompute0, cache append0/mutation0, FULL FP32, BF16/FP16/autocast0.
- W0 pointer restore=True, bytes restore=True.

## 4. HA1 Qwen 첫 false gate

|항목|값|
|---|---:|
|layer/history|4 / zero|
|Official direct equation residual|1.151874926e-06|
|Woodbury full-equation residual|3.019503936e-08|
|D direct↔Woodbury relative|1.793193514e-04|
|최종 update relative|9.607659304e-04|
|고정 backend tolerance|4.882812500e-04|
|small-system condition|1.0|

동일 evidence는 job `31020_1`과 진단-only job `31024_1`에서 재현됐다. D parity는 통과하지만 최종 update parity는 실패하므로 어떤 receipt를 authoritative parity로 채택하는지에 따라 HA1 판정이 바뀐다. 이는 `SENSITIVITY_NUMERICALLY_INCONCLUSIVE`이며 임의 tolerance 변경·direct-only fallback·post-hoc projection을 적용하지 않았다. Qwen terminal endpoint denominator는 0, imputation0이다.

## 5. 기술 이력과 실행 경계

- `31006_[0-1]`: model load/science0, preflight SHA mismatch 메시지의 actual/expected 증거 누락; fail root immutable.
- `31015_[0-1]`: model load 후 target0, pinned Official hook signature incompatibility; thin adapter로만 수리, source bytes 불변.
- `31020_0`: Llama valid. `31020_1`: Qwen numerical boundary. `31024_1`: Qwen evidence-only 재현.
- HA2 four-arm B1/Atomic B10, HA3 B10×10, B100×10 submission은 모두 0. cache history claim0, scientific promotion=false.

## 6. 후속 plan

HA2 release에는 Qwen direct/Woodbury parity authority를 식·target·stream·tolerance 변경 없이 명시적으로 닫는 numerical decision이 필요하다. 그 전에는 Official/Frozen/Equality/FzCB 네 arm의 B1 또는 B10을 제출하지 않는다. HA3와 B100×10은 계획만 유지하며, HA2 timing이 없으므로 B100×10 compute/memory 수치 추정은 `NOT_RECORDED`이다.

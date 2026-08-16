# P1R30 순차 백엔드 adapter-v3 추가 보고서

## 상태

- 상태: `BACKEND_HARDENED_WAITING_VIABLE_ATOMIC_ADAPTER`
- 범위: model-free adapter interface amendment only
- 체크포인트: `a9f846261cd625a48347fcafc5d06dd61322ccfa`
- 부모: `fa3d570bdcea49cdaa1d705dfd399648a4920778`
- 트리: `e815b10bb9fda70b69c0b890e74d06688fda3631`
- 과학적 sequential 실행: HOLD
- model/GPU/Slurm/scientific result-root action: 0/0/0/0

## FACT

기존 v2 백엔드의 BF16 상태, raw/projected history keys, exact Woodbury와 stale-cache exact fallback,
cumulative Structural-H/P, cache identity, all-or-nothing transaction, terminal six-field key identity는
수정하지 않았다.

adapter의 universal 입력 계약에서 `semantic_rho`를 제거했다. 백엔드는 더 이상
`a^T c = semantic_rho` 또는 `transform = rho/slope`를 보편 계약으로 요구하지 않는다.

새 typed adapter payload는 다음을 atomic adapter가 직접 제공하도록 한다.

- reference allocation `c0`
- signed reference mean progress
- signed reference debt-priority progress
- nonnegative reference energy
- selected coefficients
- constraint receipt identity
- certificate receipt identity
- reference feasibility 및 reference fallback 상태

두 progress 값은 finite signed 값이며 0과 음수를 허용한다. 오직 reference energy와 layer
coefficients만 finite nonnegative를 요구한다. 이는 aggregate slope/progress가 0 또는 음수인
상태도 adapter가 그대로 표현할 수 있게 한다.

backend는 target/debt preprocessing을 구현하지 않고 model F/B를 추가하지 않는다.
선택된 atomic adapter가 preprocessing 및 action/certificate 의미를 소유한다.

## 집중 검증

- focused model-free tests: 26/26 PASS, warnings-as-errors
- compile: PASS
- bash: PASS
- dry integration plan: PASS
- signed/zero progress positive fixtures: PASS
- nonfinite progress 및 negative energy negative fixtures: PASS
- `semantic_rho` dataclass field absence: PASS
- universal inverse-slope transform absence: PASS
- v2 transaction/history/H/P/cache source preservation: PASS

실제 B10x2 model smoke는 adapter가 선택되지 않았으므로 `NOT_RECORDED_NO_ADAPTER_SELECTED`이다.

## 산출물

- v3 lock: `project/locks/p1r30_backend_adapter_amendment_v1.json`
- v3 source manifest: `project/locks/p1r30_backend_adapter_source_manifest_v1.json`
- preserved v2 lock SHA-256: `eb7607ac067481cf83c4a7447b18e56624c89fa022b8dbf59ad619304c3850ff`
- preserved v2 report SHA-256: `18ddc3ffffc1b143f19fee81b50c895147c131cc87291ca76ea05d5fc1424b2a`

## 경계

이 변경은 P1R30 atomic method의 승인이나 sequential 과학 결과가 아니다. viable Atomic adapter와
별도 scientific execution release가 있을 때까지 백엔드는 준비 상태로 유지한다.
`scientific_promotion=false`.

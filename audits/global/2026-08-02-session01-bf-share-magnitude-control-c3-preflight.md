# BF-share magnitude-only c3 최소 preflight

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 대상 job: `odeedit_capacity_history_pair_c3_v1`
- 감사 범위: magnitude-only 식별, exact share identity, model-common/firewall/read-only,
  session/resource/output boundary의 필수 항목만

## 원인분리 계약

- [x] c1 common-frontier-all-layers QP를 별도 allocation policy로 복원
- [x] c1과 같이 full remaining native rewrite gap을 allocation request로 사용
- [x] `x_BF`의 nonzero ratio/zero support를 radial normalization에서 보존
- [x] global `D/4`를 독립 적용하고 fixed 4 hops/total `D`를 evaluator가 검증
- [x] allocation cap/barrier와 applied magnitude를 metadata에서 분리
- [x] applied hard-barrier claim은 false; capacity metric은 ex-post 결과로만 해석
- [x] Llama/Qwen, MEMIT/Alpha에 단일 policy와 동일 code

## 필수 검증

- capacity/QP/controller/evaluator/analysis/lineage/Alpha relevant suite: `56/56 PASS`
- Python compile 및 worker/pair/helper `bash -n`: PASS
- `git diff --check`, core tracked-path identity: PASS
- session boundary: repository `hyunjun1127/ODE-edit`, server1,
  `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`, confirmed `Sol Ultra`: PASS
- resource: active project GPU 0 + requested 4 <= cap 4; requested memory `260000M` <=
  allowed `792468 MiB`: ALLOW
- c3 controller/evaluator/combined/pair paths와 marker: fresh
- same-name active job: 없음
- clean pushed main은 helper가 submit 직전에 다시 fail-closed 검증

## Red-team kill test

- under-update 재발: evaluator가 applied `D/4 × 4`를 강제하므로 block
- share drift: evaluator가 applied/allocation radial proportionality를 강제하므로 block
- cap 의미 은폐: allocation-only와 applied hard-cap false가 없으면 block
- model-specific rescue: 단일 controller constants/pair wrapper 외 branch 금지
- leakage: 네 controller terminal/hash barrier 뒤 별도 evaluator만 허용
- artifact waste: pinned covariance/projector/history read-only; recompute/download 금지
- resource waste: 4-edit 4-GPU simultaneous pair 1회, 이 결과 뒤 Motivation retune 금지
- overclaim: c3는 cause isolation이며 lifelong/deployable hard-barrier superiority 아님

## GH 직접 제출 예외

- 사유: server1 SH 부재, 사용자의 빠른 재구현·양 모델 동시 실험 지시
- 허용 명령: `project/run_scripts/submit_session01_capacity_history_pair_server1.sh`
- 영향: server1 GPU 4 / CPU 32 / `260000M`, c3 ignored local namespace만
- 후속: startup 연속 확인, 이후 30분 monitor, raw hash/integrity, 모델별 agent 시도와
  GH synthesis, no-peer broadcast exception 기록

## Agent gate

관련 c3 code/spec/test만 허용한 Terra Ultra red agent는 runtime metadata에서
`gpt-5.6-terra / ultra`를 검증하지 못해 지정 파일을 읽지 않고 즉시 `BLOCK`
종료했다. 독립 red review로 세지 않는다. GH fallback은 원인분리 control 1회와
negative/식별 caveat 보존에만 한정하며 positive method claim을 사전 승인하지 않는다.

- 최종 판정: `PASS` — clean pushed main에서 c3 BF-share magnitude-control 4-GPU pair 1회 제출에만 유효

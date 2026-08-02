# Capacity-share exact-quarter c2 preflight

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 대상 job: `odeedit_capacity_history_pair_c2_v1`
- 감사 범위: 구현 원인분리, exact magnitude, model-common identity, firewall/read-only,
  session/resource/output boundary의 필수 항목만

## c1 RCA와 claim 정정

- c1은 QP absolute coefficient를 그대로 update로 적용했다. Layer relative share와
  global step magnitude가 분리되지 않았다.
- Llama MEMIT/Alpha는 full requested progress feasible round가 각각 `0/16`이고
  weighted progress slack/request가 `0.766971/0.716977`이었다.
- 두 Llama family 모두 overloaded layer observation이 0인데도 round 2--4에서
  대부분의 layer coefficient가 common-frontier cap에 닿았다.
- realized path/native는 MEMIT `0.671819`, Alpha `0.686245`; native rewrite utility
  fraction은 `0.698943/0.834113`이었다.
- 판정: c1 `CAPACITY_HISTORY_HARM_SIGNAL`은
  `SUPERSEDED_IMPLEMENTATION_CONFOUNDED_LOW_UPDATE`다. Raw artifact와 수치는 보존하고
  BF layer share/direction refresh의 scientific kill evidence로 쓰지 않는다.

## c2 구현 확인

- [x] QP solution은 relative layer allocation으로만 사용
- [x] per-layer cap 안 radial rescale로 applied share L2 norm을 1로 고정
- [x] global hop을 별도 `D/4`로 고정하고 evaluator가 `(D/4)^2 × 4`를 검증
- [x] total accepted path `D`; early stop와 trust shrink 없음
- [x] common frontier는 overload detector로만 사용; non-overloaded cap은 full hop
- [x] truly overloaded layer는 current capacity 이상 증가 금지
- [x] exact hop이 cap 안에서 불가능하면 under-sized endpoint 대신 fail-closed
- [x] Llama/Qwen, MEMIT/Alpha에 같은 policy/run code

## 최소 검증 증거

- relevant capacity/controller/evaluator/geometry/lineage/history suite: `40/40 PASS`
- Python compile 및 three Slurm/helper `bash -n`: PASS
- `git diff --check`: PASS
- session boundary:
  `hyunjun1127/ODE-edit`, server1,
  `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`, confirmed `Sol Ultra`: PASS
- resource cap: active project GPU 0 + requested 4 <= cap 4; requested memory
  `260000M` <= allowed `792468 MiB`: ALLOW
- c2 controller/evaluator/combined/pair output path와 submission marker: fresh
- same-name active Slurm job: 없음
- EasyEdit source는 실행 wrapper에서 tracked ODE-edit hook과 pinned precomputed
  artifact만 읽으며 online/download/recompute 환경을 차단

## Red-team kill test

- under-update 재발: evaluator가 exact per-hop/total C-distance를 강제하므로 block
- model-specific rescue: 단일 controller constants와 동일 pair wrapper이므로 block
- barrier 무력화: true-overload cap과 evaluator polynomial check 유지
- evaluation leakage: 네 controller terminal/hash verification 뒤 별도 evaluator process
- Alpha artifact waste: mmap projector/covariance/history reuse, write/recompute false
- resource waste: 4-edit fixed panel과 4-GPU simultaneous pair 1회만 허용
- overclaim: c2는 c1 magnitude confound 진단이며 lifelong/superiority evidence 아님

## GH 직접 제출 예외

- 사유: server1 SH 부재, 사용자의 time-critical 재구현·동시 실험 직접 지시
- 허용 명령: `project/run_scripts/submit_session01_capacity_history_pair_server1.sh`
- 영향: server1 GPU 4 / CPU 32 / `260000M`, c2 ignored local namespace만
- 후속: startup 연속 확인, 30분 monitoring, raw hash/integrity, 모델별 분석과 GH synthesis,
  active peer 없음 artifact-broadcast exception 기록

- 최종 판정: `PASS` — clean pushed main에서 c2 exact-quarter 4-GPU pair 1회 제출에만 유효

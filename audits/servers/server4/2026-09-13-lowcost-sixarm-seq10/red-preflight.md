# Six-arm sequential: bounded independent red preflight

Instruction: ODEEDIT-S06-LOWCOST-SIXARM-SEQUENTIAL-B100X10-SH4-V1.
검토 범위는 신규 sequential orchestration/evaluation/prepare/launcher와 재사용 fitting 계약이다. 기존 실험을 재감사하거나 source/raw를 변경하지 않았다. GPU/model/Slurm 실행·조회는 이 검토자가 수행하지 않았다.

## 판정과 검증 경계

현재 읽은 구현에서 실행을 차단해야 할 과학 계약 불일치는 발견하지 않았다. 이는 **CPU/source preflight**이며 GPU initial-valid 또는 60 batch 완료 판정이 아니다. 실제 aggregate cap2 admission, held resource/source inspection, source archive sealing은 주 실행자의 별도 receipt에 결속해야 한다. 이 문서만으로 제출 자원 적합성 또는 실제 모델 복원 성공을 주장하지 않는다.

| 항목 | source/CPU 관측 | 경계 |
|---|---|---|
| Sample | `batch_bounds`는 B51=[5000,5100), B60=[5900,6000); 각 배치 case/order/전체 record digest 검사 | 실제 prepared CPU receipt와 sample lock은 주 실행자 생성·검증 |
| 독립 chain | 각 프로세스 한 arm, 공통 prepared 복원은 loop 전 1회; 이후 ledger가 자기 previous commit과 다음 entry를 비교 | 모델 수준 B51→B52는 제출된 프로그램의 initial gate |
| First fit | 매 배치 자기 W에서 L4 fresh fit; actual native weight와 entry로 FP32 materialization | 이전 static endpoint/D4 재사용 없음 |
| Second fit | FULL8/RES8 L8, REFIT4 L4 fresh fit; 실제 partial state 기록; 다른 selected layer 불변 검사 | z/K/R 수치 동등성은 새 GPU 결과로만 확인 가능 |
| History | 두 fitting 사이 M4/M8 불변; final selected layer별 append1; REFIT4 binding 하나; unused M8 유지 | native history AST가 유지되는 CPU test 재사용 |
| P mapping | full stack physical L4→0/local0, L8→4/local0; prepared metadata와 실제 selected tensor 결속 | source-exact original P를 준비 단계에서 검사 |
| Alpha | 0/1은 endpoint copy, .75/.875는 CPU FP32 entry+alpha*(native-entry) | 성능 기반 alpha 변경/구제 없음 |
| Evaluation | actual endpoint W/M/P/context/RNG 및 parameter pointer/version 전후 검사, controller 반환값 사용 없음 | nonselected byte 전체 확인은 시작/terminal, 중간 pointer/version guard와 구분 |
| Full-seen | B60 full6000 한 번에서 Current100/suffix1000/old5000/Historical128 identity subset; B55 suffix500에서 Current 재사용 | 큰 MB16 forward의 동일 raw rows이며 별도 rebatching 수치 동등성을 주장하지 않음 |
| Checkpoint | B51/55/60 selected W/M/context/RNG+common prepared refs; CPU weights_only reload/hash 검사 삽입 | GPU continuation replay NOT_TESTED; chronological increments도 exact replay 주장 없음 |
| Audit | Audit128/MMLU68 평가 호출 없음, dev32만 고정 사용 | 후속 선택/자동 제출 없음 |
| Resources | sbatch explicit mem60416M 정확히 1개, 1GPU/8CPU/exportNONE/0-5%2; mapping N4,RES8,S875,S75,FULL8,REFIT4 | 실제 다른 project admission 포함 cap2는 별도 단발 운영 receipt 필요 |
| Cost | 2–8 GPUh/chain, 80GiB 저장 envelope는 estimate라고 명시; shared M8 새 재구성0 | 미래 실제 wall/GPU/storage 또는 사용자 hour cap으로 오기하지 않음 |

`sequential_submit.py`도 읽었다. 제출 직전 registry와 단발 squeue를 사용하며, 다른 project admission이 있으면 새 array에만 해당 작업들의 afterany를 연결한다. 기존 admission이 없을 때에는 불필요한 arm 간 직렬화 없이 `%2`이다. held 단계에서 exact owner/job/source/request/memory/throttle를 확인한 뒤 release하고 6 cell 상태를 한 번 기록한다. 제출·release 실패를 runtime PASS로 바꾸지 않으며, pre-existing job 조작은 없다. 주 실행자는 예비 확인에서 server4 project jobs 없음과 partition 최대 30일을 전달했으나 이 검토자는 scheduler를 중복 조회하지 않았다. 실제 최신 자원 판단은 제출 시 admission receipt가 우선한다.

## 좁은 보강 및 확인

검토 중 실제 model에 prepared를 복원한 `common_entry == lock['common_state']` fail-close assertion을 권고했으며 구현에 반영된 것을 확인했다. 초기 모델 준비 실패의 finally 경로가 없는 flags/RNG를 복원하며 원인을 덮어쓰지 않도록 guard를 권고했고 반영을 확인했다. 최종 성공 후 source/bytes는 주 실행자의 immutable archive로 봉인한다.

## 독립 CPU 검증

다음 16 tests PASS: sequential runtime 6, sequential evaluation 5, native fitting 5. 실제 production ledger/evaluation guard/rollback helper를 CPU toy state로 검사하며, 전체 GPU 모델의 모사라고 부르지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 /data/janghj/EasyEdit/.venv/bin/python -m unittest \
  project.run_scripts.low_cost_write_donor_pilot.test_sequential_runtime \
  project.run_scripts.low_cost_write_donor_pilot.test_sequential_evaluation \
  project.run_scripts.low_cost_write_donor_pilot.test_fitting -q
```

실행 결과: `Ran 16 tests ... OK`. Order/target drift, history between fits, duplicate finalization, rollback/evaluator mutation, prompt pairing/state mismatch/nonfinite/tie policy, 6000 cardinality 및 source AST/P/alpha가 검사 대상이다. 유한 낮은 성능, 비용 참고선 초과는 제외 또는 gate 실패 근거가 아니다.

초기 actual gate 또는 PENDING inspection 후 MONITORING_PAUSED_AWAITING_USER. 이 preflight는 추가 모니터링·후속 제출·main 통합 권한을 만들지 않는다.

## 제출 기록의 명시 경로 권한 — 좁은 사후 확인

주 실행자가 요청한 범위만 독립 확인했다. `runs/lowcost-sixarm-seq10-s4-20260913-v1/submission.json`은 governing envelope `messages/head/2026-09-13-sh4-lowcost-sixarm-seq10.md`의 서버 소유 경로 목록(122행)에 명시된 디렉터리 안에 있다. 따라서 **이번 task의 이 raw-free 기록에 한정하여 명시적 사용자 envelope상 허용 범위**라고 판정한다. 이는 일반 `runs/*` 쓰기 권한 확대나 다른 task 파일의 예외가 아니다.

- 검사한 submission SHA256: `215f75ee008634272a82673fdc79fe59155aa7335404ebc1259c8e6841d03393`.
- governing envelope SHA256: `eb1d7eccd662cc8ef1c8a3649e3c798893f22313a95e514f4b2201f1cdfccdcd`.
- generic helper SHA256: `eac168b1d379ee0aea6edf24f5171893b1375abdf6f42dce2dd00c335539bc51`.
- 파일은 job/array mapping, source/archive/lock/receipt hashes, local 경로, 자원 및 검증 상태의 작은 metadata다. 실제 prompt, tensor, weights, cache, full logs, credentials payload는 포함하지 않는다. 파일 내용의 자원·job 관측은 주 실행자 receipt를 인용한 것이며 이 검토자가 scheduler를 새로 조회한 결과가 아니다.
- generic `scripts/check-agent-access.sh`의 server-head allowlist(65행 부근)에는 이 `runs/` 경로가 없고, worker 전용 규칙(106행)은 agent-ID suffix 파일만 허용한다. 따라서 해당 helper가 거부한 사실을 **PASS로 바꾸지 않는다**. 기록 상태는 `GENERIC_HELPER_DENIED / EXPLICIT_TASK_ENVELOPE_ALLOWED`이다. shared helper/role/protocol 수정은 수행하지 않았다.
- `621ec8b202a482141771e59b141640e8fcc1d99c`의 source branch 기록은 이미 생성된 상태에서 확인했다. 이 확인은 frozen runtime `5e96dcb3745977b1f273e3f5afbee61167248d49`/archive/lock 또는 live job의 변경·재평가 승인이나 initial GPU gate PASS가 아니다. main 통합을 수행하지 않았다.

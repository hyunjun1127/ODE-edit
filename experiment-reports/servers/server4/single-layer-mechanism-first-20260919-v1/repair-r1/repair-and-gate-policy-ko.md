# Native z hook 실패 수리 및 사용자 진단-gate 정정

2026-09-20 USER recall: “실험 끝난거 fail repair 바람”. 이어 “저런 부분은 너무 strict한 gate인데 어느정도 lenient하게 파악해도 돼.”를 적용했다. 본 문서는 과학 완료 보고가 아니라 실패 RCA·구현 수리·재제출 인계다.

## 실제 실패와 완료 경계

| Job | 확인한 상태 | 실제 범위 | 비용/한계 |
| --- | --- | --- | --- |
| 50974 / odeedit_slmf_T0_s4 | FAILED, ExitCode 1:0 | 고정 4요청 native/cache z와 actual write 비교; hook gate 실패 | parent allocation 136 GPU-sec; B1 과학 batch 0 |
| 50983 / odeedit_slmf_S10_s4 | PENDING, DependencyNeverSatisfied | afterok:50974(failed); 미실행 | allocation 0; 취소·dependency 변경 없음 |

정확 두 job만 recall accounting으로 확인했다. 50974 Slurm start/end 반환값은 2026-09-20T01:50:04 / 01:52:20, program failure timer는 134.2330초다. Scheduler allocation과 program wall을 합산하지 않는다. 당시 등록 직후 PENDING/Reason=None 기록은 수정하지 않았다.

50974의 첫 예외는 `T0_NATIVE_Z_HOOK_PARITY_FAILED`다. 새 과학 결과가 나쁘거나 B1→S3 gate가 실패한 것이 아니다. 실행 source는 `1cf90d2f55e7e4f270c2c2da845f6d24707ca24b`; 후속 미실행 프로그램 source는 `0aea3452d04e71dd56e428208e04b9c04c52f82f`다.

| 기존 관측 | 수치/상태 | 기존 기준 |
| --- | --- | --- |
| production cache+head batch1, request index 2의 최대 궤적 gradient 상대차 | 1.4032985639893358e-4 | 1e-4 초과 |
| 같은 요청 초기 gradient 상대차 | 약 5.37e-6 | 후속 독립 Adam 궤적에서 차이 증가 |
| 최대 상대차 step의 gradient 성분 최대 절대차 | 약 5.58e-7 | 새 절대 기준을 만들지 않음 |
| batch1 실제 write 후 Current 최대 NLL 차이 | 8.58306884765625e-5 | 1e-4 이내 |
| batch1 strict/pair ID·25 loss/24 Adam 종료 | 일치 | exact |
| batch4 진단 actual write 최대 NLL 차이 | 1.163482666015625e-4 | 초과; production batch1로 사전 고정 |

상대 gradient 비교는 각 경로가 자신의 앞선 Adam update를 적용한 **서로 다른 delta 궤적**의 비교다. 같은 점에서 direct/cached 미분을 비교한 PASS/FAIL로 확대하지 않는다. 값이 작다는 사실만으로 bitwise 동등성이나 전체 기술 PASS를 주장하지 않는다.

## 최소 source 수리

Native 원 source SHA `a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f`는 읽기 전용이다. 새 task adapter 안에서 다음 FP32 연산 순서를 정렬했다.

- context별 sequential inplace delta add와 AD 합산 경로
- full rewrite hidden의 native norm 후 target-position full-vocab head
- 요청별 native NLL negate/sum/mean, KL batchmean, scalar norm 및 singleton backward

기존 수학식·FP32/eager/TF32off·native hparams·25 loss/24 Adam·full vocabulary·production batch1은 유지한다. 관측 오차 중 어느 연산 차이가 얼마를 유발했는지는 분리 측정하지 않았다. 수리 후 실제 GPU 결과는 등록 시점 `NOT_OBSERVED`다. batch16 실제 검증/성공을 주장하지 않는다.

## 사용자 leniency의 정확한 범위

새 lock의 `USER_TRAJECTORY_GRADIENT_DIAGNOSTIC_20260920_V1`만 적용한다. 과거 strict 결과·1e-4 ceiling·초과 요청은 계속 저장하며 과거 실패를 PASS로 바꾸지 않는다. 새로운 큰 epsilon을 결과에 맞춰 고르지 않았다.

| 항목 | 새 실행에서의 역할 |
| --- | --- |
| 독립 z 궤적의 gradient 상대차 1e-4 초과 | WARN_DIAGNOSTIC_ONLY; 단독 중단 사유 아님 |
| finite/trace 완결성·loss/Adam 종료 횟수 | 여전히 필수 |
| native 궤적 NLL 차이 ≤1e-4 | 여전히 필수 |
| actual write Current NLL 차이 ≤1e-4와 strict/pair ID exact | 여전히 필수 |
| 이후 한정 full T0의 same-point/FD·geometry·state 검사 | 변경 없음, 미실행이면 NOT_ESTABLISHED |
| 실제 candidate 수용·full512·Current/history·B1/S3 확대 gate | 변경 없음 |

따라서 `historical_strict_gate_pass=false`와 사용자 정책상 `pass_=true`는 함께 기록될 수 있다. 이것을 원 기준 전체 numerical PASS라고 쓰지 않는다. NaN/Inf, 잘못된 상태, actual write 불일치를 경고로 숨기지 않는다.

## 기존 계산 재사용 및 no-checkpoint

기존 고정 4요청 unhooked native target/trace/write를 정확 artifact SHA와 source/input/runtime binding으로 읽기 전용 재사용한다. CPU에서 case ID 14521/6907/6540/8631, target [4096,4], retained weight [4096,14336] 및 target→write 결속을 확인했다. 새 실행의 W0/runtime/tokenization 일치는 job에서 다시 결속하므로 CPU 검사만으로 actual runtime PASS라 하지 않는다.

새 bounded hook 재검증은 수정 batch1의 4 target + batch4 진단의 4 target이며, unhooked 기준의 중복 fitting은 0이다. 새 z 결과의 actual write에는 saved targets를 사용하여 추가 z를 계산하지 않는다. 기존 136 GPU-sec는 새 allocation과 별도다.

수리 T0와 조건부 후속은 한 process에서 연결한다. 새 W/M/전체 weight delta/RNG resume checkpoint 저장은 0, exact crash-resume는 `NOT_AVAILABLE`이다. 작은 z/gradient 진단과 native target/key, scalar/hash/metric ledger만 보존한다. 과거 실패 파일 및 기존 native reference는 삭제·덮어쓰지 않는다.

## 검증·자원·운영

- 전체 CPU 207 tests PASS (최종 8.520초), 별도 좁은 41 tests PASS. 서로 겹치므로 248개 독립 test로 합산하지 않는다.
- 위 수리 helper의 20 CPU tests는 전체 검사에 포함된다. 별도 bounded helper가 hook 3파일을 구현했고 부모가 diff와 통합을 검토했다. 별도 독립 red/GPU audit는 수행하지 않았다.
- CPU retained binding PASS, actual fresh tokenizer/runtime 및 GPU parity는 아직 NOT_OBSERVED다.
- session helper는 전용 worktree의 ignored 설정 부재로 NOT_PASS다. 실제 host server4, repo origin, root CWD 및 현재 registry의 SH4 session은 결속했다. 공용 helper/config 변경은 0이다.
- 기존 50983을 직접 변경하지 않으며 resource-only 목록에 나타나는 기존 job은 보수적으로 cap에 합산한다. 새 inline repair job 1GPU와 합계≤2만 허용한다. CPU8/mem60416MiB/exportNONE/Requeue0, planned wall 7일. GPUh hardcap 미지정이다.
- 초기 24GiB 및 후속 stage 저장 검사는 유지하며 기존 storage waiver를 상속하지 않는다. Stage별 여유를 실제 예약/보장으로 표시하지 않는다.
- 등록 후 자동 agent 모니터링·후속 submit은 0이다. 프로그램은 기존 기술·과학 gate를 만족하는 범위만 최대 B10까지 자체 진행하며 실패 gate에서 종료한다.

`NO_BROADCAST_NOT_REQUIRED`: 같은 server4의 기존 입력/기준을 사용하며 새 원격 raw 전송, 삭제, 다른 task 재개는 없다.

## 재현

CPU: `python -B -m unittest discover -s project/run_scripts/single_layer_mechanism_first -t . -p 'test_*.py'` (pinned local venv 및 transformers 4.44.2 overlay).

Create-once plan: `python -B -m project.run_scripts.single_layer_mechanism_first.repair_plan --output <new-task-plan.json>`.

Clean execution commit을 `freeze --phase GATED_PROGRAM --attempt hook-repair-r1`로 봉인하고 `submit_repair --lock <execution.lock.json>`로 held inspection/release한다. 이미 submission receipt가 있으면 중복 제출을 거부한다. 실제 source/archive/lock/job은 후속 compact registration receipt에 기록한다.

## 실제 재등록 인계

새 job **51055 / odeedit_slmf_repair_s4**를 held inspection 13항목 PASS 후 release했다. Admission 시각 2026-09-19T17:13:17.061245Z(한국시각 2026-09-20 02:13:17), resource-only own queue 목록은 비어 있었다. 앞서 확인한 50983의 상태를 다시 과학/task 조회하지 않았으며 SH의 cancel/hold/dependency 변경은 0이다.

Release 직후 반환 상태는 PENDING / Reason=None / Dependency=(null)이다. 이를 GPU 자원 부족이나 repaired T0 성공으로 해석하지 않는다. 사용자 모니터링 중지 지시에 따라 registration 이후 scheduler/result/log를 조회하지 않았다.

- 실행 source `5ea4e4efad5a9420674641dd13a04d4651701a08`, tree `ecb82501825c3df49d03d607d3be55ce0e444e2a`.
- archive SHA `fb394661c9abdea78b2327ea8fd858e82858ad58f5e55af7d9011d85fdd981d1` (464,841,980B).
- lock SHA `d437b1dd190153eee0a247f15a875236245d007bc549b8975a698ef9af346d9e`.
- output `/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/hook-repair-r1/output`.
- 관측 여유 51,965,227,008B, inode 225,321,719; 초기 24GiB 조건 충족. 후속 stage 여유는 보장하지 않는다.
- actual repaired hook/full T0/B1/S3/S10는 모두 `NOT_OBSERVED`; 새 allocation 비용도 아직 미검산이다.
- 표 3개/20행/열수 4·3·2 source 검사 PASS. HTML renderer는 해당 venv에서 미설치이므로 실제 HTML render NOT_RUN, 그림은 이 인계에 불필요하여 생성하지 않았다.

[compact 등록/receipt](../../../../../audits/servers/server4/single-layer-mechanism-first-20260919-v1/repair-r1/registration.json)는 repository root 기준 audit 경로에 보존한다. `monitoring_active=false`, `automatic_resume=false`, `WAITING_USER_RESUME`.

# Sequential Local-z Allocation v2 — 본실험 초기 gate 재개 인계

상태: **MAIN_GPU_RESOURCE_PENDING_HANDOFF / MONITORING_PAUSED_AWAITING_USER**. 여섯 arm 전체 완료·성능 검증 보고가 아니다.

## 권한·원본 보존

사용자 정정에 따라 기술 PENDING/PASS에서 멈추지 않고 본실험 기준으로 진행했다. Instruction `ODEEDIT-S06-SEQUENTIAL-LOCAL-Z-ALLOCATION-V2-SH4-V1`.
이전 technical49421 PENDING 인계와 원 resume manifest는 수정하지 않았다. 원 source `21297ec19e7f5aecec16d2fdb14cc79380a1df94` / tree `26be0758ee75503161c7cafffdec8397e6cf8165`, archive `c97083a1e1039c059b082bcf0ad0de1143bbfbe41a26be1db5c5d7a1bf54a7a8`, execution lock `a41cb76a25ab98b02c043397cd9cf4db0768e9ae312931b349008c3e9b303d18`를 유지했다.
이번 재개·제출·인계용 source는 `31934d3c780fdac89128015320f16618ac57ab85`이며 실제 모델 runtime source와 구분한다. 과학식/허용치/arm/sample/teacher 변경0, 새 W/M disk checkpoint0.

## CPU와 실제 기술 검증

재개·제출·held·초기 gate 결속·자원 산술 CPU 검사를 기존 테스트와 함께 수행했다. 별도 독립 red agent는 이번 재개에서 사용하지 않았고 SH4 자체 검산이다. 이 CPU 결과가 Llama 검증을 대체하지 않는다.

```

----------------------------------------------------------------------
Ran 93 tests in 1.931s

OK
```

| 단계 | 저장 판정 | 근거 SHA256 |
| --- | --- | --- |
| C45678_BOUNDED_SEARCH | PASS | `121b4d694e1c53c94beb654af1a1797fae7726ee773ed997ea4a8a7ac5b66e45` |
| CACHE_REPLAY | PASS | `77a66cb21ff0342c8b7ed4e211176465c2d8d451af1a78773ee80ea6593b1e9d` |
| GATES_REPLAY_REPEAT | PASS | `8c2e2bc0d6007a963cdcd93d4ed39e41639a8be44085f1aa0d54ec024f46f733` |
| HISTORY5_NEXT_ENTRY | PASS | `3b5c2624d7822d16c7282c37f5473074b5c78408581eafdb45ebf111a67e1488` |
| NATIVE_BLUE_45678 | PASS | `d3d52d7a1e4e91ce81434b34e717761ef271db8ae2dd22841fe8c43250ed2072` |
| NATIVE_BLUE_48 | PASS | `d009bc38d07af54f7baf0e93c9df96937b64d158f943ba64ce4242ad5e472714` |
| NATIVE_N4 | PASS | `db42c605564783435567c9de735a7bb9fa93b725142daf8adc5ec1ae7f1f1563` |
| TEACHER_REPRODUCTION | PASS | `04295c4cfd7dc4954707d3330befbc5375079fc83a5002a0f8a8383e68d810fd` |
| W0_COLD_IDENTITY | PASS | `7086d8130642d1b62af8ed4fca5f8809220513a3524626a8b268579ee50568ee` |

기술검사의 native 연결·반복 점수·coverage·history는 기술 first100에서 확인한 범위다. CPU 검사 또는 파일 존재만으로 실제 PASS를 만들지 않았다. C45678 coverage는 운영상 gate-vector 수이며 최적성·solver simplex 인증이 아니다. Teacher 재사용/재생성은 technical-READY와 capsule의 실제 identity를 기준으로 한다.

Native N4/48/45678 연결은 해당 모든 layer의 실제 weight byte가 일치했다(maxabs0). 기존 W0 teacher의 original/effective D=0으로 재생성 없이 재사용했다. C45678은 N4 제외 완료 search vector11/unique endpoint11/distinct a4=6, search suffix-fit32/추가Adam888, pruning4완료, 예산 미완료 후보1이었다. 미완료 후보는 score pool에서 제외했고 기술 오류0, 예산 확대0이다. 종료 이유 SUFFIX_FIT_CAP이며 optimizer 수렴 PASS를 주장하지 않는다.

실측 기술 비용(중첩 component 중복 합산 금지):

```json
{
  "seconds": 3372.4676804896444,
  "native": {
    "actual_adam_recorded": 5688,
    "actual_loss_evaluations_recorded": 9588,
    "clamp_hits_recorded": 5684,
    "counting": "RECEIPTS_NONOVERLAPPING_NATIVE_INVOCATIONS_ONLY",
    "native_fit_inclusive_seconds": 2315.1324399234727,
    "native_key_calls": 47,
    "native_solves": 47,
    "native_target_calls": 4700,
    "nested_component_times_not_added": true,
    "pure_writer": "NOT_SEPARATED",
    "target_calls_without_adam_loss_trace": 800
  },
  "peak_GPU_allocated": 38788203008,
  "peak_GPU_reserved": 44323307520,
  "peak_host_RSS_bytes": 35434647552
}
```

프로그램 wall/native timer와 Slurm allocated GPU-sec는 다른 단위다. 기술 stage/reference 재계산은 본실험6000 arm-request 분모에 넣지 않는다. Pure writer가 분리되지 않은 값은 NOT_SEPARATED다. 이 문서에서는 본실험 전체 비용·최종 결과를 분석하지 않는다.

완료 technical49421의 scheduler 기록(본 job 3379초×1GPU, batch/extern 행을 더해 중복 청구하지 않음):

```
49421|COMPLETED|0:0|2026-09-17T15:13:20|2026-09-17T16:09:39|3379|billing=8,cpu=8,gres/gpu=1,mem=59G,node=1|
49421.batch|COMPLETED|0:0|2026-09-17T15:13:20|2026-09-17T16:09:39|3379|cpu=8,gres/gpu=1,mem=59G,node=1|33942380K
49421.extern|COMPLETED|0:0|2026-09-17T15:13:20|2026-09-17T16:09:39|3379|billing=8,cpu=8,gres/gpu=1,mem=59G,node=1|
```

## 본실험 등록·자원

| Array index | Arm | Job ID |
| --- | --- | --- |
| 0 | C45678 | 49466_0 |
| 1 | N4 | 49466_1 |
| 2 | F48 | 49466_2 |
| 3 | G48 | 49466_3 |
| 4 | C4 | 49466_4 |
| 5 | C48 | 49466_5 |

새 scientific scope는 동일 cold W0/M0 first1000의6chains/60batches뿐이다. Held owner/source/args/resource/dependency inspection과 release 근거는 submission receipt를 따른다. 각1GPU/8CPU/60416MiB/exportNONE/Requeue0, 프로젝트 cap2. 공통 기술은 afterok로 직렬화하며 다른 job은 변경하지 않는다. 정확한 마지막 queue/resource 상태와 판정은 boundary-evidence.json에 있다.

최종 관측에서 GPU8/8 할당·가용0과 모든 main Resources 대기를 확인했다. Priority는 양수, dependency는 해제된 상태이며 수동 hold가 아니다. 동시에 host memory의 scheduler 예약 여유40960MiB도 요청60416MiB보다 작았다. GPU 부족은 실제 필요조건 위반이지만 **GPU만의 독점 원인이라고 주장하지 않는다**. 본실험 allocated GPU-seconds는 당시0이고 actual initial gate는 미관측이다.

제출16h는 실제 common technical wall×10×1.5에 반올림 여유를 둔 자원 예상이지 실측 과학시간/성능 gate가 아니다. 기술 완료 후 사용 가능 disk가 약53.1GB로 바뀌어, 처음의 미측정64GiB reserve와 별개로 아래 실측기반44GiB no-checkpoint resource 계획을 잠갔다. 원 execution lock bytes는 보존했다. 평가/후보/target 예산 축소·데이터 삭제·teacher 생성·checkpoint 추가0이며 공유 volume의 향후 증가까지 보장하는 수학적 상한은 아니다.

```json
{
  "additional_contingency_fraction": 0.25,
  "components_bytes": {
    "miscellaneous": 1073741824,
    "native_tensor": 19910676830,
    "observer": 9924341760,
    "scalar_json_ledger": 2147483648,
    "stdout_stderr": 2147483648
  },
  "disk_W_M_checkpoint": false,
  "empirical_estimate_not_mathematical_serialization_upper_bound": true,
  "new_teacher_bytes": 0,
  "old_execution_lock_unchanged": true,
  "original_pre_measurement_reserve_bytes": 68719476736,
  "reserve_bytes": 47244640256,
  "shared_volume_future_growth_guaranteed": false,
  "subtotal_bytes": 35203727710,
  "version": "MEASURED_NO_CP_POSTPILOT_RESOURCE_V2"
}
```

## 실제 본실험 초기 연결

| 실제 확인 arm | B1 요청 수 | B1 history append | B2 next ordinal |
| --- | --- | --- | --- |
| NOT_OBSERVED | NA | NA | NA |

본실험 초기 gate의 확인 규칙은 B1 selected/selection seal/commit/5층 history receipt와 B2 entry의 W/M/context/RNG 대조다. 이번 본실험은 해당 단계 미실행으로 이 실제 대조를 수행하지 않았고 NOT_OBSERVED로 남겼다. 확인하지 않은 arm의 실제 PASS나 W10 완주를 주장하지 않는다. GPU-resource PENDING으로 인계한 경우에만 GPU allocation 부족 근거와 모든6개 정상 release를 함께 요구했다. Dependency/수동 hold/Reason=None/단순 기술 PASS를 해당 예외로 쓰지 않았다.



## 저장·한계·정지

W/M은 runtime RAM snapshot만 유지한다. 필수 native target/key/scalar·candidate score/ID/budget/cache·commit/history hash·evaluation rows는 local-only다. 디스크 W/M checkpoint0이므로 exact crash-resume과 사후 selected-weight 독립 재구성은 NOT_AVAILABLE, GPU off/on continuation은 NOT_TESTED다. 원 source/raw/teacher/실패/삭제 이력과 다른 paused task는 보존했다.

인계 후 agent polling/logtail/sleep loop/heartbeat/callback/자동 추가제출/상세 결과분석0. 이미 제출된 프로그램은 hold/cancel하지 않고 정해진 B1–B10 평가·저장을 자연 진행한다. 후속 상세 리뷰는 명시 USER recall이 필요하다.

Resume `/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/resume-r1/resume-manifest.json`, SHA `dd5b95aa340dd12bcc371b1558592c632bc008494dcc98dcb600f32a0eafd52e`. `monitoring_active=false`, `automatic_resume=false`, `resume_trigger=explicit_user_call`.
NO_BROADCAST_NOT_REQUIRED. Raw/tensor/prompt/teacher/fullstdout Git0. 과학적 우열·효과 인과·최적성 판정은 이 보고 범위가 아니다.

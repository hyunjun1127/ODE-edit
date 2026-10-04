# JLZ v11 2000 edit 제출 인계

## 후속 W5 중간 결과 게시 (2026-10-04)

[repair attempt-r2-cap3의 MAIN/NOALLOC W5 누적500 결과](intermediate-W5-20261004/report-ko.md).
두 경로 W5 평가 저장을 CPU 검산했다. 전체2k 완료나 신규 MEMIT-H 결과는 아니다.
아래는 최초 attempt-r1 제출 시점 기록이며 최신 실행 결과를 뜻하지 않는다.

SH4가 v11 MAIN, NOALLOC, matched native MEMIT-H의 각 cold BS100×20 경로와 작은 pilot, CPU collector를 구현하고 등록했다. 전량 held 검사와 release를 완료했다. 2026-10-04 05:20:55 KST 확인 상태는 5개 모두 PENDING이며 실제 pilot/main 결과는 **NOT_OBSERVED**다. CPU 결과를 GPU PASS 또는 실험 완료로 표시하지 않는다.

## 등록과 source

| 단계 | Job | 선행 dependency | 요청 GPU CPU RAM | 요청 wall |
| --- | --- | --- | --- | --- |
| 독립 cold 작은 pilot 3경로 | 57923 | afterany 57900, 57899 | 1 / 8 / 59392MiB | 4시간 |
| MAIN 2000 | 57924 | afterany 57923 + 프로그램 matching READY | 1 / 8 / 59392MiB | 168시간 |
| NOALLOC 2000 | 57925 | afterany 57924 + 프로그램 matching READY | 1 / 8 / 59392MiB | 168시간 |
| matched native MEMIT-H 2000 | 57926 | afterany 57925 + 프로그램 matching READY | 1 / 8 / 59392MiB | 168시간 |
| CPU collector | 57927 | afterany 57923, 57924, 57925, 57926 | 0 / 8 / 24576MiB | 4시간 |

모든 job은 server4/gpu/QOS lab_gpu_s4, exportNONE/Requeue0이다. Task cap1, project cap2이며 기존 v9 57899/57900/57901은 변경하지 않았다. 요청 wall은 ETA가 아니다.

- 실행 source: `642948727e89e87e8054462ae6c18ac230b9fc25`
- 실행 tree: `b7c5dae4274c42000237bdd1228218d9735f4c19`
- 실행 lock SHA256: `618fec778f15ed3e4967da2ad97da54ba5e56ad38ab892e14ccd5f61db8c956d`
- Config SHA256: `34e9975810b772c919f9c0b6b2e8bd88bdf5a6bea1a536e8b6a58870965d1aab`
- Local attempt: `/data/janghj/ODE-edit/local/jlz-native-increment-v11/20261004-v1/attempt-r1/`
- [제출 receipt](../../../../runs/odeedit_jlz_v11_s4_20261004/submission-r1.server4-server-head.json)

## 검산과 미관측 범위

정본 17개 파일 SHA/size, CSV 2000행의 모든 field, main20/dev2 native pack, 26000 observer token identity를 검산했다. 원 BLUE native 함수의 lookup/token 입력 2004개도 비교했다. Production CPU 회귀 13개와 원 synthetic proof 4개 dimension은 통과했다. Owner의 red 체크리스트이며 별도 독립 reviewer 또는 actual GPU PASS는 아니다.

실제 모델 qualification은 등록된 작은 pilot에서 수행한다. MAIN B100 검산은 B1 본 fit에 포함하며 추가 B100 fit은 없다. MAIN과 NOALLOC은 pulse/replay 없이 고정 entry proxy로 직접 D를 fit하고 terminal actual increment를 쓴다. NOALLOC은 lambda0이며 uniform 배분이 아니다. Native baseline은 원 residual/divisor와 singleton z를 유지한다.

기존 W0의 package byte 및 batching 동등성이 확립되지 않아 이번 허용 범위의 first2000 W0 관측을 한 번 수행하고 세 cold 경로에서 동일 identity로 공유한다. 새 full10k W0 또는 과거 task 재개는 없다. 매 batch pre/post와 W5/10/15/20 누적 raw로 ACC, preference, NLL, paired/cohort 및 비용을 CPU 집계한다. 현재 수치는 NOT_MEASURED다.

## 자원 대기와 종료 경계

Release 직전 기존 v9의 두 job이 각각 1GPU를 사용해 project cap2를 채웠다. Node 전체도 8/8GPU, 479232/512000MiB가 할당된 상태였다. Resource helper의 code4는 pending_resource_cap이며 PASS로 바꾸지 않았다. 새 task를 정확한 기존 resource afterany 뒤에 직렬 배치했으며 v9의 과학적 성공 여부는 gate가 아니다.

인계 시 filesystem free는 94,740,414,464B, task reserve는40GiB였다. 프로그램 시작 시 공간을 다시 확인한다. `save_checkpoints=false`, `exact_resume=NOT_AVAILABLE`; v9의 tensor 저장 예외를 상속하지 않는다. 실패/부분 source와 raw는 삭제하지 않는다.

`INITIAL_NOT_OBSERVED`, `monitoring_active=false`, `automatic_resume=false`로 인계한다. 이후 agent scheduler/log/result polling이나 자동 retry를 하지 않는다. 이미 등록한 sealed runner와 collector만 자연 진행하며, 최종 상세 리뷰는 사용자 recall 때 수행한다.

`NO_BROADCAST_NOT_REQUIRED`: 같은 server 실행이다. Git에는 source/소형 보고/manifest만 게시하고 raw/model/tensor/prompt/fullstdout는 local에 보존한다. Source commit과 이번 인계 문서의 publication commit은 구분한다.

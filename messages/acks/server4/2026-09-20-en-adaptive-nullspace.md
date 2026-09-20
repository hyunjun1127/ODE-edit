# SH4 FULL_READ / M0 — EN adaptive-nullspace migration

Instruction `ODEEDIT-S06-EN-ADAPTIVE-NULLSPACE-B300-SH4-MIGRATION-V1`.
Nonce `ODEEDIT-GH-SH4-EN-ADAPT-MIGRATION-20260920-R1`.
SH3 handoff nonce `ODEEDIT-GH-SH3-EN-ADAPT-MIGRATE-TO-S4-20260920-R1` 수신.

- Authority main `938db0b765e1a3d4a1e6edc8da702d2486ad6e24` 및 migration/inherited envelope, authority 9 documents 전체 읽음. 9개 exact size/SHA CPU 대조 완료.
- SH3 final `43904c13def0aa06bb37dcd7574df5df101db135`, runner `2ff2393f19b74195ed52d016d4d73ae0878d9946` 통합. SH3 보고 GPU submit/allocation/T0/B1/B300=0. 51258은 test-only이며 실제 job 아님.
- S4 session/hostname/root/origin과 등록 `servers/active/server4.md` 결속. PROTOCOL SHA `4209c7d09fb06b81d9f0bfb2b0c86076aa884099bca8be8d9d3e14937f8ffae4`는 이전 exact FULL_READ 재사용.
- S4 기존 R512/Dev128 generated256 640문서 manifest `14bf1f5d98cf9e09e9fad6458bc4d7ae5fb2e81de635b63f2b03db4c405a73b4`와 input SHA `507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb` 재사용. 101.5GB 역전송0. 이번 large model 입력은 prior full-SHA + unchanged stat, small source93개 새SHA. 실제 모델 검증과 구분.
- 기존 S4 P-star basis `[14336,14326] FP64` exact file SHA 대조/CPU mmap 재사용. 신규 eigensolve/SH3 basis 전송0. Shared native B1 fresh100 한 번, 이후 세 arm own-entry native 총7batch/700요청 상한; 과거 approximate factor reconstruction 미사용.
- S3 launcher/READY/Python/evaluator absolute path를 S4 immutable binding으로 port 중. 첫 CPU invocation의 package-relative import 및 S3 evaluator missing은 실행 harness/path 문제로 기록, GPU 실패0.
- Source review 발견: controller quadratic curvature에 actual 내적이 들어가던 것을 설계 §6의 ideal `<G,D1>`로 수리; Armijo actual 내적 불변. 수치/품질 threshold 변경0.
- Project cap2, task 단일1GPU/1persistent job, 8CPU/60416MiB. RAM branch states/noCP, exact crash resume NOT_AVAILABLE. 기존 job/자료 변경0.
- T0→B1 네 arm→N4/EN_EXACT/EN_ADAPT own B2/B3→상세 보고/main까지. B1 성능 부호에 따른 chain 제외0, initial/PENDING pause 상속0. 현 actual T0/B1/B300 미실행.
- 저장공간 CPU preflight 약25.47GiB free(공유FS 시점 관측). 원 reference/모델 재복제 없이 새 compact ledgers/source/report 예비4GiB, edited checkpoint0. 과거 storage waiver 미상속.
- Host planning: reference prefix 약17GiB, state/P/basis 약7GiB, current capture/keys 및 FP64 TSQR/SVD phase workspace 약15GiB, controller/gradient/history 약7GiB, Python/allocator 여유 약6GiB의 phase별 보수 추정 약52GiB. 이는 실측/예약이 아니며 실제 token length/peak를 새 binding과 T0/science에서 기록. Science current oracle는 keys 수집 뒤 SVD 전에 해제하며 전체 column/문서 보존.
- Wall 예비24h는 scheduler 안전 한도이며 GPUh 사용자 hardcap=null. T0/native/512 sweep/weighted QR·SVD/observer 실제 시간으로 보고를 갱신; S3 6h dryrun을 실측으로 쓰지 않음.
- NO_BROADCAST_NOT_REQUIRED. 공유 native/EN/원 source 및 이전 task 유지. 독립 subagent/red 사용0; owner audit+CPU regression.

다음: S4 CPU routing/negative tests, source freeze/resource-only admission, held inspection→release, actual T0/첫 B1 표/최종 B300 보고.

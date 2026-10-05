# CD_Q/CD_C sequential 2k 실행 영수증

Nonce: `ODEEDIT-USER-GH-SH4-CD-CUMULATIVE-2K-20261005-R1`.
Task: `jlz-cd-cumulative-allocation-bs100x20-s4-20261005-r1`.
정본: `plans/global/2026-10-05-jlz-cd-cumulative-allocation/`.

현재 단계는 `SUBMITTED_RESOURCE_PENDING`이다. 두 arm 및 CPU collector의 전량 held 검사/release를 완료했다. 실제 GPU qualification/B1/W20은 아직 관측하지 않았다. 각 arm은 first2000 BS100×20, cold W0/H0, L4–L8, budget .75, max25평가/24update다. 기존 V13/V14 실험은 변경하지 않았다.

| stage | job | 초기 상태 | dependency |
|---|---:|---|---|
| CD_Q | 58880 | PENDING, node 자원/예약 대기 | 없음 |
| CD_C | 58881 | PENDING | afterany:58880 |
| CPU collector | 58882 | PENDING | afterany:58880:58881 |

제출 당시 server4 GPU 8개가 할당되어 있었다. 두 arm을 upfront 등록하고 producer-first resource/scalar dependency를 적용했다. CD_Q의 과학적 성능 PASS는 CD_C의 조건이 아니다. 각 GPU job은 GPU1/CPU8/59392MiB/48h/exportNONE/Requeue0, CPU collector는 GPU0/CPU8/24576MiB/4h다. 기존 job mutation은 0이다.

실행 source는 `abead2333c30cb57ea10ca9756a21f765f8dbc29`, tree는 `6f329ac5b15140606c1a3d7a90e578612567a700`이다. Source archive 178개 파일/1884160 bytes, SHA256 `1910b5fa53e1e8920de152b3af52f2cb3d5174fb65c46d3680c84b2c4e2b3eae`. Config SHA256 `d6b374a5dab0b9224b04a1398ff41f1e4776809237575e00dd958eb1804297ed`, execution lock SHA256 `24e1c53dd00e56c712d3d78f802b91d2acee8672db1e9a25138ef2b607de83c5`이다.

원 CPU11/design `PASS_WITH_WARNINGS`는 합성 설계 근거로 재사용했다. 새 production CPU 회귀 31개가 통과했다(geometry5/fit9/collector13/controller4). CPU tiny-model 검산은 target Llama GPU 검증이 아니다. 첫 preflight의 receiver receipt 상수 NameError를 제출 전 수리했고 원 CPU receipt r1/r2를 local에 보존했다. 최종 r3 receipt와 실제 archived source SHA의 일치를 확인했다. Source cross-review의 실제 reviewer 수준은 `audits/servers/server4/jlz-cd-cumulative-2k/source-review-r1.json`에 구분했다. 실제 target GPU 검산은 sealed job에서 main 외 2개 native 요청의 same-candidate check만 수행하며 fit/update는 없다. 기술 READY 뒤 main B1부터 cold chain을 진행한다.

정본 11 member+manifest/envelope SHA, first2000 CSV 모든 행/field, 20 pack/14000 native row의 owner/role/lookup/canonical, 26000 observer row identity를 결속했다. 동일 runtime/source의 기존 first2000 W0 raw는 CPU identity 검산을 거쳤고, 새 cold model W/H SHA 확인 뒤에만 조건부 재사용한다. 두 arm의 fitted state는 재사용하지 않는다.

산출물 예정 경로: `/data/janghj/ODE-edit/local/jlz-cd-cumulative-allocation/20261005-r1/attempt-r1/`.
Raw/tensor/model/prompt/fullstdout는 local KEEP, checkpoint/복원 bundle은 저장하지 않는다. `exact_resume=NOT_AVAILABLE`.
Artifact broadcast: `NO_BROADCAST_NOT_REQUIRED`; 이 단계는 source/compact receipt만 공유하며 대형 raw는 전송하지 않는다.

최종 W20 지표/분모는 아직 `NOT_OBSERVED`다. CPU collector가 각각 20commit/19join/100H 및 R2000/P4000/N20000의 실제 저장 raw를 확인한 뒤 complete/partial/technical-blocked를 구분해 사실 보고서를 작성한다.

이 등록/초기 resource-pending snapshot 뒤 `monitoring_active=false`, `automatic_resume=false`다. 추가 agent polling/heartbeat/Slurm retry 없이 sealed runner/collector가 진행한다. 상세 결과 리뷰는 사용자 recall 시 수행한다.

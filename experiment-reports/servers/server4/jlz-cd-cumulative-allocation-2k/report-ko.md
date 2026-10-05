# CD_Q/CD_C sequential 2k 실행 영수증

Nonce: `ODEEDIT-USER-GH-SH4-CD-CUMULATIVE-2K-20261005-R1`.
Task: `jlz-cd-cumulative-allocation-bs100x20-s4-20261005-r1`.
정본: `plans/global/2026-10-05-jlz-cd-cumulative-allocation/`.

현재 단계는 `REPAIR_R2_SUBMITTED_RESOURCE_PENDING`이다. 원 attempt-r1은 아래 실패 기록으로 보존했고, 사용자 `fail되었으니 repair올려`에 따라 새 cold attempt-r2의 두 arm 및 CPU collector를 held 검사 후 release했다. 새 GPU qualification/B1/W20은 아직 관측하지 않았다. 각 arm은 first2000 BS100×20, cold W0/H0, L4–L8, budget .75, max25평가/24update다. 기존 V13/V14 실험은 변경하지 않았다.

## 원 attempt-r1: 0-commit 기술검산 실패

| stage | job | 초기 상태 | dependency |
|---|---:|---|---|
| CD_Q | 58880 | FAILED 1:0, 본선 commit 0 | 없음 |
| CD_C | 58881 | FAILED 1:0, 본선 commit 0 | afterany:58880 |
| CPU collector | 58882 | Slurm COMPLETED, scientific PARTIAL_OR_TECHNICAL_BLOCKED | afterany:58880:58881 |

사용자 recall의 exact accounting snapshot에서 Q166초/C159초, 합325 GPU초를 확인했다. extern/batch 행을 중복 계상하지 않는다. 실패는 narrow qualification의 all-row physical regrouping gradient parity에 한정되었다. 두 arm에서 같은 수치였고, 기존 owner별 shape의 cached/full·projected/direct 및 독립 dense CD 검산은 통과했다. 별도 전체 B fit/물리 write/commit은 없었다. 원 physical W/H/cache/RNG 비변이 기록은 통과했지만 이것을 본선 또는 전체 GPU qualification PASS로 해석하지 않는다. 원 source/raw/receipt/log/비용은 local KEEP이다.

제출 당시 server4 GPU 8개가 할당되어 있었다. 두 arm을 upfront 등록하고 producer-first resource/scalar dependency를 적용했다. CD_Q의 과학적 성능 PASS는 CD_C의 조건이 아니다. 각 GPU job은 GPU1/CPU8/59392MiB/48h/exportNONE/Requeue0, CPU collector는 GPU0/CPU8/24576MiB/4h다. 기존 job mutation은 0이다.

실행 source는 `abead2333c30cb57ea10ca9756a21f765f8dbc29`, tree는 `6f329ac5b15140606c1a3d7a90e578612567a700`이다. Source archive 178개 파일/1884160 bytes, SHA256 `1910b5fa53e1e8920de152b3af52f2cb3d5174fb65c46d3680c84b2c4e2b3eae`. Config SHA256 `d6b374a5dab0b9224b04a1398ff41f1e4776809237575e00dd958eb1804297ed`, execution lock SHA256 `24e1c53dd00e56c712d3d78f802b91d2acee8672db1e9a25138ef2b607de83c5`이다.

원 CPU11/design `PASS_WITH_WARNINGS`는 합성 설계 근거로 재사용했다. 새 production CPU 회귀 31개가 통과했다(geometry5/fit9/collector13/controller4). CPU tiny-model 검산은 target Llama GPU 검증이 아니다. 첫 preflight의 receiver receipt 상수 NameError를 제출 전 수리했고 원 CPU receipt r1/r2를 local에 보존했다. 최종 r3 receipt와 실제 archived source SHA의 일치를 확인했다. Source cross-review의 실제 reviewer 수준은 `audits/servers/server4/jlz-cd-cumulative-2k/source-review-r1.json`에 구분했다. 실제 target GPU 검산은 sealed job에서 main 외 2개 native 요청의 same-candidate check만 수행하며 fit/update는 없다. 기술 READY 뒤 main B1부터 cold chain을 진행한다.

정본 11 member+manifest/envelope SHA, first2000 CSV 모든 행/field, 20 pack/14000 native row의 owner/role/lookup/canonical, 26000 observer row identity를 결속했다. 동일 runtime/source의 기존 first2000 W0 raw는 CPU identity 검산을 거쳤고, 새 cold model W/H SHA 확인 뒤에만 조건부 재사용한다. 두 arm의 fitted state는 재사용하지 않는다.

산출물 예정 경로: `/data/janghj/ODE-edit/local/jlz-cd-cumulative-allocation/20261005-r1/attempt-r1/`.
Raw/tensor/model/prompt/fullstdout는 local KEEP, checkpoint/복원 bundle은 저장하지 않는다. `exact_resume=NOT_AVAILABLE`.
Artifact broadcast: `NO_BROADCAST_NOT_REQUIRED`; 이 단계는 source/compact receipt만 공유하며 대형 raw는 전송하지 않는다.

원 attempt-r1의 W20은 `NOT_OBSERVED`이고 metrics 분모를 0점으로 채우지 않는다. 원 collector는 0-commit 실패를 부분/기술차단으로 집계했다.

## 새 attempt-r2: original-shape reference repair

| stage | 새 job | bounded 초기 상태 | dependency |
|---|---:|---|---|
| CD_Q | 58913 | PENDING Priority | 없음 |
| CD_C | 58914 | PENDING Dependency | afterany:58913 |
| CPU collector | 58915 | PENDING Dependency | afterany:58913:58914 |

실패한 forward 물리 regrouping은 `NOT_QUALIFIED_NOT_USED`로 유지했다. 원 owner graph의 tokens/masks/cache/shape를 그대로 유지하고, group별 backward 누적을 같은 graph들의 독립 logical SUM 1회 backward와 비교한다. 전체 S의 off-owner pullback을 유지하고 elementwise gradient 기준 `1e-6+2e-4*abs(reference)`를 바꾸지 않았다. Main도 1-owner physical grouping을 강제한다. 원 실패를 PASS로 다시 표시하지 않으며 새 실제 GPU 검사 PASS가 없으면 main 진입을 차단한다. RCA 한계와 좁은 변경은 `plans/updates/server4/jlz-cd-cumulative-2k/repair-r1.md`에 기록했다.

생산 CPU 회귀 48개(geometry5/fit9/collector13/controller15/qualification6)가 통과했다. 독립 read-only 변경 소스 red review의 확인된 blocker는 0이었다. 구현 worker의 자체 tests와 독립 검토 수준을 `audits/servers/server4/jlz-cd-cumulative-2k/repair-r1/source-review.json`에 구분했다. CPU synthetic/tiny 검산은 actual Llama GPU PASS가 아니다. Qualification의 실제 call 상한은 narrow2-request original-shape G=2에서 F9/B7/prefix0 (operator-compatible 기준), optimizer/fit/physical write 0으로 봉인했다. 신규 B100 fit은 각 main B1 자체뿐이다.

새 실행 source `8a2c1cf0e13534676c492c36229d792122748642`, tree `72efa7c7d690c57fb11561208ea05ef2be43f554`, archive 180 regular files/1925120 bytes/SHA256 `5f3aa53b9b3d5ac6afd16b50caa6773336ae1cfbb49ff9ab2dcd5c4883f05865`다. Config SHA256 `681cc0a6e6c1b114db34d048fb74221f7e29be9ed058cc32b42652c953645134`, lock SHA256 `de67487829e931777e02491329565fd5fb6813809da08bef28aab4cd402a9101`이다.

새 attempt: `/data/janghj/ODE-edit/local/jlz-cd-cumulative-allocation/20261005-r1/attempt-r2/`. 기존 model/data/C0/context/runtime/token/evaluator 및 identity-qualified W0 관측만 read-only 조건부 재사용하며, 이전 fitted W/H/u/optimizer 상태는 전달하지 않는다. 별도 새 calibration fit은 없고 새 main B1의 immutable lambda lock만 두 arm이 공유한다. 원 geometry/optimizer/calibration/adapter/writer/collector algorithm bytes는 불변이다.

신규 두 GPU lane을 전량 upfront 등록·release했다. Fresh effective projectcap3/taskcap2, physical capacity 및 scalar provenance 때문에 producer-first `afterany`로 최대 동시1GPU를 사용한다. 이는 성적 PASS dependency가 아니다. 각 GPU1/CPU8/59392MiB/48h와 CPU collector GPU0/CPU8/24576MiB/4h/exportNONE/Requeue0를 검산했다. Disk reserve12GiB, 준비 시 free 약66.5GiB; host/VRAM 본선 peak는 여전히 사전 estimate이고 tiny original qualification의 peak를 B100 PASS로 승격하지 않는다. 기존 job mutation/cancellation은 0이다.

W20은 새 attempt에서도 `NOT_OBSERVED`다. Sealed qualifier→cold20batch→CPUcollector가 새 actual raw의 20commit/19join/100H/arm, R2000/P4000/N20000을 집계하여 complete/partial/technical-blocked를 구분한다. 신규 baseline/추가 fullB fit/solver·계수·precision·허용오차 변경은 없다. NoCP/exact_resume NOT_AVAILABLE; raw/tensor/prompt/fullstdout는 local KEEP, compact source/receipts만 Git이다.

이 등록/초기 resource-pending snapshot 뒤 `monitoring_active=false`, `automatic_resume=false`다. 추가 agent polling/heartbeat/Slurm retry 없이 sealed runner/collector가 진행한다. 상세 결과 리뷰는 사용자 recall 시 수행한다.

# CD cumulative repair r1 — 동일 native shape reference 경로

사용자 recall: `fail되었으니 repair올려`.
부모 nonce: `ODEEDIT-USER-GH-SH4-CD-CUMULATIVE-2K-20261005-R1`.

원 attempt-r1의 CD_Q 58880/CD_C 58881은 `SAME_CANDIDATE_QUALIFICATION_FAILED`로 종료했다. 두 arm 모두 본선 fit/commit 0이다. Collector 58882는 실패자료를 집계하고 `PARTIAL_OR_TECHNICAL_BLOCKED`를 봉인했다. 부모 GPU 할당 시간은 166+159=325초이며 extern/batch를 중복 계상하지 않는다. 원 source `abead2333c30cb57ea10ca9756a21f765f8dbc29`, archive/config/lock/raw/stdout/qualification/CPU 결과는 그대로 보존한다. 재개 checkpoint는 없으며 새로운 독립 cold attempt-r2로 실행한다.

## 확인된 범위와 RCA 한계

두 arm의 오류 수치가 동일하다. 실패는 원래 요청별 7-row graph 두 개를 한 14-row model batch로 합친 `logical_SUM_microbatch.gradient`에 한정된다. 층 L4–L8의 실패 원소 수는 각각 63/48/53/50/27 (층마다 8192 원소), 최대 절대 오차는 2.60e-5 이하이다. 고정 elementwise 기준은 `1e-6+2e-4*abs(reference)`였고 이를 초과했다. 같은 검산의 loss/logprobs/subject, 원 shape의 cached/full gradient, projected/direct gradient, 독립 CD dense/compact solve/action/energy/u-gradient는 통과했다. Physical W/H/cache/RNG 비변이 검사가 통과했으며 물리 write/fit/update는 없었다.

생산 fit은 이미 `prepare_entry(...,1)`의 원래 complete-owner graph를 사용하고 전체 D/S를 유지해 모든 group의 SUM gradient를 누적한다. 한 14-row batch로의 물리 regrouping은 qualifier에만 있었고 생산에서 사용하지 않는다. 관측된 차이는 FP32 batch-shape/GEMM 계산 순서 변경과 일관되지만, 이 CPU/source 검토만으로 CUDA kernel별 원인을 확정하지 않는다. 입력 누락이나 목적 SUM 배율 오류는 소스/기록에서 발견되지 않았다. 원 실패한 regrouping을 PASS로 재표기하지 않는다.

## 좁은 reference fallback

실패한 all-row model regrouping은 `NOT_QUALIFIED_NOT_USED`로 명시한다. 새 실제 검사에서는 같은 원 tokens/masks/lookup/cache/forward shape로 요청별 graph를 각각 구성하고, 모든 owner loss를 합한 뒤 한 번 backward하는 독립 logical-SUM reference와 생산 group별 누적 backward를 비교한다. Forward grouping은 바뀌지 않고 전체 cross-owner S pullback 및 요청 SUM은 유지한다. 새 qualifier와 main은 complete-owner 1-request graph 경로를 강제한다. 이 검사는 별도 fit/optimizer update/physical write가 아니며 같은 고정 분석 candidate의 narrow 2-request 기술 검사이다. 실제 GPU 통과 전에는 READY/PASS를 주장하지 않는다.

원 elementwise gradient/action/solver 허용오차, native 손실, shared .75 budget, EfficiencyAdam, calibration ratio1, 25평가/24update, 공동 stop, frozen whole-B geometry, CD 실제 commit/H 정책, 두 arm 각20 batch/2000 요청은 바꾸지 않는다. P/N/성적을 수리 또는 경로 선택에 사용하지 않는다. 새 qualification의 물리 F/B 예산은 코드/config에 사전 봉인하고 실제 호출과 구분한다.

## 등록·보존 경계

새 전용 non-main WT에서 CPU 회귀, source 검토, SHA freeze 후 정확 predecessor receipt로만 cold repair를 등록한다. 기존 동일 task의 terminal owner/source/WorkDir/등록 ID와 0-commit을 재확인하며, 불명한 등록 또는 active job은 중복 submit을 차단한다. 기존 job cancel/hold/release/hotpatch는 하지 않는다. Shared scalar provenance를 위한 producer-first dependency는 성적 PASS가 아닌 기술 READY/자원 경계다. 두 새 arm과 afterany CPU collector를 전량 held 검사 후 release한다.

자원은 각 GPU1/CPU8/59392MiB (server4 hard60416)/48h, collector GPU0/CPU8/24576MiB/4h이며 현재 더 엄격 cap을 준수한다. 원 scientific 문서·공유 native/env는 불변이다. 새 W/H/u/D/optimizer 등 checkpoint/복원 bundle을 저장하지 않는다. 원 raw는 local KEEP, source/소형 보고만 Git에 게시한다. Formal resource-pending 초기 인계 후 모니터/heartbeat/자동 retry는 다시 멈춘다.

# JLZ v9 ridge 2k 실행 인계

## 후속 저장 결과 게시 (2026-10-04)

[W20 결과·실행 종료 구분·baseline 비교·소형 산출물](results-publication-20261004/report-ko.md).
A는 20commit/W20 저장, Slurm FAILED(0:11); B는 18commit 후 취소로 W20 없음.
아래는 원 제출 시점 기록이며 현재 결과를 뜻하지 않는다.

현재 `SUBMITTED_RESOURCE_PENDING`이다. A/B 각각 BS100×20=2000, 두 경로 합4000 edit occurrences이며 이전500과 별도 cold W0/H0 실행이다. 새로운 성능 수치는 아직 없다.

| 역할 | job ID | 마지막 한정 관측 | dependency |
|---|---:|---|---|
| main A | 57899 | RUNNING | 없음 |
| main B | 57900 | PENDING Resources | 없음 |
| CPU collector | 57901 | PENDING Dependency | afterany:57899:57900 |

세 job 모두 owner/name/argv/source/GPU/CPU/memory/wall/export/requeue/dependency를 held 상태에서 검사하고 release했다. A의 cold W0/import 결속은 확인했으며 B1 commit→B2 ownentry는 아직 관측하지 않았다. `INITIAL_NOT_OBSERVED`이며 초기 actual PASS가 아니다. 전체 release 후 노드 GPU8/8 할당과 B의 `Resources` 대기를 결속한 resourcepending 인계다. `monitoring_active=false`, `automatic_resume=false`; 이미 등록한 프로그램만 W20까지 자연 진행한다.

## 실행 봉인

- 실행 source `ab6f1bda4688ffc275dd5ff47649899d5a80093c`.
- Lock SHA `b83fcaf0dec643ff791be03d6aead31d89253026e6aaf5c6ce5e05e8386943c4`.
- Config SHA `633b8e27a063e6518bc6e7df0da5dcb65029146a9f01ecf5e2c51ca093090f6b`.
- Archive SHA `ad3a21182407711454cbeb5725dc34573b8fac31a41c7b5482021139ecff7d66`.
- Local attempt `/data/janghj/ODE-edit/local/jlz-realization-v9/20261004-2k-v1/attempt-r2/`.
- 각1GPU/8CPU/59392MiB(58GiB)/48h, exportNONE/Requeue0/server4. Collector0GPU/8CPU/24576MiB/4h. Task/project cap2, 별도 두 lane이며 실제 동시할당은 보장하지 않는다.

최초60416MiB 요청은 현행 Slurm58GiB 상한으로 등록 거부됐고 job0/GPU초0이었다. 57.42GiB host 계획을 유지하며 요청만59392MiB로 조정했다. 다음 held 검사에서는 `NumCPUs=8-14` 표기를 기존 exact 문자열 검사기가 거절했다. 실제 `CPUs/Task=8`/`ReqTRES cpu=8`을 검산하도록 수정하고 held57899를 그대로 유지하여 나머지만 등록했다. 취소/중복fit/봉인archive hotpatch는0이다. 등록검사 수리 source `e61e94cf`와 과학 실행 source를 구분한다.

원 v9 core25 파일은 frozen `b8c4c96f`와 같다. 20-batch controller·milestone current/allseen·collector·noB21 검산을 새 namespace에 구현했고 CPU22 tests가 통과했다. 기존 actual Q1 reuse bridge와 새로운 GPU 검증을 구분했다. 원 Q1 B의 종료기록도 보존했다.

W5/W10/W20 allseen 및 current, TF/NLL/분모·paired/cohort·비용·계층 실현 진단을 sealed CPU collector가 생성한다. 두 arm main20와 source/config identity가 모두 맞아야 `COMPLETED_2000_BOTH`다. 부분 실패를0점이나 전체완료로 집계하지 않는다.

기존first2000 BASE_ALPHAEDIT/BASE_MEMIT/MEMIT-H W20 raw는 역사 비교로 결속했다. AlphaEdit-BLUE W20은 `NOT_AVAILABLE`이다. 신규 baseline fit/exact probe/전체W0 forward/checkpoint는0이다.

## 소형 증거 및 산출물 경로

- [제출 mapping](../../../../runs/odeedit_jlz_v9_2k_s4_20261004/submission-r2.server4-server-head.json)
- [CPU 검산](../../../../audits/servers/server4/jlz-realization-v9-ridge-bs100x20-s4-20261004-v1/CPU-preflight.json)
- [입력·Q1·runtime 재사용 manifest](../../../../audits/servers/server4/jlz-realization-v9-ridge-bs100x20-s4-20261004-v1/input-reuse-manifest.json)
- [구현·자원 계획](../../../../plans/updates/server4/jlz-realization-v9-ridge-bs100x20-s4-20261004-v1/implementation-ko.md)

완료 후 local `attempt-r2/report/`에 metrics/comparison-W20/paired-cohorts/tails/candidate-summary/terminal-realization/terminal-context-decomposition/cost/accounting/artifact-index가 생성되는 구성이다. 아직 생성된 최종값으로 주장하지 않는다. 보고와 artifact index 작성 후 collector terminal을 봉인한다. 상세 완료 리뷰는 사용자 recall에서 수행한다.

raw/tensor/prompt/fullstdout는 local KEEP/Git0. NO_BROADCAST_NOT_REQUIRED. 기존 repo 보고 형식과 한국어 factual-only 경계를 유지했다.

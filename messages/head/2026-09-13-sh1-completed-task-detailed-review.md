# SH1 — E01 Middle/Late full-seen 완료 상세 리뷰
Instruction ID: ODEEDIT-S06-E01-MIDDLE-LATE-FULLSEEN-DETAILED-REVIEW-SH1-V1
Nonce: ODEEDIT-GH-SH1-E01-COMPLETED-DETAILED-REVIEW-20260913-R1
Target: SH1 server1/devbox session01a04939-f93a-7b50-bca0-65438eab2062, CWD /mnt/raid5/janghj/ODE-edit, repo hyunjun1127/ODE-edit.
사용자 최신 명령: "server1과 4 task 모두 완료되었으니 자세히 리뷰시켜라".
이는 해당 완료 작업의 명시적 recall이며 CPU 결과검산/상세보고/main 통합을 허용한다. 새 GPU 실험 권한은 아니다.
원문 지시·최신 PROTOCOL·실행 lock을 정독하고 새 analysis worktree/branch/output에서 진행한다. 실제 main/source/tree/dirty를 분리 기록하고 공유 dirty/기존 raw/실패 attempt/타 task를 보존한다.
정확한 지정 job만 단발 scheduler 확인한다. 사용자 완료 통지와 terminal-valid를 구분하고, 아직 실행 중이면 해당 부분을 NOT_TERMINAL로 남긴다. 반복 polling/대기/다른 task 모니터링0.
Slurm submit/requeue/cancel/hold/new GPU/model forward/evaluator/replay/native write 권한 NOT_ALLOWED. Resource cap2 유지; CPU tensor reload·재해시·독립 reducer·저장된 수치의 분석/그림 재생성은 ALLOWED.
미측정 endpoint/지표를 새 평가하거나 인접 state/분모로 채우지 않는다. technical failure는 정확 범위·원인·비용·필요 최소 보완을 보고하되 GPU rerun 자동제출0.
기존 완료한 유효 검산은 manifest/identity로 재사용하고 동기검증을 핑계로 전체 이전 작업을 다시 하지 않는다. 이번 신규 결과의 provenance/분모/상태/계측 coverage는 책임지고 검사한다.
복잡하고 독립적인 raw reducer/상태·계약 감사만 bounded subagent를 활용하고 단순 체크/수정/보고는 직접 수행한다. GH는 이를 중복 raw/GPU 감사하지 않는다.
Red preflight는 analysis source/input allowlist/일치하는 endpoint·평가 semantics, postrun은 independently reduced counts/NLL/transitions/누락/비용/원문 coverage를 확인한다. 허위 비교·수치 오류는 해당 비교 HOLD; 유한 저성능/비수렴/비용초과는 사실로 유지한다.
한국어 SH 보고에는 사실/수치/분모/paired 대비/오류/반대 근거/미검증만 기록한다. 최종 원인 해석·선택·허용 claim은 GH 종합 소유다.
Raw/checkpoint/prompt/full logs/tensors는 local-only. PNG는 직접 작성한 코드 실행으로 생성하고 CSV/코드/환경/hash 및 재현 명령을 기록한다.
이번 review raw broadcast는 NO_BROADCAST_NOT_REQUIRED. 기존 local/imported/봉인 Git 자료만 사용하며 원격 새 raw 전송은 필요 범위/path/SHA를 보고한다. 별도 기존 승인 없는 포괄 rsync나 삭제0.
완료된 본인 범위 source+raw-free 보고만 최신 main 포함 여부 확인 후 clean integration/nonforce main push한다. 다른 SH 커밋을 임의 중복merge하거나 미완료 runtime을 실행완료로 올리지 않는다.
첫 실제 결과표는 전체 분석 완료를 기다리지 않고 전달한다. 다음 상세 report/manifest/receipt와 최종 main HEAD/tree/report SHA를 보고하고 TASK_COMPLETE_STOP. 작업 중 단계별 재승인 질문은 불필요하다.

## 지정 결과와 lineage
- 신규 observation-only jobs46439 Middle /46440 Late: source58f50a25809779918b22ad0aceded732c097eab4 tree1c2b49a75ee7e39e03d51346fb35ccb8b0a602cf.
- WT /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-fullseen-schema-repair-r3
- root local/baseline-mechanism-first-e01/20260912-v1/fullseen-schema-repair-r3.
- 재사용 native jobs45914/45915 각각 B051–60/B091–100의10write 및 terminal Current100. 두 job은 original seen-full request_order KeyError로 FAILED였음; allocation9727/9748GPU-sec.
- 새 job은 past5900/9900 observation만 보완한다. native20batches/z/history 반복0이었는지 실제 counter로 구분.
- 실행 source b51dcf5 계열 → schema repair58f50a 실제 diff를 읽고 repair가 native equation/target/history/precision/tolerance를 바꾸지 않았는지 감사.
- 원본 비교는 같은 layer/same fixed10k의 AlphaEdit_BLUE_L4_ONLY 실제 W60/W100이다. 5layer native AlphaEdit, 역사적 다른stream baseline, 다른 entry 점수를 대체하지 않는다.
- 이전 E01 cold45719/warm45805 및 E1-A10arm CPU fact packages는 완성 검산 재사용. AOS45633 완료 리뷰는 이번 재실행/재감사 대상 아님.

## 필수 사실 표와 검사
1. 단발 job terminal/exit, incomplete shard/중복/누락/finite, 정확한 final endpoint W/M/order/contexts와 guard/restore 확인.
   repair first128/1664pairs 성공을 전체완료로 대신하지 않는다. 전체 parameter byte 보존은 실제 기록범위만 주장.
2. 각 endpoint에서 Current100과 actual fullseen6000/10000을 분리한다. Current rows reuse와 old5900/9900 결합을 검산해 중복분모0.
   원본 raw의 header 부재는 case/prompt/target identity로 처리하고 row-index-only pairing0; canonical true/new NLL과 ties=failure 유지.
3. 원본→replay RS/PS/NS counts/denominators/pp; new/true NLL·desired margin 평균/중앙/p90/p95/p99; strict/token secondary.
   전체 및 Current/기존entry-old5000또는9000/이번10batch-new1000/cohort/active-superseded/overwrite 전이를 구분.
   lost/gained/unchanged, at-write 대 final 손실과 legitimate replacement를 가능한 관측으로 연결한다. checkpoint 사이 정확 최초실패시점은 추정0.
4. W/M abs/relative Frobenius/maxabs/changed fraction 등 실제 저장으로 계산 가능한 차이와 성능차이를 나란히 보고한다.
   성능유사=weight parity로 승격0, weight nonexact=성능붕괴로 단정0. 기존 NONEXACT_CAUSE_UNRESOLVED를 hardware로 추정하지 않는다.
5. Target→write→key/token perturbation→출력 반응 E1-B 계측에서 실제 저장된 signed derivative/FD/query/general/history/conditioning을 연결한다.
   잘라낸 progress_slope를 signed derivative로 바꾸어 부르지 않는다. 계측이 빠졌으면 NOT_RECORDED.
6. E0 no-op/next batch/full native continuation/state·performance fidelity 수준을 구분한다.
   E01 계획20cell/E1-A/E1-B 중 이번까지 실제 완료/partial/not-run coverage matrix를 갱신한다.
   이번 Middle/Late task완료와 전체E01완료를 혼동하지 않는다. 미실행 layer/cell/new GPU 확대0.
7. load/restore, native z/key/solve/history, diagnostics/spectrum/F-B, canonicaleval/I-O와 allocated GPU-sec를 분리.
   45914/45915 native 및 실패비용,46439/46440 eval보완, 과거 superseded45908/45913 등 관련시도 cost를 lineage에 보존하되 재사용 항목 중복합산0.
   initial elapsed를 전체 cost로 사용0.

## 허용 write/산출물
전용 codex/server1-e01-completed-detailed-review-20260913-v1.
project/run_scripts/baseline_mechanism_first/ 내 analysis/reducer/report/tests만 허용; 기존 runtime launcher/native math 변경0.
audits/servers/server1/2026-09-13-e01-middle-late-review/
experiment-reports/servers/server1/baseline-mechanism-first-e01-2026-09-12-v1/completed-middle-late-review-v1/
messages/acks/server1/2026-09-13-e01-middle-late-review.md
messages/server-heads/server1/2026-09-13-e01-middle-late-review.md
tasks/status/server1/2026-09-13-e01-middle-late-review.json
runs/e01-middle-late-review-20260913-v1/ (small metadata only).
최종 diagnostic-report-ko.md, source/input/evidence reuse manifest, endpoint-performance.csv, paired/cohort/distribution tables, weight-history-differences.csv, coverage.csv, compute-summary.csv, code PNG, rooted receipt.
GH가 원본 대비 성능과 재현성을 판단할 수 있도록 결과 전체/제한을 제출하라. 새 실험 제안은 최소 미검증 항목으로만 기재하고 자동 실행하지 않는다.

# SH4 — odeedit_lowcost_seq10_s4 완료 상세 보고
Instruction ID: ODEEDIT-S06-LOWCOST-SEQ10-COMPLETED-DETAILED-REPORT-SH4-V1
Nonce: ODEEDIT-GH-SH4-LOWCOST-SEQ10-REPORT-20260914-R1
사용자 2026-09-14: "odeedit_lowcost_seq10_s4 끝난거 자세히 report 만들도록 명령하자."

## 1. Recall/권한
Target SH4 server4/session01a04939-b5c7-7a03-ba2d-ef3343d62cfd; CWD /data/janghj/ODE-edit; repo hyunjun1127/ODE-edit.
이 명시 recall로 완료 결과의 CPU 검산·상세 한국어 report·검증된 본scope source/report main 통합을 재개한다.
별도 clean codex/server4-lowcost-seq10-completed-review-20260914-v1 worktree와 새 analysis output을 사용한다.
공유 dirty, 실행 source/archive, raw, 기존 core/report/checkpoint와 타task를 수정·삭제·덮어쓰기하지 않는다.
원 실행 지시 messages/head/2026-09-13-sh4-lowcost-sixarm-seq10.md 및 execution-amendment.json, 원 pilot design/contract와 PROTOCOL 전체를 읽는다.
현재 main1c379d9c3194460d6c9a1c4a3a5d513fadec5ba7와 실제 fetch main/source/tree/dirty를 분리 pin.
GH는 raw/GPU 중복 감사하지 않는다. 복잡한 독립 reducer/상태 감사만 bounded blue/red subagent, 단순 확인은 직접.

## 2. 정확한 대상
Array46475_[0-5], mapping0=N4,1=RES8,2=S875,3=S75,4=FULL8,5=REFIT4.
Execution5e96dcb3745977b1f273e3f5afbee61167248d49/tree6e9f8bdb432392fd5f0d7d0937ff7058666944c0.
Archive3dadca9463ab131846bff5227fe1bbbb540dd5b8f238bfad53b15b121ed0eb42,
execution.lock695a2d985d5abb8fae1cc6f0b1933022895a47a71142aa032a9da3b6be1bbb28.
Control/provenance tip93c3e4f와 execution은 구분한다.
Root /data/janghj/ODE-edit/local/low-cost-write-donor-seq10/20260913-v1/attempt-v1/
output/cell-{0..5}/; initial-pause-receipt SHA7cd43361563a2088ebefb8040e76d4ab01f9c3688eb164542f6ff949f02c9ee8.
Sample-sequential.lock SHA1f57f8888bb13801f0748081acb61c1de448adb5bf04b6a9d02aef952a63d3bb.
이전에 agent가 확인한 actual initial은 N4 B51→B52뿐이다. 다른5arm initial을 과거 관측했다고 소급하지 않는다.

## 3. 허용 실행과 무결성
지정6job만 단발 scheduler/terminal 확인. 사용자의 완료 통지와 6/6 scientific terminal-valid를 구분한다.
미완료라면 해당cell/prefix/미측정을 표시하고 반복 polling/대기0.
새 GPU/model load/forward/evaluator/native write/history replay/Slurm submit·cancel·requeue·throttle 변경은 NOT_ALLOWED.
cap2는 그대로 유지. CPU weights_only reload/hash/저장된 algebra/독립 NLL reducer/그림 재생성 ALLOWED.
기술 실패/미측정이 있으면 원인·영향·필요 최소 보완을 보고하고 자동 rerun0.
기존 검증된 inputs는 exact manifest로 재사용; 이번60batch/18CP 예상 대비 실제 inventory와 신규 output을 검사.
File SHA/size, selected tensor dtype/finite/shape/hash, target/order/source/context/RNG, W/M commit→next-entry 54 links 예상, no duplicate/imputation을 확인한다.
Checkpoint51/55/60와 journal 가용성, 실제 restore 검증 범위를 구분한다. Hash-only를 재구성가능 checkpoint라 하지 않는다.
CPU 검산과 GPU continuation/model-level off-on parity를 구분하고 NOT_TESTED 항목을 숨기지 않는다.

## 4. 순차 실행이 실제 이루어졌는지
공통 W50/M50/preparedM8/고정10k identity에서 출발했는지, B51[5000,5100)부터B60[5900,6000)까지6arm 동일order를 대조한다.
각 arm 매batch 자기현재W/M에서 fresh L4 D4, alpha고정, 해당partialstate에서freshsecondfit인지 확인한다.
N4/S875/S75/FULL8/RES8/REFIT4의 기본정책과 실제 계측이 일치하는지; sharedbatchD4/z 재활용·매batchW50reset·cross-armstate유출0.
두fit 사이 currenthistory append0, final선택layer당append1. REFIT4 L4두번append0.
Expected fullscope fit/solve90/request-z9000, historyappend L4총60/L8donor총20을 실제counter와 대조; 누락counter 추정0.
공통M8 준비재사용과 각donor의 append-only 누적, P4asset0/P8asset4/singleton0 mapping을 기록한다.
평가가 actualendpoint에서 수행됐고 W/M/RNG를 바꾸지 않았는지, 후속write가 평가 전과 같은state에서 이어졌는지 확인한다.
기존staticcore job46451의 B51 결과와 이번각arm 첫batch를 identity가 맞는 범위에서 CPU 비교한다.
변경이 있으면 source/state/target/NLL 차이를 수치로 쓰며 quiet tuning 또는 historical parity 주장0.

## 5. 첫 표를 먼저 보고
전체 검산 완료 전이라도 검증된 final6arm 표를 먼저 전달하되 검증수준/미완료를 명시한다.
actual W60 fullseen6000의 RS/PS/NS (expected d6000/12000/60000), N4대비 pp/count 차이를 주표로 둔다.
Current B60 d100/200/1000, suffix-new1000 d1000/2000/10000, oldprefix5000 d5000/10000/50000를 별도표로 분리한다.
B55 suffix500, 고정Historical128, Wiki128, MMLUdev32 최종값도 별도표. 서로 다른state/population을 합쳐 denominator 늘리지 않는다.
Fullseen에서 reuse한 Current/suffix rows를 독립평가로중복합산하지 않는다.
W60 fullseen6000은 기존 B1–B50 상태+새1000개이며 새6000 edits/full10k/6만unique실험이 아니다.

## 6. 상세 분석용 사실·표·그림
- 전60 batch Current RS/PS/NS와 NLL/margin/strict/token; fixedHistorical R/P/N, Wiki mean 및 token수, MMLU alternative correct/invalid 추이.
- B55/B60 suffix retention, at-write-success→laterfailure/recovery, ALL 및active/superseded/legitimate overwrite 분해.
- 최종old5000/new1000 및 신규batch cohort별 성능; N4와case/prompt/target identity로 pairedlost/gained/conditionaldenominator.
- mean/median/p90/p95/p99의 true/new NLL, signeddesiredmargin 및 pairedharm 분포. Ties=failure와기존RSPSNS정의 유지.
- 시간에 따른 native대비 차이가 커지거나 사라지는지 실제수치로 비교. "10batch니 차이가 커야 함"을 전제하지 않는다.
- S875/S75 vs N4; RES8 vs S75/N4; RES8 vs REFIT4; RES8 vs FULL8의 고정대조. 단순축소/추가fitting/cross-layer를 서로 섞지 않는다.
- 저장된 실제L4/L8 incremental/net update와M4/M8 norm/변화 등 관측가능항목; path길이와endpointnet을 구분.
- 동일총점도동일문항보존아님을전이로보여준다. 미관측시점의정확최초failure 추정0.
- fixedHistorical128은전old5000이 아니며 general32/128작은표본/개발재사용/knownoverlap한계를기록.
- 품질/비용 참고선 WITHIN/EXCEEDS/NOT_RECORDED; threshold AND gate/결과좋은arm만선택0.
- 모든arm negative결과/nearstall/기술오류/누락을남긴다. 원인/우월성/허용claim은GH종합소유.
- auditN1280/MMLU68/FutureN성능은신규조회·평가0; audit/policyselection/full10k새확장은미실행으로구분.

## 7. 실측 비용
각cell Slurm allocated GPU-sec와 wall, 실제2GPU동시운영/queue시간, 총campaign할당시간을분리한다.
Load/restore/firstcompute-z/keys/solve/secondfit/finalhistory/diagnostics/evaluator/CP-I-O별 계측을 있는그대로 정리.
9000z와실제loss/Adam횟수/조기종료를구분하고 로그복원인지nativecounter인지출처를표시.
Prepared/M8setup의기존재사용비용 vs 이번새실행비용, 정책online vs 평가포함total, steady-state와setup상각을구분.
초기gate소요시간 또는 과거단일batch proxy를전체비용으로사용0.
실측GPU/hostmemory/disk(18CP+raw/journal), 실패/제외attempt비용을별도; 미기록purewriter는추정이름붙이기0.
N4대비ratio는동일host/실제같은component끼리. 낮은비용/controlledspeedup확정은GH근거판단.

## 8. write scope·보고·main
허용 code: project/run_scripts/low_cost_write_donor_pilot/ 내 review/reducer/plots/tests와 해당순차분석CLI만.
실행된기존source5e96dcb의검증된원본게시가능;원native/launcher/science수정·재실행0. 다른task source무단merge0.
완료된본인source+rawfree report를최신main포함여부확인후 clean integration/nonforce mainpush한다.
원시tensor/checkpoint/prompt/fullstdout Git0. Raw broadcast=NO_BROADCAST_NOT_REQUIRED;새원격payload전송/삭제0.
PNG는직접코드작성·실행, source/input/outputhash와재현명령/가능범위byte재현기록.
Red preflight input/source/분모/state/analysis격리; postrun 독립reducer/CP·history·order/coverage/cost/rawfree 검산.
부정확한비교는해당범위HOLD, finite열세는보존. GH중복audit요청없이맡은검산완료.
한국어상세 diagnostic-report-ko.md 및 CSV/manifest/rootedreceipt, 최소 다음경로:
experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/
audits/servers/server4/2026-09-14-lowcost-seq10-completed-review/
messages/acks/server4/2026-09-14-lowcost-seq10-completed-review.md
messages/server-heads/server4/2026-09-14-lowcost-seq10-completed-review.md
tasks/status/server4/2026-09-14-lowcost-seq10-completed-review.json
runs/lowcost-seq10-review-20260914-v1/ (smallmetadata).
필수: first-final-table, batchmetrics, finalpopulationmetrics, paired/cohort/NLL분포, statehistory/CPinventory, compute-summary, coverage+limitations, evidence-reuse/source-analysis manifest.
최초 FULL_READ/M0→첫실제6arm표→상세완료report/mainHEAD/tree/SHA 보고. 단계별재승인대기0.
검산/보고/main완료뒤 TASK_COMPLETE_STOP, 후속실험자동제출0. 이번결과로 다른entry/full10k안정성을 주장하지 않는다.

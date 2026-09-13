# SH4 실행 지시 — low-cost write·donor pilot

Instruction ID: ODEEDIT-S06-LOW-COST-WRITE-DONOR-PILOT-SH4-V1
Nonce: ODEEDIT-GH-SH4-LOW-COST-WRITE-DONOR-20260913-R1
Authority: 사용자 2026-09-13 첨부 GH-LOW-COST-WRITE-DONOR-PILOT-20260913-R2와 SH4 별도 task 배정 요청.
Owner: SH4 / server4 / session 01a04939-b5c7-7a03-ba2d-ef3343d62cfd.
Expected CWD /data/janghj/ODE-edit; repository hyunjun1127/ODE-edit.
이 지시는 신규 독립 task다. SH1 E01/AOS, SH2 B/downstream 및 중지 ORBODE를 재개하지 않는다.

## 1. 정독과 source pin

다음 원문 전체를 먼저 읽고 실제 source와 참조 자산을 점검한다.
- project/proposals/2026-09-13-low-cost-write-donor-pilot-gh-instruction.md
- plans/global/2026-09-13-low-cost-write-donor-pilot-design.md
- plans/global/2026-09-13-low-cost-write-donor-pilot-cells.csv
- plans/global/2026-09-13-low-cost-write-donor-pilot-contract.json (schema v2)
- 최신 PROTOCOL 및 이 envelope.
첨부 SHA 1df8385b9d621e1829cfa98530f5e335b07587ca765ac57f73d78c392afb6162.
게시 GH 원문 SHA 46cfb78df30fabb412c2690121b7f98259065124064258fb1029eec1ac0c3fdb.
둘은 line-ending/trailing whitespace 무시 비교 일치; 게시본을 원첨부 bytes라고 부르지 않는다.
Design SHA 585dfddb90031d719a668232be1af4ac1815cc81b48f4500c59b8b34fbd016c8.
Cells SHA 3403afae3e0afba967dc30a1a79196d34a023ce2c2de125a6ed9c577759ab2b6.
Contract SHA 086590036cd4074c8b68ec6f68405f2144f9c2bff4a8a10627f05123c6057045.
게시 시작 main 153161855468c1a72ad07cc9aca6201444bade99; 실제 시작 시 remote/local SHA/tree/diff/dirty를 별도로 기록한다.

문서의 Server1 우선 제안은 최신 사용자 SH4 지정으로 override한다.
실제 Server4에서 native와 후보를 함께 비교한다. S1 native와 S4 후보를 matched effect/cost로 섞지 않는다.
원본 adapter reference b51dcf5ab825608bee81dd13549318d8d267e835; BLUE 311b076a92e4ed0f14f5c8b4909732da781bc5f7.
b51 terminal_performance는 기존 seen-full에 없는 request_order를 직접 읽으며 fullseen 크기도 6000/10000에 묶여 있다.
기존 schema repair 58f50a25809779918b22ad0aceded732c097eab4를 읽고 필요한 row-identity adapter/guard를 명시된 diff로 통합하라.
완료되지 않은 S1 runtime을 mutable import하지 않는다. source reference와 실제 execution/import SHA를 구분해 단일 frozen 실행 source를 만든다.
향후 5500 fullseen을 지원하는 evaluator 연결을 점검하되 writer 재실행으로 평가하지 않는다.

## 2. 승인 범위와 자원

전용 branch codex/server4-low-cost-write-donor-pilot-v1 및 별도 clean worktree를 만든다.
Raw root /data/janghj/ODE-edit/local/low-cost-write-donor-pilot/20260913-v1/; attempt별 create-once.
허용 source write scope:
- project/run_scripts/low_cost_write_donor_pilot/ (주 구현, tests/launcher/adapter 포함)
- project/run_scripts/baseline_mechanism_first/ (필수 adapter 수정만; 전용 worktree에서 diff 기록)
그 외 baseline/BLUE 원본 repo, SH1 실행 source, 공유 dirty/protocol/global plan은 수정하지 않는다.
서버 소유 audit/report/control 경로와 source branch push는 허용한다.
복잡한 독립 구현/수식·상태 감사에만 bounded blue/red subagent를 사용하고 단순 확인/보고는 직접 수행한다.
Red preflight: source/asset/target-mode/panel identity, history 시점/P mapping, 평가 비개입, resource collision 및 명시적 mem.
실제 잘못된 write, NaN/Inf, 비교 identity 오류는 block; finite poor performance, 참고선 초과, history setup 비용은 scientific observation/warn으로 보존한다.
Red postrun: 분모/전이/endpoint/history/cost/reconstruction 및 원문 coverage. GH가 동일 raw/GPU 감사를 중복하지 않는다.

Slurm submission ALLOWED: 우선 P0 준비와 P1 여섯 core endpoint의 봉인된 프로그램.
Job names odeedit_lowcost_prep_s4 / odeedit_lowcost_core_s4 (technical attempt suffix 허용).
Server4 프로젝트 cap2; 기본 1GPU/8CPU/60416M per process, exportNONE, 실제 node/session/owner 확인.
기존 active와 admitted pending capacity까지 재계수하고 dependency/throttle로 aggregate cap2를 보장한다.
Walltime은 실측/추정과 스케줄러 한도에 근거해 lock한다. GPU-hour cap=null; 과거48GPUh 등을 상속하거나 임의 cap을 사용자 제한으로 부르지 않는다.
초기 비용/디스크 예상과 actual ledger를 보고하되 기존 E01 모든 자료 완결/비교 byte equality를 기다리지 않는다.

## 3. Entry와 핵심 비교

Llama-3-8B-Instruct revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32, 기존 tokenizer/position/context/eager/evaluator MB16.
AlphaEdit BLUE-style singleton L2=1. blue=False five-layer L2=10 native와 치환하지 않는다.
공통 We=L4-only W50/M50, 다음 B051 fixed10k ordinal [5000,5100).
고정 데이터 /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json.
Dataset SHA 3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1;
ordered root 5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
대체 sample/인접 checkpoint/context 재생성으로 exact라고 주장하지 않는다.

Core 6개를 고정한다.
- N4: 1.0 D4, second fit 없음.
- S875: .875 D4, second fit 없음.
- S75: .75 D4, second fit 없음.
- FULL8: 1.0 D4 후 실제 state의 fresh L8 z/K/R/write.
- RES8: .75 D4 후 실제 state의 fresh L8 z/K/R/write.
- REFIT4: .75 D4 후 실제 state의 fresh L4 z/K/R/write.

정확한 same-host capsule 없으면 Server4 fresh-native N4 B100 한 번으로 기준을 만든다.
Original-target replay / same-host fresh-native / same-host target replay를 각각 표기한다.
D4는 actual stored WN4-We; alpha0/1 snapshot exact, 중간 alpha FP32 materialization/rounding 기록.
N4 실제 호출이 M을 append하면 fanout 전에 entry M으로 되돌린다.
각 branch entry W/M/RNG 복원; 두 fitting 사이 current history append0.
최종 actual endpoint에서 선택 layer마다 current keys history append 정확히1, REFIT4도1.
기존 native wrapper는 apply에서 history를 commit하므로 이를 단순 두 번 호출하지 않는다.
원본 target/solve math를 유지하며 fit와 finalization을 분리하고 source-exact 대조/counters로 증명한다.
새 z 최대25 loss evaluations/최대24 Adam updates의 원본 의미 유지, 새 target budget/clamp/optimizer 변경0.

M8는 공통 We에서 이전5000 request events/contexts를 B100 FP32 chronological 순서로 한 번 재인코딩해 만든다.
Zero M8/역사적 BLUE M8로 대체0; alpha별 재생성0; FULL8/RES8 동일 M8; suffix는 append-only.
정확히 일치하는 기존 M8가 있으면 provenance 대조 후 재사용할 수 있다. setup 정보량/시간을 공개한다.
Physical L8 -> asset P index4 -> singleton local0; L4 -> asset0 -> local0.
FULL8을 historical BLUE chain replay라고 부르지 않는다.
z/K/R donor fresh state, finite poor endpoint 유지, Official rescue/quality fallback/rollback selector0.

## 4. P0 입력과 선택적 전송

evidence-reuse-manifest에서 완료 관측, 실제 bytes, 신규 CPU, 신규 GPU, unavailable을 구분한다.
S4 checkpoint는 S2로 이관되어 원래 경로가 삭제되었음에 유의한다.
정식 mapping: transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv.
S2 retained roots:
 /mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/
 /mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/
필요한 L4 W50/M50와 companion만 확보한다. 전체183CP/모델/P/stats 복사 금지.
S1 완료 E01/ABC/capsule/source/log evidence도 정확한 subset만 read-only 협의한다.
라이브 E0146439/46440 scientific output/진행률 조회0, 해당 task 재개0.
별도 transfers/approvals/2026-09-13-low-cost-write-donor-sh4-selective-inputs.md 범위로 input transfer를 진행한다.
SH4가 수신 root의 sole writer, source hosts는 보존. SHA/bytes/closure/space 확인 후 create-once partial→verified seal.
전송 인증 설정 변경/--delete/기존 파일 overwrite0. 필요한 입력 문제가 생기면 영향 fixture만 명시하고 독립 CPU 구현은 진행한다.

## 5. 패널·후속 단계·판정

모든 core endpoint Current R100/P200/N1000, 동일 outcome-independent Historical128 R/P/N, Wiki128, MMLU development32를 평가한다.
Historical active/superseded, signed desired margins, true/new NLL, TF strict, paired lost/gained/ties=failure 유지.
MMLU는 기존 fixed100에서 outcome-independent hash32/remaining68를 사전 seal, alternative scorer integer correct counts 사용.
Audit128requests/N1280는 사전 seal: 개발 case와 알려진 subject/relation overlap 제외, 알려지지 않은 overlap 한계 기록.
Audit/General/Official P-N/Future N을 optimizer/online alpha/donor/fallback에 넣지 않는다.
b51의 PAIR2 관측 경로를 canonical MB16로 대신 부르지 않는다. 실제 endpoint W/M hash 확인 후 평가, entry 복원.

개발 core factual table을 GH에 먼저 전달한다. GH가 최대1후보의 policy와 claim을 lock한 뒤 audit 단계로 진행한다.
Audit N4+선택후보, donor 선택이면 FULL8 포함. 감사 후 GH가 허용한 고정 정책만 B51–B55 native와 각각 반복한다.
총10 batch executions, 첫 static2의 정확 재사용 시 신규8 write. 자기 trajectory와 fresh z를 사용한다.
suffix terminal actual W55 fullseen5500, old5000/new500/overwrite 분해 및 재구성 가능한 W/M/RNG 보존.
이는 원문 범위의 조건부 후속이며 성능 수치 AND gate나 사용자 재승인 반복이 아니다. 지금 후보를 미리 정하지 않는다.
L5, alpha.5/.875 donor, 60batch 확대/full10k/광범위 sweep/ODE/PCG는 자동 제출하지 않는다.

모든 수치 참고선은 WITHIN/EXCEEDS/NOT_RECORDED로 보고하며 자동 탈락0.
GH는 ALLOW / ALLOW_WITH_LIMITED_CLAIM / NEEDS_TARGETED_CHECK / NOT_SUPPORTED / INVALID_COMPARISON을 판단한다.
소수 품질 손실/CI0 포함/cost>2x도 그대로 공개한다. 반대로 핵심효과 없는 결과를 양의 claim으로 바꾸지 않는다.
SH는 원수치/분모/paired contrasts/반대 근거/누락만 보고하고 허용문장·최종선택은 GH global report 소유다.

## 6. 초기 gate와 중지 정책

기존 사용자 INITIAL_GATE_ONLY 지시 유지.
필요한 CPU/source/history/materialization checks와 실제 native/첫 유효 partial-state second fit 적용·복원 확인까지만 초기 모니터링한다.
공통 preparation/model load만으로 core INITIAL_VALID를 선언하지 않는다.
필요 범위의 검증을 같은 봉인 core 프로그램 초기에 배치해 나머지 여섯 endpoint 실행을 끊지 않는다.
초기 actual-valid 뒤 MONITORING_PAUSED_AWAITING_USER: 이미 제출된 프로그램은 그대로 계속.
그 뒤 agent polling/heartbeat/terminal wait/follow-up submit/자동 analysis/report/main integration0.
사용자 명시 recall을 GH가 전달하면 완료 확인·사실 분석을 재개한다. audit/후보 suffix에는 위 GH policy lock도 필요하다.
기술 오류는 typed failure와 immutable attempt/cost 보존; scientific threshold를 바꿔 gate를 통과시키지 않는다.
전체 pilot 완료/성능 PASS를 initial gate로 주장하지 않는다.

## 7. 산출물과 종료

messages/acks/server4/2026-09-13-low-cost-write-donor-pilot.md
messages/server-heads/server4/2026-09-13-low-cost-write-donor-pilot.md
tasks/status/server4/2026-09-13-low-cost-write-donor-pilot.json
runs/low-cost-write-donor-pilot-s4-20260913-v1/ (small metadata)
audits/servers/server4/2026-09-13-low-cost-write-donor-pilot/
experiment-reports/servers/server4/low-cost-write-donor-pilot-2026-09-13-v1/
GH synthesis: experiment-reports/global/low-cost-write-donor-pilot-2026-09-13-v1/ (SH write 금지).

최소 evidence-reuse-manifest/comparison-capsule/endpoint-metrics/paired-transitions/quality-frontier/history-provenance/compute-ledger.
선택 뒤 policy-lock/suffix-summary; GH claim-decision은 별도 global 산출물.
준비/online(z,keys,solve,history)/진단/평가/I-O/allocated total/technical cost를 분리한다.
공유 D4/z/M8 study cost 중복0, 각 policy 필요한 비용 누락0. 같은 host 실측 전 저비용 claim0.
raw tensors/checkpoints/prompts/cache/full logs local-only, PNG는 코드 실행 생성 및 provenance.
원문 계획과 변경/누락/한계/비용을 한국어 factual report에 기록한다.
완료·recall 이후 검증된 본인 source+raw-free report만 기존 권한에 따라 clean integration/nonforce main push하고 HEAD/tree/report SHA 보고.
현재 initial pause 뒤 자동 push 금지; 미완료 범위를 완료로 승격하지 않는다.
Raw broadcast는 이번 GH 위임의 NO_BROADCAST_NOT_REQUIRED: S4 보존+Git compact provenance, 명시 input/후속 필요한 공유만 별도 allowlist.
최초 ACK는 FULL_READ, 실제 host/source, available/missing inputs, core 계획/비용 estimate와 첫 factual table 예상(실측 아님)을 담는다.
GH 추가 중복 감사 대기0; 이번 승인 범위의 준비·구현·submit·initial gate를 맡아 진행하라.

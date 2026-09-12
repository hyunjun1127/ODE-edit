# GH → SH2: 기존 downstream 평가 보고서 확인·미작성 시 상세 보고

instruction_id: ODEEDIT-S06-SH2-DOWNSTREAM-COMPLETED-REPORT-20260912-V1
nonce: ODEEDIT-GH-SH2-DOWNSTREAM-REPORT-RECALL-20260912-R1
from: GH / 01a04939-8873-7673-8dca-4c7fc5e31af0
to: SH2 / 01a0493a-074c-7f91-9a13-769116326fef
server/CWD/repository: server2 / /mnt/raid5/janghj/ODE-edit / hyunjun1127/ODE-edit
사용자 원문: “이전에 server2에서 downstream task 측정시킨거도 report 만들어진게 잇는지 확인해보고, 없으면 만들라고 지시해”
유형: 기존 완료 평가에 한정한 명시적 사용자 recall; CPU report-only. GPU/model/evaluator/Slurm submission 금지.

## 1. 보고서 존재부터 확인

GH가 origin/main 8a1ce01cedcaece5f765e61b02389b8ca28efb4d의 experiment-reports, server2 messages/audits 및 remote downstream-named branches에서 최종 보고서를 찾지 못했다. 이는 SH2 local branch/보고서까지 확인한 결론이 아니다. SH2가 자기 전용 branch/worktree/output 및 기존 index를 확인한다.
- 실제 완료 보고서가 있으면 source/job/범위/manifest를 확인해 경로·SHA·main 포함 여부를 반환한다. 같은 보고서를 불필요하게 새로 쓰거나 원본을 덮어쓰지 않는다.
- 없으면 아래 기존 결과에서 CPU 검산·분석·한국어 보고서를 작성한다.
- partial report만 있으면 기존 bytes를 보존하고 새 final/partial version으로 보완한다. 결과가 incomplete/failed면 그대로 명시하며 새 평가나 rerun으로 몰래 채우지 않는다.

## 2. 정확한 대상과 현재 확인 범위

원 instruction ODEEDIT-S06-BLUE-CHECKPOINT-DOWNSTREAM-S2-S4-V1.
대상 job42706의 최초 전달:
- execution source257fe5483dc3407690fec9c9bb45cc7dcda91215/tree5da774085483d8f88de2b146417efa5c11cd1127
- worktree /mnt/raid5/janghj/.codex/worktrees/odeeditsh2-blue-checkpoint-downstream-v1
- root /mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/attempt-v1/
- execution.lock SHA cf9b6ad68d7c68683799bbe1acb0598f0c2fd37ba54c0c144545f3120ba15abc
- 기존 pause receipt monitoring-pause.json SHA922982a943c778a5ee7f3a9015b8b363968ab88427a95e74b2a6bd17291fdae9
- initial-valid SHA4f557b49906205e8ed3d2c4fbcd48a9db2223b53a374f885bd6bd2c00e705dc6
이는 최초 W0+첫 checkpoint gate이며 전체 완료 증거가 아니다. 대상 exact job/children의 단발 bounded scheduler 확인과 해당 root terminal/run registry만 조사한다. RUNNING이면 설정을 건드리지 않고 현재 상태·기존 partial 여부만 보고 후 pause. 반복 polling/terminal 대기0.

원 계획은 W0 한 번 + 완료6 chains의 각12 checkpoint=72 checkpoint, 총73 model states의 6 tasks다. 각 task 고정100 examples였으므로 모두 완료됐으면438 state-task cells/43800 task-example observations이지만, 이 숫자를 증거 없이 actual count로 쓰지 않는다.
대상 arms:
- MEMIT_BLUE (L4+L8), MEMIT_BLUE_L4_ONLY, MEMIT_BLUE_L8_ONLY
- AlphaEdit_BLUE (L4+L8), AlphaEdit_BLUE_L4_ONLY, AlphaEdit_BLUE_L8_ONLY
Checkpoints edits100/500/1000/2000/.../10000 (실제 manifest의12개 지점 확인).

L5/L6/L7/native baseline의 추가 checkpoint, GPT stats, PRE_EDIT42673 재평가, 현재 A/B44970·45029·45631/45633과 다른 task는 이번 범위가 아니다. 특히 B-BF4 MONITORING_PAUSED_AWAITING_USER를 재개하지 않는다.

기존 S2 checkpoint imports:
 /mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/
checkpoint-manifest SHA e4625ab025e6bf57c30a5c3a1e6eece01557cd2d204c3266a37777368368884a, 72CP/62011141768B.
이 경로는 server4→server2 보존 이관 후에도 그대로 보존됐다. Server4 원 checkpoint는 삭제됐으므로 원격 source path를 읽거나 복원·재전송 요청하지 않는다. Existing exact CP/source/schema/restore 검증을 SHA 결속하여 재사용하고 보고서 작성을 모든168CP 전수 GPU검증과 같은 새 gate로 확대하지 않는다.

## 3. source/data/evaluator 계약

이미 승인·실행한 고정 데이터 파일과 evaluator를 사용한다. 원 data 출처는 /mnt/raid5/janghj/00.KE/02.LTE/AlphaEdit/glue_eval/dataset, EasyEdit에 복제한 파일 및 raw-free data manifest다. 실제 실행 lock의 경로/10member SHA/100-row indices/labels를 우선한다. 없는 파일을 존재한다고 하지 않는다.
- SST/SST2, MRPC, CoLA, RTE, MMLU, NLI의 실제 dataset/config/split/100rows와 indices/fewshot/generation length를 source/lock에서 밝혀라.
- 과거 전달은 tests[10:110], fewshot0, generation length5였다. 실제 bytes/config와 대조하여 적용값을 기록한다.
- F1 averaging/ACC/MCC 및 generation/alternative branch를 명확히 분리한다. Figure6 축을 그대로 보고 metric이 모두 같은 F1이라고 추정하지 않는다.
- RTE_LABEL_MAPPING_CORRECTED_V1: scoring-only inverse label mapping, 두 branch 및 invalid 보존. 실제 applied adapter와 SHA를 확인하고 원본 raw labels/의미를 적는다.
- MMLU source-exact parser의 A vs A-newline 등 invalid 특성은 보존·count 공개. 결과를 좋아 보이게 새 parser/label/추출법으로 조용히 재평가하지 않는다.
- 모델/FP32/backend/TF32/tokenizer/padding/생성 deterministic 여부, checkpoint restore 및 nonselected W0/eval nonmutation evidence의 실제 범위를 기록한다.
- Source hash와 dataset hash/재사용 검증은 실제 endpoint 결과나 CPU/GPU parity의 대체 증거가 아니다.

## 4. CPU 검산과 상세 보고

완료·봉인 output의 상태/누락/중복/count/분모/state identity를 점검하고 raw stored prediction/label/score에서 metric을 독립 재계산한다. 새 model forward/GPU evaluator 호출 없이 가능한 분석만 한다. 누락은 NOT_MEASURED/FAILED/INCOMPLETE로 남기며 보간하지 않는다.

보고서에는:
1. 실제 상태수·task수·examples·완료/실패/누락/초기gate와terminal 구분
2. family별 MEMIT/AlphaEdit 별도 표, W0 및 BLUE/L4/L8의 checkpoint별6task 결과; BLUE를 original로만 표기하지 않음
3. final10k checkpoint 표와 전체12점 경로, W0 대비 Δ, 같은 task/item 기준 paired lost/gained; raw prediction이 없으면 해당분해 NOT_RECORDED
4. F1·ACC 등 실제 metric 정의/평균방식/invalid count, MMLU 두branch 별도, RTE correction 경계
5. 같은100 examples 반복관측을 독립43800 samples로 오해하지 않음. Full GLUE/MMLU/NLI benchmark나 전체capability 평가로 확대하지 않음
6. Current editing RS/PS/NS와 downstream task score를 별도지표로 구분. 다른14chain lifelong report와 연결 가능하나 측정하지 않은 arms의 downstream 점수 추정0
7. 지식손실·회복·layer/계열 차이는 관측과 가능한 설명을 나누고 인과·보편 우월성 주장0
8. actual wall/GPU seconds, state/eval/restore/load cost와 시간누락·하드웨어 차이; 최초512.91s/remaining10.12h는 추정으로만 구분
9. 실행 source/runtime/data/CP identities, 분석source, 실패/미측정 한계, 재현 명령·raw 경로·manifest/receipt
10. 동일 raw에서 생성한 family/task별 code-generated PNG(예:6panel curves)와 CSV; imagegen/외부시각화 도구0. 보고서가 이미 있으면 불필요한 재생성0

사용자가 상세 보고를 요청했으므로 이번 instruction 한정으로 SH factual-only 기본 규칙에 대한 분석 예외를 허용한다. 관측사실·가능한 설명·미분리 한계를 구분, scientific_promotion=false. Model correctness 결함을 발견하면 근거/영향만 보고하고 새 실행/수정은 자동 진행하지 않는다.

## 5. write/Git/전달

새 dedicated branch codex/server2-blue-downstream-report-review-20260912-v1 및 고유 clean worktree 권장. shared dirty/root, 기존 source/산출물을 덮어쓰기·삭제·되돌리기0.
새 write scope:
- project/run_scripts/blue_checkpoint_downstream_review/ : CPU reducer/tests/plot/report source만
- local/blue-checkpoint-downstream/20260912-report-v1/ : CPU scratch/비공개 per-item 분석·verification
- experiment-reports/servers/server2/blue-checkpoint-downstream-review-2026-09-12-v1/
- audits/servers/server2/2026-09-12-blue-checkpoint-downstream-report/
- messages/server-heads/server2/2026-09-12-blue-checkpoint-downstream-report.md
- tasks/status/blue-downstream-report-review-sh2-20260912-v1/server2.json
- runs/blue-downstream-report-review-sh2-20260912-v1/
- plans/updates/server2/2026-09-12-blue-checkpoint-downstream-report.md

기존 완료 report가 있으면 그 canonical path를 우선 안내하고 불필요한 새 디렉터리를 만들지 않는다.
最종 Korean diagnostic-report-ko.md와 raw-free table/code/manifest/checksum/test 결과를 own-scope non-force branch/main push한다. 이전 사용자 완료 source/report main 정책에 따른 승인이다. 기존 unpushed execution source는 exact run provenance와 본인 소유를 확인한 완료분만 포함 가능; 현재multilayer sharedkernel/B 미완성 변경을 함께 merge하지 않는다.
Raw weights/prompt/prediction/log payload는 Git0. 기존 local 보존으로 NO_BROADCAST_NOT_REQUIRED; 새 대형 transfer/SH4나SH1작업재개0. Git conflict는 사용자 변경을 덮지 말고 보고.
완료시 report path/SHA/실제평가범위/핵심수치·한계/mainHEAD를 간결히 전달하고 STOP. Existing report가 있다면 그 path/identity/main상태만 먼저 알려도 된다. GH의 중복 raw감사·재승인을 기다리지 않는다.

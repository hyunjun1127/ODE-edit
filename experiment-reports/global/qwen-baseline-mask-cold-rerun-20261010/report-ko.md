# Qwen baseline context-mask 수정 및 cold rerun

nonce USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1.
공통 source ac19db2f, 공식 EasyEdit cached context generator의 mask만 fullprefix로 변경.
샘플링/길이/seed/hparams/수식 변경 없음. 원 upstream SHA는 SOURCES.json에 보존.
Qwen/GPTJ/Llama 작은 CPU FP32/eager 모델로 실제 generator·서로 다른 길이 rightpadding,
무편집 weights 불변, 잘못된 Qwen mask negative control, 네 native import/FEhistory 경로 검사3개 PASS.
source166 검증 PASS. pretrained/GPU 성공 및 실험 완료를 의미하지 않는다.

11개 cold 편집: SH2 본표8, SH1 FE_HISTORY1, SH4 heldout500 두개.
SH2의 FT61900은 편집 오류 대상이 아니며 최신공식 evaluator 최종CP 평가-only를 별도 등록/중복 확인.
OURS/PRICE/tuning, BLUE/FT 편집, 기존 Llama/GPTJ 실행 보존. 무관자원 cap 증액0.
각SH는 exactowner/현재상태 확인후 scopedcancel/의존성재연결/새freeze/held검사/release를 수행한다.
CF FLUCON은 최신 사용자 명령으로 DEFERRED, 최종2K(heldout500)CP 보존.

원 수치/원source/raw/CP는 유지하지만 영향받은 기존 점수는 수정된 본표 성능에서 제외했다.
새 ID 도착 전 README는 RERUN_REQUIRED로 표시하며 Slurm PENDING을 가정하지 않는다.
현재 단계: 공통 source 게시 및 SH1/SH2/SH4 직접 수락 완료.
SH1은 old61975를 보존하고 새 62061(official-s1-cf-qwen25-memit-fe-history)을
held 검사 후 release했다. 최초 관측 PENDING, dependency 없음, 실행 source
8da621a9e12fd341a4a70fba95531af6fe4fdb80. W20 및 온라인 기록은 아직 미관측이다.
새 context SHA는 실제 B1에서 기록되며 지금 생성 완료로 주장하지 않는다.
SH4도 old61776/61777을 보존하고 새 MEMIT62063/AlphaEdit62064를 held 검사 후
release했다. source a28515284cc8152a9bf3d6c2a75f0f7c0b4cc387,
afterany:62037 → 62063 → 62064의 단일 lane이며 최초 PENDING이다.
이 두 run은 CF heldout500/B5이며 본표 2K에 포함하지 않는다.
보호 OURS62038은 HELD 그대로다. context SHA/온라인/B5 결과는 아직 미관측이다.
SH2 8개 cold 및 FT eval-only도 전량 held 검사 후 release했다.
실행 source 7b5097aa447946e35de42229c22b0c0feabd11ae,
2026-10-09T15:46:20Z 단발 snapshot은 모두 dependency PENDING이다.
SH1 근거: audits/servers/server1/qwen-baseline-mask-cold-rerun-20261010/submission.json.
SH4 근거: audits/servers/server4/qwen-baseline-mask-cold-rerun-20261010/submission.json.
직접 수락과 실제 제출/결과 완료는 별개다. 장기 GPU 완료 대기/새 recurring monitor 없음.

[정본 실행 envelope](../../../messages/head/2026-10-10-qwen-baseline-mask-cold-rerun.md) ·
[공통 검산·원 조사 SHA](../../../audits/global/qwen-baseline-mask-cold-rerun-20261010/common-repair.json).

## 실제 등록 통합

| Server | Dataset / method | old → new | 실제 dependency |
| --- | --- | --- | --- |
| 1 | CF2K MEMIT_FE_HISTORY | 61975 → 62061 | 없음 |
| 2 | CF2K MEMIT | 61954 → 62073 | afterany:61946 |
| 2 | CF2K AlphaEdit | 61958 → 62075 | afterany:61947 |
| 2 | CF2K MEMIT-FE | 61966 → 62077 | afterany:61944 |
| 2 | CF2K SPHERE | 61970 → 62079 | afterany:62072 |
| 2 | zsRE2K MEMIT | 61956 → 62081 | afterany:62073 |
| 2 | zsRE2K AlphaEdit | 61960 → 62083 | afterany:62075 |
| 2 | zsRE2K MEMIT-FE | 61968 → 62085 | afterany:62077 |
| 2 | zsRE2K SPHERE | 61972 → 62087 | afterany:62079 |
| 2 | zsRE2K FT 평가-only | 61900 W20 → 62072 | afterany:61945 |
| 4 | CF heldout500 MEMIT | 61776 → 62063 | afterany:62037 |
| 4 | CF heldout500 AlphaEdit | 61777 → 62064 | afterany:62063 |

11개 cold 편집 + 1개 저장 W20 평가-only이며 모두 actual 등록/release를 확인했다.
SH2 GPU0 archive는 62074/76/78/80/82/84/86/88, collector62089다.
SH2의 영향받은 활성15개는 정확 대상 downstream-first 취소했고 원 완료61956/61960은 보존했다.
FT/BLUE 및 GPTJ 평가 source/ID는 보존; GPTJ pending head의 자원 edge만 안전하게 재연결했다.
새 graph에 취소된15개 old ID가 dependency로 남지 않음을 GH가 compact receipt로 재검산했다.
cap은 각 owner의 현재 정책 server1=4/server2=4/server4=2이며 이번 작업으로 증액하지 않았다.
checkpoint/context/config/source 경로 및 SHA는
[통합 영수증](../../../audits/global/qwen-baseline-mask-cold-rerun-20261010/submission-integration.json)에 기록했다.
SH2 실제 원 receipt는 server2 local/qwen-baseline-mask-cold-rerun-20261010/registration-r1/released.json이다.
원 checkpoint/raw/비용/실패·취소 이력은 보존했다. 새 context SHA는 실제 생성 전 미관측으로 남긴다.
GPU 결과·W20·W&B online readback 완료를 주장하지 않는다. 미제출 항목은 없다.

후속 별도 사용자 context 생성 순서 지시도 SH4에 직접 전달했다:
Qwen → GPT-J → Llama, nonce USER-GH-SH4-NATIVE-CONTEXT-ORDER-20261010-R1,
owner accepted turn 01a12156-29cb-7bc3-9bc1-75e0d132de43.
이는 새 baseline 편집 arm이 아니며 기존 frozen context와 62038/62063/62064를 변경하지 않는다.

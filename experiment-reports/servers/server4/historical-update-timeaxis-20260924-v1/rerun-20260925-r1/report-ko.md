# Historical timeaxis rerun recall: 수치계약 차단 인계

Instruction: `ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RERUN-20260925-R1`.

**판정: BLOCKED_NUMERICAL_CONTRACT. 신규 제출0, job 변경0, GPU0.** 재제출 자체는 승인됐으나 이번 정본은 수치기준 변경과 반복-to-PASS를 금지한다. 최초 실패가 고정 MB16↔MB1 수치계약 위반이며, 정의를 유지하는 구현 수리 근거는 아직 확인되지 않았다. 단순 같은 실행 재등록이나 임의 threshold 완화로 우회하지 않았다. 이는 계획만 남긴 것이 아니라 exact accounting·실패 source·보존 raw를 대조한 차단 판정이다.

## 정확 job 및 최초 원인

| job / 역할 | scheduler | 실제 scientific 경계 | 부모 GPU초 |
|---|---|---|---:|
| 53182 / BASE_ALPHAEDIT | FAILED, 1:0, 1463초 | 자체 T1 PASS 후 MEMIT 실패를 받은 join에서 종료 | 1463 |
| 53183 / BASE_MEMIT | FAILED, 1:0, 106초 | W100 MB16↔MB1 NLL 비교에서 고정 허용치 초과 | 106 |
| 53184 / CPU collector | COMPLETED, 0:0, 0초 | TECHNICAL_BLOCKED, science_gate_bypassed=false | 0 |

종료한 job은 scontrol 메모리 기록에서 빠져 Invalid job id가 반환됐다. 이를 NOT_FOUND/미실행으로 해석하지 않고 원 immutable submission의 owner/name/command/fullargv/node/dependency와 sacct 부모 accounting을 결속했다. 현재 RUNNING/PENDING인 해당 실행은 없다. 이전 SH1 이관은 철회 그대로다. 다른 job의 상태나 과학자료는 조회하지 않았다.

최초 예외는 MEMIT frozen `worker.py:84`의 `check_raw(a,single)` → `compare:14`다.

- 기록된 최대 답변 평균 NLL 절대차: **0.0005242824554443359 nats/target token**.
- 고정 상한: **0.00025**, 초과량 **0.00027428245544433593**, 상한의 **2.0971298218배**.
- MEMIT W100이라는 위치는 실행 순서(0/1/20/50/100), 앞선4개 저장 receipt 및 stdout으로 결속했다.
- 실패한 W100의 MB16/MB1 개별 raw는 저장 전 assert로 소실됐다. 문제 category/case/토큰과 당시 margin 차이는 **NOT_RECORDED**다. aggregate 오류값을 per-case raw 독립검산 값처럼 쓰지 않는다.
- AlphaEdit의 PEER_TECHNICAL_FAILURE는 후속 전파이며 최초 원인이 아니다. CPU collector exit0도 T4 과학 성공이 아니다.

OOM/TIMEOUT/ENOSPC/API/serialization을 최초 원인으로 나타내는 증거는 없다. Batch shape·padding·floating-point 경로 차이는 검토 가능한 후보일 뿐 구체적 low-level 원인으로 확정하지 않는다. 동일 frozen evaluator의 수동 left padding/label alignment/FP32 full-vocab log-softmax/MB16/MB1 경로를 확인했다. 현재 증거로 코드를 바꾸면 수치경로/계약이 그대로라는 주장을 할 수 없어 production runtime은 변경하지 않았다.

## 독립 CPU 검산과 재사용 가능 범위

원 source member SHA, lock, token manifest, 원 design29와 main 게시29의 exact SHA를 재검산했다. CP24/model4는 기존 fullSHA receipt+현재 unchanged stat로 결속했으며 큰 payload 재해시/전송/모델 load0이다. 실행source는 `730a4a9768e5650e01fd9afdc4e0f7895c86ea92`, tree `48b61f229bedb75ec025a8371c160b6a1aad3dab`, lock `54abf9a7572eceda008db9c001ed8d1fc9de8f5a7397990a4bed26f313bc1699`다. 앞선6ef71ed2→730a4a97의 force_removal/collector 수리 이력은 유지했다.

| 보존 근거 | CPU 대조 | 재사용 한계 |
|---|---|---|
| Alpha endpoint W0/W1/W20/W50/W100 5개 | 원 raw/repeat/MB1/restored의 identity·finite·토큰정확도·NLL·margin·flip 재계산 | 같은 source/input/state/layout 검증 evidence; 전체 science 완료가 아님 |
| MEMIT W0/W1/W20/W50 4개 | 같은 독립 비교 통과 | W100 실패를 대체하지 못함 |
| Alpha T1 PASS와12개 diagonal | receipt identity·5weight SHA·direct/arithmetic·restore 및 저장 오차 대조 | 대각선 raw NLL이 없어 해당 숫자는 receipt 기반, GPU 재실행 검증 아님 |
| 공통 T0/model/24CP/data/token | 기존 fullSHA와 현재 작은 파일SHA/큰 파일stat 결속 | 향후 다른 numerical contract에는 무조건 PASS 이식 불가 |
| T2P/T2F/T3A/T3B/T4 science | 저장 score task0 | 재사용 가능한 과학 row0, 미실행은 음성 결과가 아님 |

9 endpoint×3비교=27개 비교, 답변행 **2592**개를 독립 재집계했다. 16 case×rewrite/두paraphrase×new/true=96답변행/endpoint/comparison이다. 성공집합·분모·TF strict는 저장 토큰 label과 prediction으로 대조했고, 각 variant의 요청 순서·prompt/target pair identity·target IDs를 봉인 token manifest와 대조했다. 이 검산은 실패한 W100 row에는 적용되지 않는다. 상세 [compact 검산표](../../../../../audits/servers/server4/historical-update-timeaxis-20260924-v1/rerun-20260925-r1/saved-fidelity-independent.csv)와 [진단 receipt](../../../../../audits/servers/server4/historical-update-timeaxis-20260924-v1/rerun-20260925-r1/diagnosis.json)를 보존했다. 신규 분석기의 합성 CPU4 tests PASS는 GPU 증거가 아니다. 실제 입력 weight를 재구성하거나 GPU continuation을 수행하지 않았다. noCP이므로 실행중 RAM 상태의 exact crash-resume은 제공되지 않는다.

## 비용·보존

부모 allocation 합 **1569 GPU초 = 0.4358333 GPU시간**. batch/extern을 중복 합산하지 않았다. CPU collector는 GPU0, scheduler elapsed0이다. allocation은 utilization이 아니다.

Alpha program wall1458.4697초, T1 PASS시 program347.3663초로 이후 join 대기 약1111.1034초가 포함됐다. MEMIT program103.4179초. 각각 forward37.9100/26.2318초, host peak33833.0234/33835.4688MiB, GPU allocated peak33098108416B로 기록됐다. load/materialization/restore/forward timer는 nested이므로 합쳐 전체 wall로 쓰지 않는다. 이전 T0/원BASE 학습비용은 이번 allocation에 다시 더하지 않는다.

원 source/archive/실패/로그/partial receipt/입력24CP를 모두 그대로 보존한다. 새 edited checkpoint나 native fit/history append/새평가0, source numeric/runtime 변경0이다.

## 이번 실행 경계와 다음 판단

현재 승인 안에서는 원인을 기록한 **typed blockage**가 종료 조건이다. 허용치를 올리거나, MB1 비교를 생략하거나, dtype/kernel/panel을 바꾸거나, 같은 실행을 통과할 때까지 반복하지 않는다. 원인의 추가 분리 진단 또는 수치계약 처리 방향이 정해져야 후속 rerun을 판단할 수 있다. 원 source에 PASS 파일을 복사해 join을 우회하지 않는다.

공유 root의 ignored session 설정은 과거 session을 가리켜 helper BLOCK이었다. 실제 CODEX_THREAD_ID와 registry는 현재 SH4와 일치하므로 승인된 별도 child worktree에만 실제 session/CWD를 결속해 helper PASS를 확인했다. 공유 설정은 수정하지 않았다.

새 branch `codex/server4-historical-update-timeaxis-rerun-20260925-v1`; 새 분석 모듈/테스트·소형 보고만 추가했다. Owner 직접검산, 별도 독립 red agent 미사용. Markdown table renderer와 raw-free·hash·분모를 검사하고 ownscope main 게시로 인계한다. NO_BROADCAST_NOT_REQUIRED. `new_job_ids=[]`, `monitoring_active=false`, `automatic_resume=false`. 이후 scheduler/log/result 반복조회나 자동수리 없이 STOP한다.

재현(CPU, 원본 read-only): `python3 -m unittest project.run_scripts.historical_update_timeaxis.test_diagnose_rerun -v`. 기존 audit 파일을 덮지 않는 재검증 명령은 `python3 -m project.run_scripts.historical_update_timeaxis.diagnose_rerun --verify-only`다. 최초 출력 생성은 create-once다.

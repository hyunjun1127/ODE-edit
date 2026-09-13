# GH 전달 지시문 — low-cost write·donor pilot

Instruction ID: GH-LOW-COST-WRITE-DONOR-PILOT-20260913-R2  
수신: Global Head  
상태: 사용자 전달용 지시문 초안. 이 파일 작성 자체로 GH 전송·SH 배정·GPU job 제출을 수행하지 않았다.

GH는 아래 세 문서를 기준으로 **기존 근거 재사용 → Middle 여섯 endpoint 비교 → 후보 하나의 audit 및 반복 5-batch suffix → claim과 후속 방향 판정**을 진행하라. 연구 질문은 native L4 write를 약하게 줄이거나 두 번째 fitting을 추가했을 때, 편집 품질 대비 보존 성능을 개선할 수 있는지다. 대안 writer의 유망성을 확인하기 위한 최소 구현·실험을 진행하고, 모든 선행 분석의 완결을 기다리지 말라.

- [설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-13-low-cost-write-donor-pilot-design.md)
- [실험 cell 목록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-13-low-cost-write-donor-pilot-cells.csv)
- [판정 계약 — schema v2](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-13-low-cost-write-donor-pilot-contract.json)

**1. 가장 우선하는 판정 원칙**

사용자 지시는 **“판정 기준 자체는 너무 strict하게 잡지 말고, ours claim이 적절히 드러난다면 허용으로 판정하되, report에는 냉정하게 적는다”**이다. 이 원칙을 성능·비용 판정에 적용하라. 이전 문서나 구현에 수치 gate가 남아 있으면 현재 계약에 맞춰 바꾸고, 사소한 기준 초과마다 사용자 재승인을 요청하지 말라.

Current R 새 실패 0개, P 손실 2/200 이내, Historical NS +1%p, Current N 비악화, NLL/Wiki 참고선, 신뢰구간 하한 >0, online 비용 1.5배/2배를 모두 만족해야 하는 AND gate를 만들지 말라. 수치는 유지해 초과 여부를 보고하되 **자동 탈락 조건으로 쓰지 말라.** 소수 R/P 손실, 일부 Current N 악화, 작은 양의 효과, 0을 포함하는 신뢰구간, 비용 참고선 초과가 있어도 claim을 지지하는 해석 가능한 이득이 남으면 허용할 수 있다.

허용 판단에는 다음을 적어라: 어떤 대조가 핵심 효과를 보여 주는지, 손실의 크기와 위치는 무엇인지, 그 손실·비용을 감수하고도 어떤 주장을 할 수 있는지. 좁은 claim으로도 연구 가치가 드러나면 그 범위에서 허용하라. 모든 지표에서 이길 필요는 없다. 다만 대조군으로 설명되는 효과를 새 mechanism의 고유 기여로 쓰거나, 핵심 효과가 없는 결과를 양의 결과로 바꾸어 쓰지 말라.

판정은 다음 다섯 가지로 남겨라.

| 판정 | 적용 |
|---|---|
| 허용 — ALLOW | 유효한 비교에서 의도한 범위의 ours claim이 드러나며, 손실·비용을 공개한 채 후속 실험을 진행할 근거가 있음 |
| 범위를 한정해 허용 — ALLOW_WITH_LIMITED_CLAIM | 일부 패널·품질·비용·지속성 한계가 있으나 더 좁은 claim은 지지됨. 허용 문장을 구체적으로 적음 |
| 추가 확인 필요 — NEEDS_TARGETED_CHECK | 비교는 유효하지만 핵심 효과의 해석이 갈림. 이를 가를 최소 비교 하나와 필요한 비용을 적음 |
| 현재 근거로 비지지 — NOT_SUPPORTED | 해당 claim의 핵심 이득이 관측되지 않거나 대조 결과가 이를 반박함 |
| 비교 무효 — INVALID_COMPARISON | 입력·target mode·history·평가 identity 등 비교 성립 조건이 어긋남. 영향받는 비교를 특정하고 수정함 |

허용은 **관측 범위 안의 claim 채택과 정해진 후속 실험 진행**이다. 통계적 유의성, 무손실, 낮은 비용, 전 구간 우월성, full10k 안정성은 각각 근거가 있을 때만 주장하라. 비교 유효성·평가 분리·기록의 정확성은 유지한다. 한 부분의 복원 문제가 독립적으로 가능한 유효 비교까지 계속 막게 하지 말라.

**2. 첫 실행은 Middle의 여섯 endpoint로 고정하라**

주 entry는 L4-only W50/M50, 다음 B051의 ordinal [5000, 5100)이다. Llama-3-8B-Instruct revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32, 기존 AlphaEdit BLUE-style L2=1과 봉인 evaluator를 사용하라. 재사용 자산이 있는 Server1 환경을 우선하되 실제 host를 확인하고, host가 달라지면 해당 host에서 native와 후보를 함께 비교하라. Server4 native·BLUE·L4–L8 결과는 역사적 기준으로 연결하라.

| ID | 첫 L4 write | 두 번째 fitting | 확인할 질문 |
|---|---:|---|---|
| N4 | 1.0 × D4 | 없음 | 같은 환경의 native 기준 |
| S875 | 0.875 × D4 | 없음 | 약한 축소의 편집 품질·보존 trade-off |
| S75 | 0.75 × D4 | 없음 | 기존 품질 경계 신호의 대조 |
| FULL8 | 1.0 × D4 | L8 fresh z/write | full-L4→L8 기준 |
| RES8 | 0.75 × D4 | L8 fresh z/write | 축소 후 품질 회복과 보존 이득의 공존 |
| REFIT4 | 0.75 × D4 | 같은 L4 fresh z/write | 다른 layer 효과와 추가 fitting 효과의 구분 |

N4가 정확한 comparison capsule과 일치하면 재사용하라. 불일치하면 같은 환경의 native B100 한 번으로 비교 기준을 만들어라. 첫 D4/z를 공유하고 FULL8·RES8·REFIT4의 두 번째 z/K/R은 각 실제 부분 적용 상태에서 새로 계산하라. Core의 추가 target 계산은 N4 재사용 시 300 request-z, 새 N4 1회 포함 시 400 request-z다. Logger 검증을 위한 추가 실행과 준비·평가 비용은 별도다.

E01·ABC·E1-A 완료 자료, prepared state, 실제 update, 로그를 먼저 점검해 재사용 manifest를 작성하라. z step 수는 기존 stdout에서 복원 가능한 부분을 먼저 추출하라. 4-cell 분석, L4/L5×Early/Middle/Late 전체 재계측, 역사적 10k trajectory의 byte equality, full-GGN·PCG·ODE/barrier 구현을 이 pilot의 선행조건으로 요구하지 말라.

**3. 실행에서 반드시 맞춰야 할 비교 조건**

- 각 분기는 동일 entry W/M/RNG에서 시작한다. D4는 실제 저장 weight 차이로 정의하고, 첫 N4 호출이 history를 append했으면 분기 전에 entry M을 복원한다.
- 같은 batch의 두 fitting 사이에는 current history를 append하지 않는다. 최종 endpoint에서 선택한 각 layer에 해당 batch history를 한 번만 append한다. REFIT4도 두 번 append하지 않는다. 성능에 따라 commit 여부나 endpoint를 바꾸지 않는다.
- M8는 공통 W_e에서 과거 request event/context를 봉인 B100·FP32 순서로 한 번 재인코딩해 만든다. FULL8/RES8에 같은 M8를 사용하고 α별 전수 재구성을 하지 않는다. 이후 suffix는 append-only다. 추가 과거 입력 접근과 준비비용을 기록한다.
- FULL8은 **동일 entry·재구성 history의 BLUE-style reference**다. 역사적 native BLUE trajectory의 재현이라고 부르지 않는다. Donor는 physical L8 mapping을 쓰며 singleton index와 혼동하지 않는다.
- Original-target replay, same-host fresh-native, same-host target replay를 구분한다. 서로 다른 mode의 차이를 후보 효과로 집계하지 않는다.
- Official P/N, general, audit, Future N을 optimizer·online α/donor 선택·품질 기반 fallback에 넣지 않는다. 수치 오류·실행 실패는 기록하고 entry를 복원한다. 조용히 native로 대체하지 않는다.
- Adapter source와 실제 import SHA, model/config, capsule/panel hashes를 실행 전에 결속한다. 기존 CPU fixture와 필요한 소형 검증을 재사용한다. Passive logger가 native 동작을 바꾸면 logger를 수정하거나 기존 로그 기반으로 진행한다.

**4. Claim별 대조를 먼저 고정하고 결과에 맞게 주장 범위를 적어라**

| Claim | 필요한 대조 | 결과 해석 |
|---|---|---|
| Write 축소로 편집 품질 대비 보존을 개선할 수 있음 | S875/S75 vs N4 | 작은 품질 손실을 동반한 이득도 trade-off claim으로 허용 가능. Scalar 결과만으로 donor 기여를 주장하지 않음 |
| 두 번째 fitting이 축소로 잃은 품질을 회복하면서 보존 이득을 일부 남김 | RES8 vs S75 및 N4 | 완전한 품질 회복·모든 N panel 동시 개선을 필수로 하지 않음. 단순 under-edit 설명에서 얼마나 벗어났는지 수치로 제시 |
| 다른 layer의 보완 효과가 같은 layer 재피팅보다 유리함 | RES8 vs REFIT4 | 같은 target-call budget에서 비교. 비슷하면 cross-layer 고유 기여는 미확인으로 보고하고 추가 fitting 수준으로 claim을 좁힘 |
| L4 축소를 포함하는 정책이 full-L4→L8보다 유리함 | RES8 vs FULL8 | 특정 품질·보존 trade-off 우세만 있으면 그 범위로 허용. 이 대비가 안 좋아도 N4 대비 다른 이득은 별도 판정 |

Historical 개발 NS를 주요 요약값으로 보되 Current 변화, R/P 전이와 NLL tail, general, 실제 비용을 함께 판단하라. 근거가 비슷하면 낮은 online 비용, 적은 추가 정보, 적은 변경 순으로 선호하라. S875가 RES8과 비슷하면 scalar를, REFIT4가 유리하면 같은 layer 재피팅을 후보로 채택할 수 있다. 결과를 donor 우월성에 억지로 맞추지 말라. 새로운 발견은 탐색적 발견으로 표시하고, novelty나 전체 최적성까지 입증됐다고 쓰지 말라.

**5. 평가와 다음 실행의 범위를 정하라**

여섯 endpoint 모두 Current R100/P200/N1000, Historical128의 R/P/N, Wiki128, MMLU 개발32를 동일 canonical MB16 evaluator로 평가하라. 개발 성능이 나쁜 arm도 예정된 static 평가 결과를 남겨라. 기술 실패는 누락시키지 말고 실패와 미측정 항목을 표시하라. 원분모, active/superseded, lost/gained, true/new NLL, signed margin, TF strict를 유지하라.

개발 결과에서 최대 한 후보를 고정한 뒤, 개발 문항과 알려진 subject/relation 중복을 제외해 미리 봉인한 N1280 audit 및 MMLU68로 N4와 비교하라. Donor 후보라면 FULL8도 audit 대조에 포함하라. 이미 본 fixed10k에서 분리한 audit는 새 corpus blind test가 아님을 명시하라. Audit를 보며 재튜닝하면 소비한 audit를 개발 자료로 재분류해야 한다. Future N은 정책 동결 이후 평가에만 사용하라.

신뢰구간이 0을 포함하거나 audit가 혼합된 결과여도 핵심 claim의 제한적 근거가 남으면 정해진 짧은 suffix를 허용할 수 있다. 반대 근거가 claim을 무너뜨리면 해당 claim을 좁히거나 비지지로 기록하라. “유의하지 않음”을 자동 실패로, “점추정 양수”를 확증으로 쓰지 말라.

허용한 한 후보의 α/layer/target/history/정보 사용을 고정하고 Middle B51–B55를 native와 각각 자기 state·fresh z로 반복하라. 총 10 batch executions이며, 첫 static batch 두 개를 정확히 재사용하면 추가 write 실행은 8개다. One-off intervention 후 native로 돌아가는 실험으로 바꾸지 말라. 매 batch 관측과 종료 시 full-seen5500을 남기고 prefix-old5000/suffix-new500 및 active overwrite를 나누어 보고하라.

L5 두 endpoint, α=.5 세 endpoint, .875 donor 대조는 해결할 구체적 질문과 실측 비용이 있을 때만 추가하라. 이들을 모두 자동 실행하지 말라. 다음 entry 확대는 Middle10·Early10·Late10의 총 60 batch executions이며 기존 Middle5를 포함한다. 초기 pilot 결과와 가용 자원을 확인해 GH가 범위를 기록하라. full10k 확증과 광범위 solver/grid sweep는 별도 단계로 남겨라. 사용자가 정하지 않은 GPU-hour cap을 만들어 사용자의 제한으로 보고하지 말라.

**6. Report는 허용 판정과 무관하게 냉정하게 작성하라**

SH는 저장소 PROTOCOL에 따라 실행 사실·수치·분모·차이·reference 초과 여부·오류·artifact만 보고한다. 성능 참고선은 이내/초과로 표시하고 자동 PASS/FAIL 판정에 사용하지 말라. GH는 SH 사실 보고서와 분리한 global report에서 해석과 최종 허용 여부를 책임져라.

모든 arm의 결과표, N4 대비 차이, claim별 matched contrast, 반대 방향 문항 전이, 품질 손실·tail·general 악화, 신뢰구간·소형 표본 한계, source/history 차이, 실패·미측정을 남겨라. 측정하지 않은 값은 NOT_RECORDED로 표시하라. 개선된 metric·entry만 골라 쓰거나 사후에 분모를 바꾸지 말라. 최초 계획 대비 변경 사항과 시점도 남겨라.

비용표는 준비, online policy, 진단, 평가, 실제 총 지출을 구분하라. 공유 L4 target과 M8 준비를 연구 총 지출에 중복 계상하지 않되 각 policy에 필요한 비용은 빠뜨리지 말라. Native 대비 같은 host의 실측 비율, warm replay의 과거 입력 접근, 준비비용 포함 상각값과 steady-state를 함께 적어라. 2배 초과라도 의미 있는 claim은 허용할 수 있지만 실측 근거 없이 “저비용”이라고 쓰지 말라.

각 최종 판정에는 다음 항목을 포함하라.

1. 판정과 **현재 허용하는 정확한 ours claim 한 문장**.
2. 이를 지지하는 matched comparison과 수치·분모.
3. 반대 근거, 손실·비용, 넘은 참고선, 불확실성.
4. 그 한계를 감수하고 허용한 이유 또는 비지지 이유.
5. 아직 주장할 수 없는 내용과 이를 확인할 최소 후속 비교.

보고 문장 예시는 다음 형식을 따른다. “범위를 한정해 허용한다. [후보]는 [비교 대상·entry·panel]에서 [관측 이득]을 보였지만 [품질 손실·불확실성·비용]이 동반됐다. 따라서 현재 지지되는 주장은 [한정된 claim]이며, [cross-layer 고유 기여/저비용/full10k 안정성 등 미확인 항목]은 확인되지 않았다.” 대괄호는 실측 결과로 채우고 성과를 예단하지 말라.

**7. GH의 실행 관리와 산출물**

담당 SH에 entry·source·host·run ID·write scope를 명시한 실행 envelope를 발행하라. 구현은 기존 adapter 연결과 pilot runner에 필요한 변경으로 제한한다. 위임할 source 범위는 project/run_scripts/baseline_mechanism_first/ 및 project/run_scripts/low_cost_write_donor_pilot/이며, 전용 codex/ branch에서 기존 사용자·다른 작업의 변경을 보존하라. 서버별 보고는 해당 SH 영역, global 판정과 정식 설계 변경은 GH 영역에 남겨라. 실행·push·merge는 기존 역할별 절차로 GH가 관리하라.

최소 산출물은 evidence-reuse-manifest, comparison-capsule, endpoint-metrics, paired-transitions, quality-frontier, history-provenance, compute-ledger, claim-decision 및 선택 시 policy-lock·suffix-summary다. 각 근거는 source/config와 artifact 경로로 추적 가능하게 하라. 큰 tensor·checkpoint·raw log는 ignored local 경로에 두고 Git에는 compact manifest·수치표·보고서를 남겨라.

최초 회신은 재사용 가능한 자산, 새로 계산할 core 범위, 실제 실행 환경·비용 추정, 첫 산출물을 언제 확인할 수 있는지로 작성하라. 준비가 끝나면 P0→P1을 진행하고, 위 원칙에 따라 후보를 허용하면 audit와 고정 5-batch suffix까지 이어가라. 원래 수치 참고선을 모두 충족하지 못했다는 이유로 연구를 정지시키지 말고, 지지되는 claim의 범위와 한계를 결과로 보여 달라.

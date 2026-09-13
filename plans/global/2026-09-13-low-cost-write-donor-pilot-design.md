**첫 실험은 Middle의 동일 L4 entry에서 여섯 endpoint를 비교한다.** 기존 native·축소·E01 관측을 재사용하고, 약한 축소와 L8 보완이 같은 L4 재피팅보다 유리한지를 확인한다. 유망한 정책 하나를 고정한 뒤 native와 각각 5 batches를 진행한다. 첫 실행을 L4/L5×3 entry 전체 재계측이나 full-GGN solver 개발로 시작하지 않는다.

작성: 2026-09-13. 상태: **첨부 검토 및 구체 설계안. 이 문서 작성으로 GPU·모델 실험이나 job 제출을 실행하지 않았다.** 같은 날짜의 사용자 지시에 따라 판정을 개정했다. **수치 기준은 진단용 참고선이며, 비교가 유효하고 ours claim이 적절히 드러나면 그 범위의 주장과 후속 실험을 허용한다.** 손실·비용·반례·불확실성은 보고서에 그대로 남긴다. 비용 참고선은 사용자가 지정한 GPU-hour 한도나 방법의 성능 보장이 아니다. 실행 인계는 [GH 지시문](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-13-low-cost-write-donor-pilot-gh-instruction.md)을 함께 사용한다.

검토 대상은 [새 첨부](/mnt/raid5/janghj/.codex/attachments/9dc9c28c-38c7-4dbf-82d8-ccb3c4e63fe8/pasted-text.txt)다. 기존 [baseline mechanism 설계](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md)의 첫 실행 범위를 최신 evidence에 맞춰 좁힌 companion plan이다. [이전 첨부 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-13-server4-fixed10k-attached-analysis-review-ko.md)의 capacity·지표·인과 해석 경계를 유지한다.

**첨부의 방향은 채택하되, 이미 진행된 부분을 반영해 시작점을 바꾼다.**

| 첨부의 판단/제안 | 이번 확인 | 설계 반영 |
|---|---|---|
| A-OS Middle 완료, 보호 solve 6.44h, PCG 미수렴 | 보고서·비용·PCG CSV 일치 | 현재 full-GGN 경로의 확대와 PCG iteration 증가를 기본 실험에서 제외 |
| E1-A 문항 연결 완료 | 10 singleton arms의 CPU 분석 완료 | 원장 재작성 대신 새 endpoint/telemetry를 기존 identity에 연결 |
| E01은 일부 관측이고 B020 replay nonexact | 맞음. 이후 B060/B100 replay Current와 상세 관측도 로컬에 존재 | 최신 파일을 재사용하되 full-seen/최종 receipt 완료와 구분 |
| z의 실제 step/stop이 미계측 | Structured 필드는 누락. 정상 완료 stdout에서는 상당 부분 CPU 복원 가능 | step 계측만을 이유로 여섯 native cell을 다시 돌리지 않음 |
| α=.5/.75 축소를 처음 screen | ABC에 세 entry의 해당 결과가 이미 있음 | .5는 under-edit 대조, .75는 품질 경계 대조로 재사용 |
| L5/L8×세 α의 6 donor endpoints | 질문은 적절하나 한 번 더 fitting하는 효과가 섞임 | L8부터 시작하고 같은 L4 second-fit을 추가. L5는 조건부 확장 |
| NS만으로 방법을 선택하지 않음 | Downstream의 task/평가 방식별 순위 차이 확인 | 작은 general sentinel과 선택 후 audit를 포함 |

자료 snapshot의 origin/main은 153161855468c1a72ad07cc9aca6201444bade99다. 현재 공유 worktree HEAD ddc178584ef14efd5d4e1271b3c324e3ebd3e443 및 로컬 main 627139777a347f543f82749ef58cf4da82116031과 구분한다. 첨부의 sandbox 설계서/zip 자체는 이번 로컬 자료에 없으므로 첨부 본문과 실제 repository 근거에서 설계를 작성했다.

**기존 축소 결과는 이미 품질 저하의 위치를 알려 준다.**

| Entry | Native Current RS/PS | α=.75 RS/PS | α=.5 RS/PS |
|---|---:|---:|---:|
| Early | 100 / 195 | 100 / 191 | 97 / 180 |
| Middle | 100 / 197 | 100 / 195 | 99 / 185 |
| Late | 100 / 193 | 100 / 188 | 98 / 174 |

분모는 R100/P200. 기존 α=.75는 RS를 유지해도 PS가 −2/−1/−2.5%p이며, α=.5는 명확한 under-edit 신호다. 따라서 처음 질문은 “아무 축소나 해 보자”가 아니라 **약한 축소에서 품질 손실을 피할 수 있는지, 품질을 잃는 축소에서는 두 번째 fitting이 그 손실을 되돌리면서 N 이득을 남기는지**다. [기존 ABC 축소](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/experiment-reports/servers/server1/single-layer-cumulative-risk-abc-2026-09-10-v1/A/paired-summary.csv:57).

ABC 축소는 curve panel이다. N은 각 panel 200개이며 Fixed/Past PS도 빠져 있다. 이를 full N/P 관측으로 대체해서 쓰지 않는다. 보존 tensor 또는 정확한 prepared state·actual delta가 있고 source/target/entry identity가 일치할 때 endpoint를 재사용한다. 파일명에 같은 α가 있다는 이유로 새 native의 대조로 삼지 않는다.

A-OS의 23,174.48초와 실제 PCG relative residual 0.110079/0.600044는 완료된 실행의 미수렴 결과다. A0→A-OS paired 비교와 SH2 N4/B-OS aggregate 재사용의 검증 수준을 구분하며, cross-hardware parity까지 확인됐다고 쓰지 않는다. [A-OS 근거](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-aos-cpu-review-recall-v1/experiment-reports/servers/server1/multilayer-joint-edit-a-2026-09-11-v1/middle-aos-review-recall-v1/factual-report-ko.md:5).

Downstream의 고정 MMLU100 alternative weighted F1은 BLUE 50.0355, L4-only 42.9751이다. Generation에서는 반대 순서지만 원 parser의 invalid 문제도 있다. Full MMLU 결과나 지식 소실률로 읽지 않는다. 원 Git object a9e0f7babe70dae252938210d93b3523bb7f7ca1의 [보고서 보존본](/mnt/raid5/janghj/ODE-edit/local/reviews/low-cost-write-donor-design-2026-09-13/downstream-diagnostic-report-ko.md)과 [출처 receipt](/mnt/raid5/janghj/ODE-edit/local/reviews/low-cost-write-donor-design-2026-09-13/downstream-source-receipt.json)를 사용한다.

**P0는 새 장기 재개가 아니라, 재사용 가능한 비교 상태를 확정하는 작업이다.**

주 시작점은 Llama-3-8B-Instruct revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2의 L4-only 5,000-edit checkpoint W50/M50, 다음 B51의 ordinal 5000–5099다. FP32, 원 tokenizer/context/position/evaluator, AlphaEdit BLUE-style singleton L2=1을 유지한다. 세부 실행 source는 아래 E01 repair의 검증된 adapter와 봉인된 editor import를 사용하도록 고정한다.

새 비교의 기본 실행 위치는 재사용 자산과 E01 adapter가 있는 Server1 환경으로 잡는다. Server4의 기존 결과는 역사적 기준이다. 다른 host에서 실행한다면 그 host에서 native와 후보를 함께 만들고 비용도 그 host의 native로 측정한다. 한 비교 안에서 host/source를 바꾸지 않는다. 이 문서는 서버 자원 예약이나 실행 지시를 발행하지 않는다.

각 cell의 capsule은 W_e의 수정 가능 weight와 미수정 모델 provenance, M/P physical-layer mapping, context/tokenizer, request 순서, RNG, editor source/config, target mode, evaluator를 결속한다. Native endpoint N4, actual D4=N4−W_e, native target, native K/R, 종료 후 history와 비용 receipt를 붙인다.

Target mode는 다음 세 가지를 구분한다.

| Mode | 의미 | 사용 |
|---|---|---|
| original-target replay | 역사적 Server4 z를 현재 host의 writer에 주입 | target 고정 진단 및 기존 ABC 재사용 |
| same-host fresh-native | 실제 W_e에서 현재 pinned compute-z를 다시 계산 | 새 정책 비교의 주 native |
| same-host target replay | 바로 위 native가 저장한 z로 같은 writer를 재검산 | target을 고정한 solve/adapter 대조 |

서로 다른 mode의 endpoint를 후보 효과로 빼지 않는다. 주 native capsule과 정확히 같은 조건의 기존 E01 산출물은 재사용 가능하다. Source/입력/target mode가 결속되지 않으면 한 native B100을 새로 계산해 capsule을 만든다. 비용이 큰 장기 trajectory의 historical byte equality를 연구 전체의 선행조건으로 두지 않는다.

E01의 최신 파일에는 B060/B100의 원본 대비 W 상대차 0.0821363/0.0743356과 Current R/P/N 100/196/726, 100/193/653이 있다. 분모는 100/200/1000이다. Full-seen 6k/10k가 아니며 검사 시 최종 full-seen/receipt 완료는 확인되지 않았다. [Middle 상태](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1/local/baseline-mechanism-first-e01/20260912-v1/attempts/warm-l4-n5000-performance-r2/output/resume_fidelity.json), [Late 상태](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1/local/baseline-mechanism-first-e01/20260912-v1/attempts/warm-l4-n9000-performance-r2/output/resume_fidelity.json).

B051/B091에는 K/R/dense solve/actual fitting/full-position/signed/general 관측이 이미 있다. Dense 재계산은 captured native update와 일치하고, original/recomputed z는 다르다. 이를 solver 오류 또는 hardware 원인으로 미리 확정하지 않는다. PAIR2 diagnostic 관측을 canonical MB16 metric의 parity 검증으로 대체하지 않는다. [Middle 관측](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1/local/baseline-mechanism-first-e01/20260912-v1/attempts/warm-l4-n5000-performance-r2/output/B051/observations/observation-receipt.json).

정상 완료 z segment의 stdout과 pinned source를 결합하면 loss 평가 횟수−1이 Adam update 횟수다. 최대 25는 Adam 25회가 아니라 최대 24회다. 기존 n1000/n5000/n9000 10-batch 로그에서 각각 23,976/24,000/23,952 updates가 복원된다. Exact loss, clamp event, per-z 시간은 여전히 누락이다. 로그 순서·완결성과 request identity를 결속한 CPU 추출을 먼저 한다.

새 structured logger가 필요하면 한 native B100에서 passive logger 유무를 비교한다. 입력 W/M/RNG는 byte 일치, native update의 상대 Frobenius 차이 ≤1e−6, 고정 32개 R pair의 NLL 최대 차이 ≤1e−4와 성공 bit 일치를 adapter 허용 기준으로 제안한다. Logger가 이 조건을 깨면 logger를 수정하거나 기존 로그 기반으로 진행한다. Historical replay의 큰 차이를 이 허용폭으로 정당화하지 않는다.

**P1의 기본 비교는 아래 여섯 endpoint로 고정한다.**

\[
D_4=N_4-W_e,\quad S_\alpha=W_e+\alpha D_4,\quad
T_{\alpha,d}=S_\alpha+D_d(S_\alpha;P_d,M_{d,e}).
\]

D_d는 해당 상태에서 fresh local z, current K, residual R을 구해 수행하는 기존 AlphaEdit fitting이다. 첫 L4 z/D4만 공유하고 두 번째 z/K/R은 α·layer별 실제 상태에서 계산한다.

| ID | L4 첫 write | 두 번째 fitting | 주 비교 |
|---|---:|---|---|
| N4 | 1.0 | 없음 | 같은 환경의 native |
| S875 | 0.875 | 없음 | .75 실패로 모든 scalar shrink를 배제하지 않기 위한 새 약한 축소 |
| S75 | 0.75 | 없음 | 기존 신호가 있는 품질 경계 |
| FULL8 | 1.0 | L8 fresh local-z/write | 같은 entry의 full-L4→L8 reference |
| RES8 | 0.75 | L8 fresh local-z/write | 줄인 L4의 품질을 L8이 회복하면서 N 이득을 남기는가 |
| REFIT4 | 0.75 | 같은 L4 fresh local-z/write | 다른 layer의 효과와 추가 target/write의 효과를 구분 |

이는 entry를 제외한 6개 상태다. N4는 capsule 조건을 충족하면 재사용하며, S875/S75는 추가 target 최적화가 없다. FULL8/RES8/REFIT4는 각각 100 request-z 계산, 총 300개를 추가한다. 최대로 24 Adam updates/request인 기존 정책을 쓰되 실제 종료 횟수를 기록한다. 동일 호출 예산이 동일 시간·실현 품질을 뜻하지는 않는다.

α=.875는 새 결과를 보고 보간한 최적값이 아니라 이 설계에서 사전 지정한 단일 근접 대조다. 이 grid로 연속 α의 최적 frontier 전체를 찾았다고 주장하지 않는다. L4와 L8의 norm을 50:50으로 강제하지 않는다.

Actual FP32 endpoint를 기준으로 분기한다. α=0/1은 W_e/N4 snapshot을 그대로 사용한다. 다른 α는 W_e와 N4의 차이로 한 번 materialize하고 실제 D와 scaling 오차를 기록한다. Nominal solve update와 stored-weight 차이를 혼동하지 않는다. 분기마다 W와 M/RNG를 모두 복원한다.

Native 함수를 호출해 N4를 얻을 때 history가 이미 append될 수 있다. Fan-out에는 entry M을 사용하고, 같은 batch의 두 번 fitting 사이에 M을 append하지 않는다. 최종 accepted endpoint에서 각 선택 layer에 해당 batch key Gram을 정확히 한 번만 더한다. REFIT4에서도 동일 request를 두 번 append하지 않는다. 이것은 두 subwrite를 한 batch transaction으로 보는 정책이다.

**Donor history는 공통 entry에서 한 번 준비하고 α 사이에 고정한다.**

L4-only warm state의 L8는 W0 weight지만, 이전 L4 edits가 만든 L8 key geometry는 W0와 다르다. M8를 0으로 두거나 역사적 BLUE M8를 가져오지 않는다.

\[
M^{\mathrm{recon}}_{8,e}
=\sum_{b<t} K_8(W_e;\mathcal C_b)K_8(W_e;\mathcal C_b)^\top.
\]

과거 모든 request event와 봉인 context/key 평균 규칙을 사용하고, B100 단위 FP32 append 순서를 고정한다. Active-fact만 남기는 변화는 기본 실험에 넣지 않는다. 같은 W_e에서 얻은 M8를 FULL8/RES8에 공통으로 사용하며, α마다 Sα에서 전수 재구성하지 않는다. 후자는 shrink와 history refresh를 함께 바꾸는 별도 실험이다.

이 M8는 모든 과거 keys를 현재 W_e에서 읽은 재구성 Gram이다. 각 과거 시점의 모델에서 저장한 chronological native M8와 같지 않다. FULL8은 **동일 entry·재구성 history의 BLUE-style reference**이며 역사적 BLUE chain 재현이 아니다. 기존 A0 준비 M8는 W_e/P/context/event order/누적 dtype identity가 일치할 때만 재사용한다.

과거 request/context 재접근은 native M-only보다 많은 정보·준비 비용이다. Warm replay를 명시하고 online method 비용과 함께 공개한다. L5 확장 시에도 같은 정책으로 M5를 준비한다. P는 full-stack physical L4/L5/L8 mapping을 사용하며 singleton index0을 physical layer0으로 읽지 않는다.

**평가 패널은 개발용 관측과 후보 선택에 쓰지 않은 audit로 나눈다.**

| Panel | 크기/구성 | 사용 |
|---|---|---|
| Current | 자연 next B100의 R100/P200/N1000 | 여섯 상태 모두 canonical MB16 평가 |
| Historical | 기존 E01의 성능과 무관하게 고른 128 requests R/P/N | 여섯 상태 모두; active/superseded와 lost/gained 분리 |
| General sentinel | 기존 Wiki128의 고정 token/mask에 대한 TF NLL | 여섯 상태 모두; 기존 z 최적화 입력으로 사용하지 않음 |
| General task | 기존 MMLU100 중 고정 hash로 정한 32 rows, alternative scorer | 여섯 상태의 개발 경보; weighted F1보다 정수 정답/오답을 먼저 표시 |
| Prospective audit | 새 후보 선택에 사용하지 않을 고정 128 requests의 N1280 및 MMLU의 나머지 68 rows | 후보 하나를 고정한 뒤 N4/선택 후보 비교. FULL8은 donor 후보일 때 추가 |
| Future N | suffix 평가용 고정 문항 | 선택·학습·fallback에 사용하지 않고 policy 동결 후만 공개 |

Current+Historical은 2,964 prompt pairs다. Query/signed diagnostic의 PAIR2 결과 대신 동일 canonical evaluator로 비교한다. 중복 prompt는 계산 cache를 재사용할 수 있지만 panel별 분모와 identity는 유지한다.

Prospective N audit는 개발 panel의 case IDs를 제외하고 hash seed 20260913으로 128개를 고정하며 가능한 canonical subject/relation overlap도 제외한다. 제거 규칙·잔여 수·판정 불가능한 semantic overlap을 기록한다. 이미 읽은 fixed10k에서 뽑는 경우 “새 후보 선택에 미사용”이라는 뜻일 뿐 새 corpus blind test가 아니다. 별도 미관측 fact-disjoint 자료가 있으면 최종 확증에 사용한다. Audit 결과로 같은 candidate를 계속 retune하면 그 audit를 개발 자료로 재분류하고 새 평가 분리를 마련한다.

MMLU 개발 32와 audit 68은 exact row 단위로 분리한다. 전체 100 결과를 함께 보고할 때는 개발 항목을 포함한 descriptive aggregate임을 명시한다. 과거 baseline에서 이미 공개된 corpus라는 한계도 유지한다. 원 generation parser의 invalid 문제를 해소하기 전에는 generation weighted F1을 방법 채택 기준으로 쓰지 않는다.

기존 official P/N는 이미 관측한 탐색 자료로만 사용한다. Runtime α/donor 선택이나 z/objective에 P/N·general/audit를 넣지 않는다. 이 pilot의 online policy는 고정 α·고정 layer다. 개발자는 개발 패널에서 후보를 선택할 수 있지만 그 점수는 선택 후 독립 검증 성능으로 보고하지 않는다. Future N은 개발 선택에서도 제외한다.

W0→entry, entry→post, post→future를 같은 문항별로 기록한다. R/P desired margin은 true NLL−new NLL, N은 new NLL−true NLL이며 tie는 실패다. Raw 저장 margin과 부호를 구분한다. NS뿐 아니라 true/new NLL 각각, W0-success/entry-success lost·gained, TF strict를 보존한다. 의도적 overwrite는 원분모와 active-fact 보조표를 함께 보고한다.

**P1은 claim에 맞는 근거를 종합해 판정하며, 수치 참고선의 일괄 충족을 요구하지 않는다.** 다음 값은 손실을 빠뜨리지 않고 표시하기 위한 개발용 참고선이다. 단일 항목 초과나 여러 항목의 미충족을 자동 탈락 조건으로 구현하지 않는다. 통계적 동등성을 증명하는 기준도 아니다.

| 항목 | N4 대비 진단용 참고선 — 자동 탈락 기준 아님 |
|---|---|
| Current RS | N4가 성공한 R에서 새 실패 0개 |
| Current PS | 순성공 감소 ≤2/200; 양방향 문항 전이는 별도 보고 |
| Current R/P new NLL | category별 paired 평균 증가 ≤0.10 nats/token |
| Current R/P tail | category별 paired NLL 증가의 q95 ≤0.50 nats/token |
| Active Historical R | N4-success 중 새 실패 0개; 분모·superseded 별도 |
| Active Historical P | 순성공률 감소 ≤1%p, new NLL 평균 증가 ≤0.10 |
| TF strict | Current R 감소 ≤1/100, Current P 감소 ≤4/200 |
| Wiki128 general | mean NLL 증가 ≤0.05 nats/token |
| MMLU32 개발 경보 | 정답 순감 ≥2개이면 경보로 표기하고 원문 응답을 확인; 일반능력 동등성 주장 금지 |

Tail·개별 손실은 작은 표본의 기술통계이며 가능한 신뢰구간과 함께 해석한다. 참고선을 넘은 arm도 끝까지 예정된 static 평가를 하고 결과를 남긴다. 성능을 보고 실행 중 α나 cap을 바꾸지 않는다. 입력·source·target mode·history·평가 identity의 비교 유효성 요건은 유지한다. 유효하지 않은 비교는 좋은 점수만으로 허용하지 않고 해당 부분을 수정한다.

Historical NS +1%p, Current NS 비악화, online 비용 1.5배/2배는 참고선이다. 소수 R/P 손실, Current N 일부 감소, +1%p 미만 개선, 신뢰구간의 0 포함, 비용 참고선 초과만으로 탈락시키지 않는다. Historical/Current N, R/P 성공 전이와 NLL tail, general, 추가 정보·비용을 함께 보고 **얻은 이득이 어떤 손실을 대가로 하는지, 그 trade-off가 어떤 ours claim을 지지하는지**를 GH가 설명한다. 작은 품질 손실을 동반한 보존 개선이나, 동일 호출 예산에서의 다른 layer 보완 효과도 한정된 주장으로 허용할 수 있다. 핵심 이득이 관측되지 않거나 대조군으로 설명되는 경우에는 그 주장을 지지한다고 쓰지 않는다.

실행 전에 아래 claim별 비교를 기록하고, 실행 후에는 지지되는 범위로 문장을 좁힌다. 사후 발견은 탐색적 발견으로 표시한다.

| Claim | 주 비교 | 허용 가능한 해석과 한계 |
|---|---|---|
| 약한 native write 축소로 보존/편집 품질의 trade-off를 개선할 수 있다 | S875/S75 vs N4 | 제한적 R/P 손실이 있어도 이득과 손실을 공개한 trade-off claim 가능. Scalar 결과만으로 새로운 donor method의 기여를 주장하지 않음 |
| 줄인 L4 write 뒤 추가 fitting으로 품질을 회복하면서 보존 이득을 일부 남길 수 있다 | RES8 vs S75 및 N4 | 완전 회복이나 모든 N panel 동시 개선을 필수로 요구하지 않음. 단순 under-edit 설명을 어느 정도 벗어났는지 명시 |
| 다른 layer를 쓰는 것이 같은 layer를 한 번 더 fitting하는 것보다 유리하다 | RES8 vs REFIT4 | 같은 target-call budget 아래 품질·보존·비용을 함께 비교. 비슷하면 cross-layer 고유 이득은 미확인 |
| L4 축소를 포함한 두 단계 정책이 full-L4→L8보다 유리하다 | RES8 vs FULL8 | 특정 보존/품질/비용 trade-off에서만 우세해도 해당 범위로 주장. FULL8은 역사적 BLUE trajectory와 구분 |

판정은 **허용 / 범위를 한정해 허용 / 추가 확인 필요 / 현재 근거로 비지지 / 비교 무효**로 남긴다. 허용은 관측된 범위의 claim 채택과 정해진 후속 실험 진행을 뜻한다. 모든 지표 PASS, 통계적 유의성, 무손실, 저비용, full10k 안정성이 자동으로 성립한다는 뜻은 아니다. 비교가 유효하고 핵심 대비에서 해석 가능한 이득이 보이면 허용을 기본으로 검토하되, 무엇을 근거로 손실을 수용했는지 기록한다. 하나의 NS 최고값만으로 판정하지 않는다.

N4와 후보의 prospective audit 비교에서 NS 점추정 ≥+1%p, paired request-cluster bootstrap 95% 구간 하한 >0은 “뚜렷한 국소 신호”의 보조 표기다. 이를 허용의 필수 조건이나 선택 편향까지 해결한 최종 확증으로 쓰지 않는다. 구간이 0을 포함하면 불확실성을 명시하며, audit가 혼합되거나 소폭 음수여도 다른 근거와 함께 제한적 claim이 남는다면 사전 정의된 한 후보·5-batch suffix를 허용할 수 있다. Audit가 claim을 실질적으로 반박하면 그 claim을 좁히거나 비지지로 남긴다. Audit를 보며 같은 후보를 재튜닝하지 않는다.

후보 선택은 Historical 개발 NS를 주요 요약값으로 삼되 claim별 대조, Current 변화, 품질 손실과 general, 실제 비용을 종합한다. Current NS 비악화를 적격 조건으로 쓰지 않는다. 근거가 비슷한 후보는 낮은 online 비용 → 적은 추가 정보 → 적은 변경을 선호한다. Historical NS 0.5%p 차이는 동률 판단의 참고선일 뿐 기계적 순위 규칙이 아니다. 특히 S875가 RES8과 비슷하면 scalar shrink를 먼저 진행한다. REFIT4가 RES8과 비슷하면 cross-layer 고유 이득을 주장하지 않는다. FULL8 대비 이득이 없으면 full-L4→L8 대비 우월성 주장을 보류하되 N4 대비 별도 이득까지 부정하지 않는다. 최대 한 후보만 고정하며 승자를 만들기 위해 사후 metric·분모·참고선을 바꾸지 않는다.

SH는 모든 arm의 원시 수치·차이·참고선 초과 여부와 기술 실패를 사실 보고서에 기록한다. 성능 참고선은 PASS/FAIL 대신 참고선 이내/초과로 표시한다. GH는 별도 global report에 판정, 허용한 정확한 claim 문장, 지지 근거, 반대 근거, 허용 이유, 미확인 사항과 다음 비교를 적는다. 허용 결과도 냉정하게 보고하며, 불리한 arm·entry·문항이나 준비비용을 숨기지 않는다.

**분담의 결과는 parameter 강도 이동으로 해석한다.** 이 실험도 두 layer가 Current B100 전체를 처리하고 각각 전체 batch history를 append한다. 요청 수·history 수를 분담하지 않으며, α만큼 capacity가 비워졌다고 말하지 않는다.

고정 N margin m에서
\[
m(T_{\alpha,8})-m(N4)
=[m(S_\alpha)-m(N4)]
+[m(T_{\alpha,8})-m(S_\alpha)]
\]
는 정확한 차이 분해다. 축소가 얻은 N 이득을 donor가 얼마나 유지/소모했는지 문항별로 본다. 같은 문항의 R/P 품질 회복과 N 손실/회복을 연결한다. α마다 donor z/K/R이 달라지므로 고정 L8 update의 독립 효과나 전체 lifelong 인과 기여율은 아니다.

RES8만 품질을 회복하면 donor 참여 신호, N4·FULL8 대비 보존 이득까지 남으면 해당 두 단계 정책의 유용성 신호다. REFIT4보다도 유리하면 같은 target-call budget 아래 다른 layer의 역할이 더 강하게 지지된다. 그래도 탐색 공간 전체의 최적성·intrinsic capacity는 증명되지 않는다.

**P2 확장은 필요한 질문만 연다.**

| 조건 | 추가 상태 | 수 |
|---|---|---:|
| L8에 국소 신호가 있거나 조합에 따른 차이가 핵심 질문 | FULL5, RES5: 동일 L4 entry·α=1/.75 후 L5 | 2 |
| .75의 donor 회복 양상이 분명해 더 큰 L4 축소를 시험할 이유 | S50, RES50-d, REFIT50-4; d는 이미 고정한 donor | 3 |
| Proxy와 실제 출력의 불일치가 선택을 막음 | 기존 signed 자료에서 소형 위치별 intervention | 별도 고정 subpanel |
| 정적 축소/재피팅 한계가 남고 방향 신호가 있음 | 4방향 frozen-response probe | 별도 설계 후 실행, core matrix에 포함하지 않음 |

L5 singleton의 1k/5k/9k entry/targets/P는 준비돼 있지만 세 cell의 새 E01 계측을 모든 분기의 선행조건으로 요구하지 않는다. L5 donor는 **L4-only의 공통 W_e**에 적용하며 L5-only trajectory의 entry를 대신 쓰지 않는다.

RES8은 α=.75만 시험한다. S875에 donor를 붙인 대조는 core에 없으므로 RES8 실패를 “가벼운 축소+donor도 실패”로 확대하지 않는다. .875에서 donor 품질 회복을 물을 구체적 이유가 생기면 해당 donor와 동일 L4 second-fit을 함께 추가하는 별도 bounded 비교로 정의한다.

**P3는 one-off 복구가 아니라 반복 정책의 5-batch suffix다.**

최대 한 후보의 α/layer/단계 순서/정보 사용/history 정책을 동결한다. N4와 후보가 공통 Middle W50에서 B51–B55를 각각 자기 state로 진행한다. Scalar면 매 batch fresh L4 native update를 같은 α로 축소한다. Donor면 매 batch fresh L4→부분 적용→fresh donor z/write를 반복한다. REFIT4가 선택되면 매 batch 같은 L4에서 두 단계 fitting한다.

첫 static endpoint가 정확한 source·policy·history·RNG를 포함하면 첫 batch 결과로 재사용한다. 이후 native future-z cache를 후보에게 주입하지 않는다. 후보별 다음 z는 자기 실제 state에서 계산한다. No-adaptation 기본 정책이며 성능을 보고 α/donor를 바꾸거나 거절 후 다른 endpoint를 골라 쓰지 않는다.

수치 오류·비유한 update·실행 실패는 실패한 batch로 기록하고 원자적으로 entry를 보존한다. 이를 조용히 native로 대체해 성공한 candidate 정책처럼 집계하지 않는다. 별도 fallback을 연구하려면 사전에 정책과 비용을 다시 정의한다.

M4는 최종 endpoint에서 한 번 append한다. Donor M은 warm-start 재구성 이후 매 batch 최종 key를 한 번 append하며 과거 전수 refresh는 하지 않는다. 이후 L4 변화에 대해 상위 과거 key가 stale할 수 있다는 명시적 근사다. Append 정책과 warm replay 준비 비용을 보고한다. One-off intervention 후 L4-only로 진행하는 실험은 별도 지속성 질문이며 이번 repeated policy와 섞지 않는다.

매 batch Current100과 고정 Historical/General sentinel을 평가하고, suffix 끝에서 full-seen 5,500 requests를 평가한다. 동일 고정 N의 W0/entry/post/future 전이를 추적한다. Full-seen의 대부분이 공통 과거라는 점 때문에 suffix-new500과 prefix-old5000의 R/P retention을 별도 집계한다. 모든 과거 target의 동시 보존을 요구하지 않고 active overwrite를 구분한다.

두 policy×5 batches=10 batch executions다. Static first batch 두 개가 재사용되면 추가 write 실행은 8개다. 유망하면 Middle10까지 이어가고 Early/Late10을 더해 총 2×3×10=60 batch executions로 넓힌다. 이미 수행한 Middle5는 총수에 포함한다. 완료된 E01 native suffix는 source/state/RNG/policy identity가 같은 경우만 reuse한다.

최종 full10k 확증은 별도 단계다. 동일 host의 paired native 대비 RS −0.5%p, PS −1.0%p, NS +3%p는 장기 engineering 목표의 참고값으로 유지하며, pilot 허용이나 향후 claim의 자동 탈락 기준으로 쓰지 않는다. 최종 주장 강도는 실제 효과·손실·비용·반복 근거에 맞춘다. Pilot 참고선을 통계적 noninferiority margin으로 옮기지 않는다. Order 반복은 학습 경로 변동을, request-cluster bootstrap은 한 경로의 평가항목 변동을 다룬다. 이미 본 fixed10k의 새 order는 blind corpus 검증이 아니다.

**비용은 공유된 연구 지출과 실제 policy 비용을 따로 계산한다.**

| Ledger | 포함할 항목 |
|---|---|
| 준비 | model restore/load, P/C0, donor history replay, panel/teacher 생성 |
| Online policy | 매 batch z/K/solve/commit/history 및 runtime에 필요한 모든 선택·보호 연산 |
| 진단 | passive telemetry overhead, signed/FD/position probe, spectrum |
| 평가 | Current/Past/N/general/full-seen |
| 실제 지출 | 공유 준비·copy·I/O·실패/취소 allocation을 포함한 총 wall/GPUh |

새 P1의 S875/S75는 target 재계산 없이 만들지만 배포 scalar policy의 z 비용이 0이라는 뜻은 아니다. FULL8/RES8은 L4+donor 두 target 단계 비용을 각각 부담한다. Study 안의 공통 L4 z·M8 준비 재사용은 총 연구 지출에서 한 번만 세고, policy 원가에서는 필요한 비용을 빠뜨리지 않는다. Host copy byte counter는 bandwidth가 아니다.

Online 시간이 같은 환경의 fresh L4 native 대비 1.5배/2배인지 참고 구간을 표시하되 자동 허용·탈락 gate로 쓰지 않는다. 2배를 넘어도 추가 비용에 비해 의미 있는 보존/품질 이득 또는 한정된 기전 claim이 남으면 정해진 후속 범위 안에서 허용할 수 있다. 반대로 실행이 싸다는 이유만으로 이득 없는 정책을 채택하지 않는다. 저비용 claim은 실측 비용이 지지할 때만 사용하고, 비교 기준과 준비비용 상각 조건을 밝힌다. 이는 job timeout이나 사용자 GPU-hour cap이 아니다. 준비비용을 suffix 길이에 상각한 값과 준비비용 제외 steady-state를 함께 공개한다.

Server4의 평균 284.57초/B100이나 Server1 E01 813.32초/B100을 새 실행 예약 시간으로 사용하지 않는다. 첫 실제 native와 donor 단계의 관측 시간·메모리·복사량으로 이후 범위를 산정한다. 비용 허용 범위가 정해지지 않은 채 모든 conditional arm을 실행하지 않는다.

**구현은 기존 adapter를 연결하는 범위로 제한한다.**

| 책임 | 재사용 위치 | 필요한 변경 |
|---|---|---|
| Capsule/복원 | E01 fixtures, continuation receipts | selected layer 전체 W/M/RNG 복원, target mode와 hash 결속 |
| Native 관측 | native_runner, observer | 원 stdout CPU 추출, 누락된 z structured telemetry만 보완 |
| Static arm | 기존 ABC amplitude·snapshot | actual N4 기반 α materialization, branch별 rollback |
| Second fitting | 봉인 AlphaEdit compute-z/K/solve | donor physical P/M mapping, 중간 append 분리, 최종 once-only commit |
| 평가/집계 | terminal_performance, E1-A ledger | canonical MB16, active/history/audit panel, paired transition과 비용 join |
| Suffix | continuation | frozen policy, branch별 fresh z, donor append 정책 |

기준 adapter source는 b51dcf5ab825608bee81dd13549318d8d267e835다. [복원](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1/project/run_scripts/baseline_mechanism_first/fixtures.py:103), [native 실행](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1/project/run_scripts/baseline_mechanism_first/native_runner.py:11), [관측](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1/project/run_scripts/baseline_mechanism_first/observer.py:34), [terminal 평가](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1/project/run_scripts/baseline_mechanism_first/terminal_performance.py:68).

Warm_plan에는 L4/main-cell3/ABC 경로가 hard-coded이므로 L5 singleton을 실행할 때는 mapping을 일반화해야 한다. Terminal evaluator의 n6000/n10000 제한도 B55/n5500 suffix에 맞춰 일반화한다. Source를 여러 branch에서 가져올 때에는 실제 import 파일 SHA를 새 manifest에 결속하며 main 통합 여부로 대체하지 않는다.

필요한 실행 전 검증은 α0/1 snapshot 일치, 실제 full-position write와 observer의 일치, sibling 간 W/M/RNG rollback, current history 중복 append 없음, donor P physical mapping, 평가 패널과 runtime 입력의 분리다. 기존 검사와 작은 CPU fixture를 우선 재사용한다. 새 heavy solver나 전체 Jacobian 저장은 추가하지 않는다.

구현 단계의 산출물은 evidence-reuse-manifest, comparison-capsule, endpoint-metrics, paired-transitions, quality-frontier, history-provenance, compute-ledger, policy-lock, suffix-summary다. 정책 lock에는 α/layer/target mode/history/실패 시 동작/정보 조건이 들어간다. 이번 설계의 [cell 목록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-13-low-cost-write-donor-pilot-cells.csv)과 [판정 기준](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-13-low-cost-write-donor-pilot-contract.json)은 아직 실행 결과나 검증 완료 capsule이 아니다.

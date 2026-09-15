GH 전달 지시문 — EP-TW-1 alpha cap sweep, W0부터 SEQ1000

Instruction ID: `GH-EP-TW1-ALPHA-CAP-SWEEP-20260915-V1`

수신: Global Head(GH). 실행 담당: Server4 SH.

상태: 사용자 전달용 지시문이다. 이 파일 작성 자체로 GH/SH에 메시지를 보내거나 GPU 작업을 제출한 것은 아니다. 아래는 GH가 사용자로부터 본 지시를 전달받았을 때 수행할 작업이다.

**EP-TW-1의 cap sweep을 구현하고, cap=10/100/없음의 세 정책을 각각 편집 전 W0/M0부터 동일 1,000개 요청의 B100×10 sequential chain으로 실행하여 완료 보고하라.** 기존 cap=1(job47962)은 대조로 재사용하라. 과학적 동작을 바꾸는 공통 구현 변경 때문에 cap1 재사용이 부적절해지는 경우에만 cap1도 같은 W0→SEQ1000으로 재실행하라. 기본 신규 범위는 **3 chains/30 batches**, 조건부 cap1 재실행까지 포함한 최대 범위는 **4 chains/40 batches**다.

이 sweep에서는 **cap만 바꾼다.** N4 calibration, 고정 KL budget, 새로운 품질 허용 오차, target refresh, quality restoration, old replay, 후보 검사 조기 종료를 함께 도입하지 말라. 한 batch의 성능이나 RAW 선택 빈도를 이유로 arm을 탈락시키지 말고, 기술적으로 유효한 각 chain을 1,000개까지 진행하라. 초기 batch 완료 후 사용자 재호출을 기다리는 이전 BG-1 규칙은 이번 지시에 적용하지 말라.

먼저 다음 문서와 원시 근거를 읽고 실행 버전에 결속하라. 링크는 GH workspace 기준이며, Server4 실행 패키지에는 같은 내용의 문서·계약과 source identity가 포함되게 하라.

- [Cap sweep 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-design.md)
- [기계 판독 설계 contract](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-contract.json)
- [Cell 목록](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-ep-tw1-alpha-cap-sweep-cells.csv)
- [EP-TW-1 완료 산출물 독립 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review-ko.md)
- [Cap별 저장 tensor 기하 계산](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-alpha-cap-geometry/summary.json)

설계 JSON의 `execution_authorized_by_this_file=false`는 그 파일 작성만으로 작업을 제출하지 않았다는 뜻이다. GH는 본 사용자 지시의 범위를 반영한 **새 dispatch와 arm별 execution lock**을 발행하라. 예전 calibration dispatch를 재사용하거나 과거 승인·검증 상태를 임의로 PASS로 바꾸지 말라.

**실행할 설정은 다음 네 개로 고정한다.**

| Arm | Cap 규약 | 시작과 길이 | 이번 처리 |
|---|---|---|---|
| CAP1 | bounded, `alpha_cap=1` | W0/M0 → B001–B010 | job47962 결과 재사용; 공통 과학 동작 변경 시에만 재실행 |
| CAP10 | bounded, `alpha_cap=10` | W0/M0 → B001–B010 | 신규 1,000개 sequential |
| CAP100 | bounded, `alpha_cap=100` | W0/M0 → B001–B010 | 신규 1,000개 sequential |
| NORM_ONLY | disabled, `alpha_cap=null` | W0/M0 → B001–B010 | 신규 1,000개 sequential |

세 신규 chain은 동일한 1,000개 요청을 각각 처리한다. 신규 처리 횟수 3,000회를 서로 다른 신규 요청 3,000개라고 보고하지 말라. 실행 순서는 CAP10→CAP100→NORM_ONLY를 기본으로 하되, 앞 결과를 보고 뒤 arm의 cap·beta·threshold를 바꾸지 말라. 자원 배치는 실제 Server4 상황에 맞춰 정하되 다른 작업을 중단·덮어쓰지 말라.

모든 chain은 pre-edit W0와 native M0에서 시작한다. W50 등 warm entry, 다른 cap의 W10, 기존 cap1의 중간 상태에서 새 arm을 시작하지 말라. B002 이후에도 자기 모델에서 target·K/A·gE/gD·history를 새로 계산·갱신하라. 기존 cap1에 저장된 미래 방향을 scale만 바꿔 재생하는 것은 이번 scientific experiment가 아니다. 공유할 수 있는 것은 W0 원본, P, 고정 요청/contexts/token, original teacher 같은 불변 자산이다.

**기존 완료 run의 실제 실행 자산을 재사용하라.**

| 항목 | 기준 |
|---|---|
| Server4 기존 run root | `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1` |
| 기존 execution lock | 위 root의 `execution.lock.json` |
| Lock SHA256 | `5a19c2be919362d08b2ea80e69a406d7b6de569aade8de639036f69db5d5d8f9` |
| 실제 실행 source | `6d317bdb2660d7e9919bc3a9fb878564e9729e37` |
| W0 revision | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| Seed | `20260915` |
| Reference | `C4-WebRef-v2`, 동일 S64/Dev128 |
| Reference identity | `f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0` |
| Teacher manifest SHA256 | `f81b798f44ce626ac1e2e402ca7438363b1dd60f5681ec0ac17b9e92d096761a` |

Sample lock, contexts, config4, P와 mapping, tokenizer·kernel·dtype·microbatch 및 실제 import 경로는 위 execution lock과 연결하라. Teacher192는 완료된 job47592 자산을 재사용한다. 이전 문서의 PENDING 기록 때문에 teacher를 다시 만들지 말라. 실제 파일·identity를 확인하고, 누락 또는 손상된 자산이 있을 때 해당 범위만 복구·기록하라.

가장 가까운 N4는 기존 AlphaEdit-BLUE L4-only다. 기존 N4 B001과 cap1 보정 전 상태의 native target 100개 및 R/P/N 1,300문항 NLL은 독립 감사에서 일치했다. 전체 경로의 인과적 동등성까지 확인됐다고 확대하지 말라. 기존 N4 및 AlphaEdit/MEMIT/AlphaEdit-BLUE/MEMIT-BLUE/L4-only baseline은 비교 자료로 재사용하고, 이번 scope에서 새 baseline editing chain을 제출하지 말라. REFIT4 W50 suffix 수치를 W0 comparator로 사용하지 말라.

CAP1의 중간 checkpoint 10개가 현재 지정 경로에 없다는 사실도 반영하라. 남은 native/route tensor로 selected L4 weight가 재구성되고 기록된 hash와 일치한 것은 확인됐지만 전체 resume 상태를 재검증한 것은 아니다. 기존 결과 재사용에 중간 checkpoint 전체가 반드시 필요한 것처럼 처리하거나, 가용하지 않은 상태에서 warm resume을 시도하지 말라. Cap1 재실행 필요성은 과학 동작·비교 범위로 판단하고 이유를 남겨라.

**구현 변경은 최소 범위로 제한하라.** GH는 Server4 SH에 `project/run_scripts/bg_tw_reference/ep_tw/`의 cap config/validation/receipt 및 필요한 arm별 lock·launcher 수정 책임을 명시하라. 필요한 경우 `project/run_scripts/bg_tw_reference/`의 실행 결속 부분만 추가로 수정하라. 별도 `codex/` branch 또는 격리된 작업 공간을 사용하고, 다른 작업자의 변경을 되돌리지 말라. Native fitting·writer source, reference recipe, 별도 실험의 코드를 함께 변경하지 말라. 기존 완료 run/source/output을 덮어쓰지 말고 새 source/archive와 arm별 output root를 사용하라.

현재 `NumericalPolicy`는 수치 항목에 `math.isfinite`를 적용한다. 따라서 cap 없음을 `inf`, NaN, 0 또는 임의의 큰 유한 수로 전달하지 말라. `alpha_cap_mode="bounded"/"disabled"`를 명시하고 disabled일 때 `alpha_cap=null`을 허용하는 schema를 구현하라. Mode·null 검사와 실제 수치 항목의 finite 검사를 분리하라. 기존 cap1/10/100의 유한 값 계산 순서는 유지하라.

\[
\alpha_{norm}=\frac{0.25\|V_p-W_{entry}\|_F}{\|dA_t\|_F+10^{-12}},
\qquad
\alpha=\min(\alpha_{cap},\alpha_{norm})\ \text{또는}\ \alpha=\alpha_{norm}\;(disabled).
\]

Disabled는 min의 cap만 생략한다. 계산된 α·target·candidate의 finite 검사, exact-zero 처리, epsilon, native per-request target ball, 25% executable write trust, actual current E·strict 조건을 유지하라. Cap mode 및 실제 사용된 α, αnorm, numeric cap 활성 여부를 receipt에 기록하라. 기존 single-chain runner에는 arm당 하나의 chain lock을 전달하라. 전체 3-chain aggregate contract를 기존 single-chain validator에 직접 넣거나 validator를 포괄적으로 무력화하지 말라.

변경된 config 분기, cap1 기존 산술, cap disabled와 finite/trust 의미는 좁은 구현 확인으로 검증하라. 저장 episode와 기존 fixture를 활용할 수 있다. **사용자가 생략한 FD grid, ULP/jitter, full-gradient/VJP 진단 등을 새 GPU 선행 gate로 복원하지 말라.** 기존 생략 상태는 `SKIPPED_USER_DIRECTED / NOT_ESTABLISHED`로 계승하고 PASS로 쓰지 말라. Source·자산 결속, 실제 finite/state 처리 오류는 정상적인 구현 문제로 해결하되, 한 batch의 과학적 성능을 실행 허가 조건으로 만들지 말라.

**EP-TW-1의 나머지 규약은 고정하라.**

- L4 native proposal 1회/B100, 요청당 native target 최대24 Adam update/25 loss 평가, 원 early stop·anchor/radius를 유지한다.
- Actual stored native preview Vp에서 current canonical E와 fixed-W0 C4 S64 D의 gradient를 각각 한 sweep 계산한다. 원 residual-Euclidean projection과 frozen A를 사용한다.
- Target ball 투영 후 C를 만들고 원 write trust를 적용한다. Gradient 재계산, target refresh, 두 번째 writer stage, quality restoration, old replay를 추가하지 않는다.
- 후보는 같은 Vp에서 구성한 `RAW`, `Vp+CA`, `Vp+0.5CA`, `Vp+0.25CA`다. Scale은 correction에만 적용하고 native delta 전체를 줄이지 않는다.
- 실제 materialized 후보에서 `E(candidate) ≤ E(own RAW)` 및 RAW strict-success ID의 보존을 확인한다. 전체100요청 mean E와 ID 집합 규약을 유지하고 양의 E allowance를 추가하지 않는다.
- Feasible 후보 중 실제 D64 최소를 선택한다. 기존 byte deduplication, RAW 우선 tie 규약을 유지한다. 후보 메뉴 확장이나 최초 feasible 후보에서의 조기 종료를 함께 넣지 않는다.
- 선택이 RAW이면 native edit를 commit한다. Parent 무편집 유지·요청 제외로 바꾸지 않는다. Inner history append는0회, 최종 B100당 native finalizer는1회다.
- Accepted ledger의 기준과 exact subject/relation active/superseded 처리를 유지한다. Old ledger·공식 P/N·Dev는 observer이며 이번 controller의 추가 입력으로 쓰지 않는다.

새 scientific output은 예를 들어 `/data/janghj/ODE-edit/local/ep-tw1-alpha-cap-sweep/20260915-v1/<arm>/<attempt>/`처럼 격리하라. 실제 경로는 GH/SH가 봉인하고, 각 arm에 source/import identity, dispatch/lock, seed/context, teacher/sample identity를 남겨라. 상태 복원 시 weight뿐 아니라 M/history/RNG/ledger 및 필요한 cache 상태를 구분하라. 기술적 중단 후 **같은 arm의 완전한 자체 checkpoint에서 이어가는 재개**는 허용하되, 중복 history 등록과 policy 변경을 피하고 재개 경계를 명시하라. Weight만 복구한 상태를 full resume이라고 쓰지 말라.

각 chain은 10개 batch 완료까지 진행하라. 보정 후보의 E 실패, D 비개선, 많은 RAW 선택은 과학적 결과다. Native nonfinite, OOM, 실제 state/코드 불일치 등 기술 실패는 typed failure와 마지막 정상 commit을 기록하고 수정·재개하라. 정책을 바꿔 이어붙인 chain이나 1,000개 미만의 실행을 완주로 보고하지 말라. 한 arm의 기술 문제로 독립적으로 실행 가능한 다른 arm을 불필요하게 중단하지 말라.

**평가와 산출물은 보고서만으로 끝내지 말라.** 기존 EP-TW-1의 평가·계측 schedule을 유지하고 다음을 재계산 가능하게 보존하라.

| 구분 | 필수 기록 |
|---|---|
| 후보별 실제 행동 | αcap mode/값, αnorm/사용 α, pre/post-ball 및 FP32 actual norm, correction/native, ball hit, trust retraction, gE/gD/d/C/A와 source identity, `〈gE,C〉`·`〈gD,C〉` |
| 후보별 선택 근거 | Actual E/D, RAW 대비 ΔE/ΔD, 1차 예측, strict lost IDs, finite/feasibility/rejection 사유, 선택 ID와 모든 probe 비용 |
| Batch 관측 | Current와 accepted-old의 entry→RAW→selected R/P/N, canonical/P TF-strict, 두 P 모두 strict, new/true NLL·margin |
| Sequential 유지 | All-request W10, at-write→W10 lost/gained·NLL tail, 첫500의 W5→W10, ACTIVE/SUPERSEDED, acceptance coverage |
| Generic | S64 각 batch, Dev128 W5/W10 observer, 동일 fixed original teacher |
| 비용 | Native fitting, 별도 A map, current/S64 F/B, teacher read, 모든 rejected probe, observer, history, I/O, peak memory, 실제 allocation |
| 보존 상태 | Native/route tensor·평가 raw·checkpoint·ledger/RNG의 실제 저장 경로와 hash, 생성 당시 inventory와 현재 retention, 명시한 미저장 범위 |

주표에는 **최대 C1의 크기, RAW=0을 포함한 실제 selected 크기, nonzero selected 크기와 선택 빈도**를 따로 제시하라. All-request 원분모를 유지하고 edited/accepted-only 점수로 대체하지 말라. At-write의 서로 다른 모델을 W0 평가로 부르지 말고, policy 간 endpoint 비교와 시간상 forgetting을 구분하라.

Raw text·teacher·모델·큰 tensor는 local artifact root에 두고 Git에는 코드·compact manifest·집계·보고를 남겨라. 각 arm W10의 L4 weight와 W0/model/token/context identity를 유지해 후속 inference 평가가 가능하게 하라. Resume을 주장하려면 M/history/RNG/ledger의 실제 보존을 확인하라. 보고서 작성 당시 존재했던 checkpoint가 이후에도 있다고 가정하지 말라.

**예상 비용은 신규3 chains 약6.4 GPUh**다. Job47962의 실제 7,694 GPU초를 동일 하드웨어·평가/저장 schedule에 선형 적용한 값이며 실측이나 hard budget이 아니다. 조건부 cap1 재실행을 포함하면 약8.55 GPUh다. Teacher 준비는 재사용하고 중첩 timer는 중복 합산하지 말라. 필요한 scheduler 자원·시간·디스크는 현재 환경에서 산정해 봉인하되, 이 예상치를 임의의 과학적 탈락선으로 사용하지 말라.

Report256/Audit128/MMLU68/FutureN, 별도 N4 calibration, 새로운 reference 구축을 이번 구현·실행의 선행조건으로 두지 말라. Cap1000, 더 촘촘한 beta, scalar guard, 다른 layer, full10k와 추가 order도 이번 제출 범위가 아니다. 필요한 후속 실험은 완료 후 GH가 근거를 붙여 제안하라. 범위 밖 chain을 자동 제출하지 말라.

**SH는 사실 보고, GH는 과학적 해석을 담당하라.** SH 보고에는 실행 사실·source/config·수치·원분모·실측 비용·오류·미측정만 담고, 방법 우열·원인 추론·승격은 별도 GH global review에 작성하라. GH는 최종 endpoint, paired 문항 전이, 실제 선택된 action과 비용을 함께 검토하라.

해석 시 다음 경계를 지켜라.

- 저장 cap1 episode의 C1 기하 범위는 cap1 0.0086–0.0265%, cap10 0.086–0.265%, cap100 0.861–2.649%, NORM_ONLY 24.31–24.63%다. 새 sequential chain에서 이 범위가 유지된다고 보장하지 않는다.
- NORM_ONLY의 C025도 저장 상태에서는 약6.1%다. Cap 간 메뉴는 포함 관계가 없으므로 큰 cap의 RAW 선택을 모든 중간 action의 불가능성으로 해석하지 않는다.
- 큰 후보의 D 개선과 실제 선택된 정책의 terminal preservation을 구분한다. S64 local KL만 개선됐으면 그 범위로 보고한다.
- Current mean E와 strict ID 보존은 모든 요청 NLL·paraphrase·old edit 보존을 보장하지 않는다. NS와 true likelihood도 분리한다.
- 이번 sweep은 한 fixed order의 개발 결과다. ODE, barrier bypass, multistep의 필수성이나 parameter-free 방법의 증거로 확대하지 않는다.

첫 회신에는 **CAP1 재사용 판단, 신규3/조건부4 chain의 범위, 필요한 cap-disabled 구현 변경, 실제 실행 source·output 계획, 자원 추정**을 간결히 남겨라. 이후 범위 안의 구현·자산 재사용·봉인·제출을 완료하고, 각 arm의 1,000개 sequential 결과와 raw 산출물까지 정리하라. 최초 batch의 성능이나 미측정 downstream panel 때문에 다시 허가를 요청하거나 단일 batch pilot으로 축소하지 말라.

최종 전달물은 arm별 실행 receipt와 정확한 완료/실패 범위, source·reuse·artifact manifest, 후보·batch·cohort·baseline 비교 CSV/JSON, SH 사실 보고, 그리고 이들과 분리된 GH 리뷰다. 실제 모델 forward를 다시 하지 않은 재집계·CPU 기하 계산은 새 과학 실험 결과와 명확히 구분하라.

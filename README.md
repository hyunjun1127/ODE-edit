# ODE-Edit / PRICE

**2026-10-08: 공식 배포·실험 기준은 저장소 최상위 [official/](official/README.md)입니다.**
CAKE 형식을 참고해 ours·baseline 구현, hparams, 공통 평가기와 서버별 runner를 한 폴더에서 관리합니다.
새 baseline 실험은 `main`에 게시한 `official/` 코드와 설정을 사용하고 commit/tree SHA를 기록합니다.

현재 준비 범위는 **FT·MEMIT·AlphaEdit·AlphaEdit-BLUE·MEMIT-FE·AlphaEdit+SPHERE ×
Llama3-8B-Instruct·Qwen2.5-7B-Instruct·GPT-J-6B × CF·zsRE = 36개 본 실험 행**입니다.
각각 2,000건을 100건씩 20 batch 순차 편집합니다. Qwen BLUE 격자·강도 대조 5개 행을 추가하며,
선택된 격자 실행의 본문 재사용을 포함하면 전체 편집 chain은 40개입니다.
ours 구현은 배포 폴더에 포함하되 이번 실행 목록에서는 제외합니다.

- [전체 체크리스트·설정·평가·checkpoint·실행 순서](official/README.md)
- [방법별 hparams와 공통 실험 계약](official/hparams/contract.json), [원본 설정 SHA](official/hparams/sources.lock.json)
- [서버별 EasyEdit 자산 연결과 runner](official/runners/README.md)
- [배포 source provenance](official/SOURCES.json), [CPU 준비 결과](official/PREPARATION.md)

**FLU/CON 참조 자산은 기존 SHA 그대로 유지합니다.** 코드는 현재 server1 baseline
job 61519·61520·61521의 실행본을 기준으로 CAKE·BLUE의 연산 방식을 참고합니다.
case별 prompt batching과 KV cache를 사용하고, 한 번 생성한 원문을 FLU/CON이 함께 평가합니다.
자세한 source pin과 생성 규칙은 [공통 FLU/CON 설명](official/README.md#flucon-기존-자산-유지-현재-server1-코드-기준)에 기록합니다.

```bash
python3 -m official.tools.verify
python3 -m official.experiments.prepare matrix --output local/official-baselines/configuration
```

## Main results

모델별로 CF와 zsRE의 최종 성능을 기록합니다. 각 실험은 2,000건을 100건씩 순차 편집한
최종 checkpoint(W20)를 기준으로 하며, 지표 정의는 [공통 실험 계약](official/hparams/contract.json)을 따릅니다.
**이번 표는 새 cold-start 재실험 전용입니다.** 과거 완료 수치나 과거 checkpoint 재개 결과를
새 실험으로 소급 입력하지 않습니다. checkpoint가 있어도 sample 구성·순서가 다르면 다시 실행합니다.
**사용자 승인 재개 예외(2026-10-10):** Qwen zsRE AlphaEdit+SPHERE 한 chain은 원62087의
durable B9(900건)에서 B10–B20만 이어간다. B1–B9와 재개 구간의 source/config 출처를 분리하며
새 cold 결과라고 부르지 않는다. 최종 W20에서 같은 first2K 전체를 평가한다.
[정확한 parent SHA·재개 등록 이력](experiment-reports/global/qwen-zsre-sphere-b9-resume-20261010/report-ko.md).
미제출은 빈칸, 실제 제출 후 대기는 `PENDING: <실제 job ID>`, 실행 중은
`ING: <실제 job ID>`로 해당 dataset의 칸에만 표시합니다(CF 6칸 / zsRE 3칸).
qualification·W0 준비·collector·tuning job은 본실험 job으로 표시하지 않습니다.

GH가 서버별 제출·상태·완료 보고를 받아 이 표를 통합합니다. 각 dataset의 상태/수치에는
server·job ID·관측 시각·실행 commit·config SHA·ordered sample identity를 담은 보고서를 연결합니다.
새 W20 실측이 완료된 지표만 수치로 교체하며 실패·취소·결측을 0이나 완료 성능으로 쓰지 않습니다.
사용자가 FLU/CON을 후속 checkpoint 평가로 미룬 실행은 factual W20 결과와 별도로
CF Flu/Con을 `DEFERRED`로 표시하고 전체 평가 완료로 주장하지 않습니다.
Ours 행은 결과 기록용이며 새 arm/실험을 자동 승인하지 않습니다.
2026-10-10 07:37–07:38 KST 서버별 단발 관측과 CPU 검산을 반영했다.
Qwen CF BLUE/MEMIT/MEMIT-FE/SPHERE 4개, GPT-J zsRE MEMIT 및 별도 Qwen FE_HISTORY의
새 완료 수치를 추가했다. SH3/SH4는 신규 본표 적격 완료 baseline이 없다.
[최신 통합·검증 범위](experiment-reports/global/baseline-completed-zsre-audit-20261010/report-ko.md) ·
[SH1 원자료·상태](audits/servers/server1/baseline-completed-zsre-audit-20261010/table-rows.json) ·
[SH2 원자료·상태](audits/servers/server2/baseline-completed-zsre-audit-20261010/audited-final.json).
세부 [표 관리 정책](control/main-results-policy.json)과 [사용자 지시](messages/head/2026-10-09-main-table-fresh-rerun.json)를 따릅니다.

**zsRE 재평가 안내 (2026-10-09):** 기존 evaluator에서 Eff/Gen decode-retokenize 및
Llama Loc loader BOS 차이를 확인하여 저장된 W20 checkpoint의 최종2K E/G/Loc만 재평가한다.
Llama는 2026-10-10 재검산 6종, SH2는 17:18 KST 검산 **GPT-J 6종·Qwen FT/BLUE/MEMIT/AlphaEdit/MEMIT-FE 5종**으로,
총17개 완료 행에 공개-query 최종 평가 수치를 기록했다.
새 완료 GPT-J SPHERE **61947**(원 편집61735)과 수정 Qwen MEMIT **62081**은 각각2,000건이다.
MEMIT-FE **62085**와 AlphaEdit **62083**도 완료 raw 검산 수치를 반영했다. SPHERE **62087**은 B10 OOM/W20 없음으로 보존하고,
수리 cold job **62534**는 실행 전 취소했다. 사용자 승인 B9 재개 **62538**는
17:18 KST 후속 snapshot에서도 PENDING(`afterany:62532`)이다.
깨진 편집 context를 사용한 과거 MEMIT/AlphaEdit 값은 철회하고 수정 cold run 상태를 표시한다.
입력 CPU 일치가 pretrained 출력의 bitwise 논문 재현을 뜻하지는 않는다.
기존 값·raw와 CF/FLUCON 일정은 보존한다. 사용자 명시 삭제로 오류 Qwen
61956/61960/61968의 전용 checkpoint payload 3개(11,251,362,043 bytes)만 삭제했다.
로그/source/manifest는 보존하며 [삭제 영수증](audits/servers/server2/qwen-broken-checkpoint-delete-20261010/deletion.json)에 구분했다.
[공통 평가·최초 등록 기록](experiment-reports/global/zsre-2k-reeval-20261009/report-ko.md) ·
[SH1 최신 완료 검산](experiment-reports/servers/server1/baseline-completed-zsre-audit-20261010/report-ko.md) ·
[SH2 최신 완료 검산](experiment-reports/servers/server2/baseline-completed-zsre-audit-20261010/report-ko.md).
10:31 KST SH2 후속 검산은 저장 prediction/target ID의 요청별 평균 E/G/loc_ans, 실제 frozen query/source와
raw SHA를 확인했다. 원 W0agreement/token-micro 값으로 대체하지 않는다.
[새 완료2행·cap3 실제 적용](experiment-reports/global/completed-table-flucon-cap3-20261010/server2-integration.md).

**FE 전면 전환 — USER-FE-ORIGINAL-W0-RESET-20261011-R1:**
기존 native MEMIT-FE, FE_HISTORY, FE author-hparams(+history), FE sink 실험은 모두
**WITHDRAWN / SUPERSEDED**다. 아래 FE 행은 새 **FE (author repo, W0-fixed z)** 전용이며
기존 점수를 복사하지 않는다. 이전 표의 history/author/native FE 수치와 출처는
[철회된 FE 결과 이력](experiment-reports/global/fe-original-w0-reset-20261011/withdrawn-results.md)에 보존했다.
본문의 이전 FE 완료·KEEP 서술은 당시 이력이며 이 지시가 우선한다.
이전 FE checkpoint는 owner 영수증 기준 SH1 11개·54,211,593,935 bytes,
SH2 6개·23,008,049,946 bytes, 합계 **17개·77,219,643,881 bytes 삭제 완료**다.
백업/복구사본은 없으며 raw/log/config/source는 보존했다. SH3/4의 제한된 조사에서는 복제본을 찾지 못했으며
전체 filesystem에 잔여가 없다는 뜻은 아니다.
HF 원본·C0·데이터·비FE checkpoint 및 raw/log/source는 삭제 대상이 아니다.

새 실행은 저자 commit `478134df`의 native FE-MEMIT, YAML/default BF16,
편집 전 W0 전체2K z 선계산/고정과 native history/solve를 사용한다.
매 batch current100의 R/P/N, 500/1000/1500/2000 누적 all-seen을 평가하고
각 milestone마다 실행별 latest checkpoint 하나를 atomic overwrite한다. 평가 raw는 누적 보존한다.
CF factual / zsRE public-query requestmacro loc_ans 비교 evaluator는 편집정책과 분리한다.
SH1: Llama CF→zsRE, GPT-J CF→zsRE의 두 lane. SH2: Qwen CF/zsRE 두 lane.
각 서버 cap2이며 비FE RUNNING과 원자료를 보존한다. SH1은 네 job을 held 검사 후 release했다.
2026-10-11 요청자 readback 기준 Llama CF **63151 RUNNING**, zsRE **63152 PENDING**,
GPT-J CF **63153 PENDING**, zsRE **63154 PENDING**이다. 이는 완료 성능이 아니다.
SH2 Qwen CF/zsRE는 공통 source/CPU 결속을 마쳤지만 **미제출 — 저장공간 부족**이다.
admission 가용59,029,065,728B / 필요60,363,309,056B로 1,334,243,328B 부족하며,
reserve 축소·추가 삭제·자동 재시도 없이 보존한다. 신규 job ID가 없으므로 scheduler PENDING으로 표시하지 않는다.
미제출 Qwen FE 셀은 공란, 등록된 CF의 Flu/Con은 DEFERRED로 유지한다.
[실제 등록·삭제·저장공간 영수증 통합](experiment-reports/global/fe-original-w0-reset-20261011/report-ko.md).
[정본 범위·담당 수락](messages/head/2026-10-11-fe-original-w0-reset.json).

**W0 행은 편집 전 base model을 동일한 2,000개 요청에 평가한 기준값**이며 W20 결과와 구분합니다.
2026-10-09 각 서버의 저장 원자료를 직접 확인하고 재집계했습니다.
W0의 CF Score/Eff/Gen/Loc 및 zsRE 지표는 %다. 모든 표의 CF Flu/Con은 논문 표시 배율인 **반올림 전 원 평균값 ×100**을 적용한다.
FLU는 entropy 가중 평균으로 100을 넘을 수 있고, CON은 TF-IDF cosine ×100이며 둘 다 정확도 %가 아니다.
소수 둘째 자리 half-up은 변환 후 한 번만 적용한다. 원 JSON/W&B raw 키는 bits/cosine 단위를 유지한다.
[표시 교정·원자료 검산 및 평가 등록 진행](experiment-reports/global/flucon-paper-scale-20261010/report-ko.md).
Llama3 W0 및 별도 SH2 provenance로 결속한 Qwen W0 Flu/Con을 기록하고, 미측정 생성 지표는 `DEFERRED`입니다.
**W0 zsRE는 기존 exact-token-prefix evaluator 관측**입니다. Loc은 `loc_ans` 정답 정확도이며 W0 자기 일치율 100%가 아닙니다.
Llama3·GPT-J의 원본 코드와 tokenizer 처리 차이가 알려져 있어, 이 값은 원본 호환 재평가 완료를 뜻하지 않습니다.
Qwen CF는 server3 job **61813**의 편집 전 W0, zsRE는 server2 job **61900**의 완료된 W0 관측을 사용합니다.
Qwen **CF 생성 두 셀만** SH2 producer **61898**의 cold W0 관측이다. 모델·tokenizer SHA와 동일 first2K를 대조했으며,
SH3 factual 셀은 그대로 유지한다. SH3의 generation 실측 또는 cross-hardware bitwise 동등성을 뜻하지 않는다.
[분리 출처·generation protocol·참조 SHA 결속](experiment-reports/servers/server2/flucon-paper-scale-20261010/report-ko.md).
[모델별 출처·단위·원자료 SHA·검산 범위](experiment-reports/global/w0-main-table-20261009.md).

**비FE 완료 결과 유지:** 2026-10-11 검산한 GPT-J FT/MEMIT/SPHERE generation 및
기존 non-FE CF/zsRE 결과는 유지한다. 이전 FE 결과는 위 reset에 따라 철회했으며
PRICE/W0 행은 변경하지 않았다. 이전 검산의 원자료는
[철회 전 통합 보고](experiment-reports/global/author-main-refresh-20261010/report-ko.md)에 남긴다.

### Llama3-8B-Instruct

**2026-10-10 SH1 검산: CF FT/SPHERE/MEMIT-FE 및 공개-query zsRE 6종 완료.**
CF FT **61771**, SPHERE **61770** 및 zsRE FT/MEMIT/AlphaEdit/BLUE/MEMIT-FE/SPHERE
**61716/61717/61718/61719/61720/61721**의 실제 2,000건·20 commit·순서·raw NLL·분모를 검산했다.
이는 원 편집 완료 기록이다. 아래 zsRE 표는 새 공개-query 최종 평가 `61932–61937`의 상태로 대체했다.
CF source `34e4d52d`, 원 zsRE 편집 source `94304dc9`; CF FT/SPHERE/MEMIT-FE Flu/Con은
저장 checkpoint 평가 **62259/62260/62261**의 완료 raw(각2,000 case/20,000 prompts)를 반영했다.
원 평균 bits/cosine에 ×100 후 half-up2 표시만 적용했으며 factual·zsRE 수치는 변경하지 않았다.
[생성 원자료·완료 collector 검산 근거](experiment-reports/servers/server1/completed-table-flucon-cap3-20261010/report-ko.md).
CF MEMIT-FE **61773**는 W20/2,000건 검산 완료다. zsRE 재평가 **61932/61933/61935/61937**도
저장 predicted/target ID에서 요청별 평균을 독립 재집계했다(E/G 각6,035, Loc12,465 token).
**61934/61936**도 공개-query W20 평가가 완료되어 아래 E/G/Loc을 갱신했다.
[최신 결과와 원자료 결속](experiment-reports/servers/server1/flucon-paper-scale-20261010/report-ko.md).
AlphaEdit **61769**·MEMIT **61772**는 사용자 지시로 CANCELLED; 아래 CF 두 수치는
그 신규 job 결과가 아닌 과거 **42657/42658 B020** 결과다(‡).
zsRE Loc은 **loc_ans 정답 token 정확도의 요청별 평균**이다. 이전 W0 prediction agreement는
본표에서 제외하고 별도 보조지표로 보존했다.
[Loc 정정·원 token ID 재집계](experiment-reports/servers/server1/official-baselines-20261008/zsre-loc-recalculate-20261009/report-ko.md).
[W20 검산 보고](experiment-reports/servers/server1/official-baselines-20261008/results-review-20261009/report-ko.md) ·
[job/source/config/raw SHA·정확한 수치](audits/servers/server1/official-baselines-20261008/results-review-20261009/results.json).
별도 GPU qualification은 `NOT_RUN_USER_DISABLED`로 유지하며, CPU 결과 검산을 GPU 검증이나 온라인 전송 검증으로 표시하지 않는다.

| Method | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 | zsRE Eff | zsRE Gen | zsRE Loc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [W0 (편집 전)](experiment-reports/global/w0-main-table-20261009.md) | 13.36 | 8.20 | 10.95 | 88.56 | 635.22 | 24.64 | 38.10 | 37.61 | 38.59 |
| FT | 56.59 | 89.25 | 73.30 | 35.50 | 449.50 | 2.90 | 14.63 | 11.82 | 25.25 |
| MEMIT | 58.88‡ | 64.75‡ | 61.70‡ | 51.83‡ | PENDING: s2-flucon-llama3-memit-42658 (62874) | PENDING: s2-flucon-llama3-memit-42658 (62874) | 44.30 | 39.95 | 22.30 |
| AlphaEdit | 84.80‡ | 99.30‡ | 93.23‡ | 68.59‡ | PENDING: s2-flucon-llama3-alphaedit-42657 (62875) | PENDING: s2-flucon-llama3-alphaedit-42657 (62875) | 95.18 | 91.40 | 31.10 |
| AlphaEdit-BLUE | 89.84† | 99.60† | 97.15† | 76.59† | PENDING: s2-flucon-llama3-alphaedit_blue-39283_1 (62876) | PENDING: s2-flucon-llama3-alphaedit_blue-39283_1 (62876) | 95.87 | 92.28 | 32.83 |
| FE (author repo, W0-fixed z) | ING: official-s1-llama3-cf-fe-original (63151) | ING: official-s1-llama3-cf-fe-original (63151) | ING: official-s1-llama3-cf-fe-original (63151) | ING: official-s1-llama3-cf-fe-original (63151) | DEFERRED | DEFERRED | PENDING: official-s1-llama3-zsre-fe-original (63152) | PENDING: official-s1-llama3-zsre-fe-original (63152) | PENDING: official-s1-llama3-zsre-fe-original (63152) |
| AlphaEdit+SPHERE | 86.87 | 99.50 | 94.95 | 71.68 | 619.20 | 33.64 | 95.13 | 91.36 | 31.38 |
| PRICE (Ours) | 90.98§ | 99.90§ | 93.50§ | 81.50§ | ING: pf2k-flucon-llama3-P-beta100 (63125) | ING: pf2k-flucon-llama3-P-beta100 (63125) | 99.62§ | 94.83§ | 45.31§ |

§ 사용자 2026-10-10 결정에 따른 **최종 PRICE method** CF 2K이며, server4 job **62604**로 실행했다.
- 설정: arm `llama3-P-beta100`. β = c = β_max = 1.0, unit-lr ρ0.05, γ1, cap 끝점 cast, HC-PRICE, 요청별 early exit.
- 데이터: eval-2K 2,000건, B1–B20, baseline과 같은 순서와 context.
- arm과 runner는 server4 task branch commit `885a0e26`에 있고 main에는 아직 병합하지 않았다. resolved config sha256 `1208f9ce…`.
- HC 가격은 정규화하지 않았다. 가장 싼 층 가격이 B20까지 1.034 이내라 잡음 수준이라는 사용자 판단(2026-10-11)이다.
- Eff/Gen/Loc은 strict NLL preference R/P/N 성공률 1,998/2,000 · 3,740/4,000 · 16,300/20,000이다.
- Score는 반올림 전 성공률의 조화평균(90.9776)이다. 표시는 decimal half-up 둘째 자리다.
- 이전 FREE100 job 60103 값(99.70 / 92.78 / 82.21, Score 90.98)을 대체했다.
- CF Flu/Con은 W20 checkpoint를 baseline과 같은 생성 평가기(server1 = devbox, job **63125**)로 평가하는 중이다. zsRE는 같은 설정의 job **62889**(official zsRE stream·evaluator, W20/2,000 edits)이다. W0가 본표 zsRE W0 행(38.10 / 37.61 / 38.59)과 같다. Loc(45.31)이 W0(38.59)보다 높은데, W0 예측 일치율은 62.68%다(원인 미확인). [zsRE 보고](experiment-reports/servers/server4/price-final-2k-20261010/llama3-zsre-report-ko.md)
- 500 edit마다 resumable checkpoint를 저장했다(가중치, history H, HC 통계).
[실행·수치·checkpoint 보고](experiment-reports/servers/server4/price-final-2k-20261010/llama3-cf-report-ko.md) ·
[정확한 수치·SHA](audits/servers/server4/price-final-2k-20261010/llama3-cf-results.json) ·
[이전 FREE100 출처](experiment-reports/servers/server1/official-baselines-20261008/llama-free100-table-20261009.md).

† 사용자 2026-10-09 지시에 따라 표본·순서를 대조한 기존 Llama BLUE job **39283_1**의
**B20/2,000 edits** 결과를 반영했다(새 official 재실행 결과 아님).
동일 fixed10k 파일의 첫2,000 순서가 현재 official lock과 일치한다.
Eff/Gen/Loc은 기존 strict NLL preference 집계, Score는 기존 R/P/N 성공률의 조화평균이다.
Loc은 N success 15,317/20,000 = 76.585%를 소수 둘째 자리로 반올림했다.
Flu·Con은 이 historical B020 checkpoint를 사용하는 별도 평가62876을 제출해 PENDING으로 표시한다.
native source/runtime·평가기 차이와 분모 및 대조 근거는
[BLUE 2K 확인 기록](experiment-reports/servers/server1/official-baselines-20261008/llama-blue-2k-table-check.md)에 구분했다.

‡ 사용자 `USER-SIDE-GH-LLAMA-CF-MEMIT-ALPHA-2K-TABLE-20261009-R1`의 명시 예외로
기존 native MEMIT **42658**, native AlphaEdit **42657**의 **B020/2,000 edits**를 반영했다.
새 official rerun 완료 또는 최종 10K 결과가 아니다. R/P/N 분모는 2,000/4,000/20,000,
Score는 반올림 전 성공률의 조화평균이며 표시는 소수 둘째 자리 decimal half-up이다.
Flu/Con은 이 historical B020 checkpoint의 별도 평가62874/62875를 제출해 PENDING이며,
zsRE는 별도 새 W20 결과다. 현재 evaluator로 평가한다고 과거 편집 source/runtime을 재명명하지 않는다.
동일 표본·순서는 runtime·hparams·평가기 완전동등을 뜻하지 않는다.
[원 raw SHA·분자/분모·사용자 예외 및 취소 처리 근거](experiment-reports/global/llama-native-cf-historical-b020-20261009.md).

### Qwen2.5-7B-Instruct

**2026-10-10 context-mask 수정 재편집:** 편집용 cached context mask 오류가 확인되어
MEMIT/AlphaEdit/MEMIT-FE/SPHERE의 CF·zsRE 8개 chain은 **cold rerun 대상**이다.
기존 zsRE61956/61960 점수는 당시 잘못된 context로 편집한 가중치의 측정값으로 원 보고서에 보존하고,
수정된 baseline 성능으로 제시하지 않는다. 평가-only로 편집 입력 오류를 복구할 수 없다.
SH2 상태 셀은 **2026-10-10 17:18:09 KST** owner 단발 snapshot이다.
BLUE61962와 수정 MEMIT62073/MEMIT-FE62077/SPHERE62079의 CF W20/2K 수치를 반영했다.
CF AlphaEdit62075 및 zsRE62081/62083/62085는 W20/2K 검산 완료다.
zsRE MEMIT-FE62085의 Eff/Gen 0.00, Loc0.03은 실제 저장 token correctness의 요청별 평균이며
결측을 채운 0이 아니다(2,000 requests, E/G/Loc token 분모6,691/6,691/11,476).
FLU/CON 후속평가는 실제62870–62873으로 등록·release했으며 아래 생성 셀만 상태를 표시한다.
[새 완료 수치·raw/source/config 및 평가 준비](experiment-reports/servers/server2/baseline-refresh-s2-flucon-20261010/report-ko.md).
62087은 B10 `eigh` CUDA OOM으로 FAILED/W20 없음이며 B9 checkpoint를 보존한다.
최신 사용자 지시로 cold **62534**는 PENDING/실행시간0에서 취소하고, **62538**
`s2-qwen25-zsre-sphere-resume-b9`를 held 검사 후 release했다(08:51 KST PENDING,
`afterany:62532`). 원62087의 B1–B9(900건) 출처는 보존하고 수리 source78017702에서
B10–B20(1,100건)만 이어간다. W/H/context/RNG/cursor를 복원하며 새 W0/첫900건 재편집은 없다.
이 한 건만 fresh-only의 사용자 승인 예외다. 원 hparams/FP32 GPU eigh 및 buffer 수명 수리 유지,
CPU17 PASS이며 실제 GPU 복원/OOM 해결/W20은 미관측이다.
[parent SHA·취소·재개 등록 근거](experiment-reports/global/qwen-zsre-sphere-b9-resume-20261010/report-ko.md).
[최신 원자료·query·상태 검산](experiment-reports/servers/server2/baseline-completed-zsre-audit-20261010/report-ko.md).
context 판정은 [실제 context 감사](experiment-reports/servers/server2/qwen-context-audit-20261010/report-ko.md)에 구분했다.
CF 네 rerun의 원 generation 일정은 DEFERRED로 보존하며, 별도 W20 평가62870–62873을 제출했다.
원 최종 checkpoint는 평가 consumer가 끝날 때까지 보존한다.
FT/BLUE는 이번 mask 오류 재편집 대상이 아니다. FT CF61898, FT zsRE61900의 평가-only62072,
BLUE zsRE61964는 완료 원자료 검산 수치를 아래 표에 반영했다.
표의 CF Flu는 raw entropy(bits) ×100, Con은 raw TF-IDF cosine ×100이며 둘 다 정답률이 아니다.
이 기존 FT에서 실제 측정한 generation 점수를 DEFERRED 상태인 새 rerun에 복사하지 않는다.
[사용자 실행 계약](messages/head/2026-10-10-qwen-baseline-mask-cold-rerun.md) ·
[수리·재제출 진행](experiment-reports/global/qwen-baseline-mask-cold-rerun-20261010/report-ko.md).
아래 10월9일 상태·등록·dependency는 현재 상태가 아닌 원 관측 이력이다; 61956/61960 숫자는 이번 supersession 이전 기록이다.

**Server2 원 결과 이력: 2026-10-09 23:05:48 KST SH2 단발 snapshot/CPU 검산 기준.**
zsRE **MEMIT 61956·AlphaEdit 61960**의 공개-query W20/2,000건 평가는 완료됐으나,
편집 context 오류가 확인되어 당시 반영한 6칸의 수치는 철회하고 재편집 상태로 대체했다.
저장 predicted/target ID를 독립 재집계했고 요청별 token accuracy의 평균이며,
분모는 Eff/Gen/Loc **6,691/6,691/11,476 tokens, 각 2,000 requests**다.
[최신 완료·상태 검산](experiment-reports/servers/server2/main-table-refresh-20261009/report-ko.md) ·
[raw/source/config/query SHA 및 정확한 수치](audits/servers/server2/main-table-refresh-20261009/table-rows.json).
FT CF **61898**은 RUNNING. zsRE **61900**은 편집 W20 완료지만 구 token-prefix source이므로
**공개-query 재평가 필요(미등록)**이며 새 점수로 반영하지 않았다. 원 source `69bfbb2c`는 보존한다.
미시작 **61899 및 61901–61922(23개)**만 취소하고, 나머지 5방법 × CF/zsRE의
**10개 GPU 본실험**을 새 source `5503935821b0ececb4aef09a5bccb5308879a6b5`로
held 검사 후 release했다. 현재 CF BLUE **61962**, zsRE BLUE **61964**·FE **61968**은 RUNNING,
나머지 미완료 작업은 아래 PENDING 상태다. 초기 등록 PENDING과 현재 완료 상태를 구분한다.
GPU0 archive/KEEP 단계는 **61952/61953/61955/61957/61959/61961/61963/61965/61967/61969/61971/61973**,
collector는 **61974**다. cap4 안에서 보호 RUNNING **61951**도 자원 dependency에 포함했고 변경하지 않았다.

새 zsRE caller는 공통 공개-query 평가와 loc_ans request-macro를 사용한다.
실제 Qwen tokenizer의 전체 2,000 요청·24,858 query CPU 대조에서 입력/target mismatch0;
이는 모델 forward 점수나 GPU qualification PASS가 아니다.
원 FT 및 역사 W0 provenance는 보존하며 새 평가 경로의 관측으로 소급 표시하지 않는다.
CF native fit/hparams와 **W0_AND_W20_FIRST2000** 생성 일정, zsRE 생성 없음,
qualification **NOT_RUN_USER_DISABLED**, checkpoint 보존은 유지한다.
이번 CPU 검산에서는 W&B remote 전송 성공을 별도 조회·확정하지 않았다. archive는 consumer/receiver 조건 충족 전
**KEEP**이며 실제 전송·삭제는 0이다.

GPT-J 최종 checkpoint 재평가 **61942–61948**은 같은 ID/source로 유지·release했다.
자원 edge만 **61942→61970, 61943→61972, 61944→61966, 61945→61968**
(왼쪽 job이 오른쪽 종료를 기다림)로 재연결했다.
기존 취소된 job ID를 현재 실행 대상으로 표시하지 않는다.
[교체 등록 당시 SH2 보고](experiment-reports/servers/server2/qwen-pending-eval-refresh-20261009/report-ko.md) ·
[실제 job name/config/dependency·snapshot](audits/servers/server2/qwen-pending-eval-refresh-20261009/submission.json).

이전 Server4 CF FT **61783**의 디스크 부족 실패 및 후속 23개 취소,
Server2 최초 이전 등록과 source/raw/checkpoint 이력은
[Server4 이전 보고](experiment-reports/servers/server4/qwen-migration-20261009/report-ko.md)와
[Server2 원 등록 보고](experiment-reports/servers/server2/qwen-migration-results-20261009/report-ko.md)에 보존한다.
PRICE/OURS/tuning 및 무관 job은 이번 교체 범위가 아니다.

별도 SH3 **61813 Q3-beta250** cold 최종 W20은 E98.05/G89.95/S68.985/Score83.770638%로
[CPU 검산 완료](experiment-reports/servers/server3/official-baselines-20261008/qwen-61813-review/report-ko.md).
이는 Q3 선택 설정의 final 평가이며 튜닝 실행 자체와 구분한다. 기본 PRICE 본표 승격 승인은 없어
아래 PRICE 행에 자동 합치지 않았으며 Flu/Con은 DEFERRED다.
SH3 최신 CPU 재검산에서도 위 네 지표 delta0, 신규 본표 적격 완료0이다.
별도 Llama **61821 llama-price-L1-2k**는 RUNNING/W20 없음이며 기존 PRICE 예외 값을 대체하지 않는다.
[SH3 완료·제외 inventory](experiment-reports/servers/server3/main-table-refresh-20261009/report-ko.md).

| Method | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 | zsRE Eff | zsRE Gen | zsRE Loc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [W0 (편집 전)](experiment-reports/global/w0-main-table-20261009.md) | 20.88 | 13.95 | 16.60 | 85.14 | 625.21 | 25.91 | 36.42 | 35.26 | 38.40 |
| FT | 57.45 | 85.50 | 67.50 | 38.90 | 471.02 | 3.01 | 23.25 | 18.54 | 2.34 |
| MEMIT | 62.23 | 68.50 | 65.83 | 54.30 | PENDING: s2-flucon-qwen25-memit-62073 (62870) | PENDING: s2-flucon-qwen25-memit-62073 (62870) | 41.49 | 39.44 | 26.39 |
| AlphaEdit | 83.03 | 99.05 | 97.55 | 63.36 | ING: s2-flucon-qwen25-alphaedit-62075 (62871) | ING: s2-flucon-qwen25-alphaedit-62075 (62871) | 85.05 | 78.18 | 30.79 |
| AlphaEdit-BLUE | 77.98 | 97.70 | 96.75 | 55.86 | 602.99 | 37.43 | 58.62 | 53.66 | 5.73 |
| FE (author repo, W0-fixed z) |  |  |  |  |  |  |  |  |  |
| AlphaEdit+SPHERE | 83.73 | 99.40 | 97.70 | 64.37 | PENDING: s2-flucon-qwen25-sphere-62079 (62873) | PENDING: s2-flucon-qwen25-sphere-62079 (62873) | PENDING: s2-qwen25-zsre-sphere-resume-b9 (62538) | PENDING: s2-qwen25-zsre-sphere-resume-b9 (62538) | PENDING: s2-qwen25-zsre-sphere-resume-b9 (62538) |
| PRICE (Ours) | ING: pf2k-qwen25-cf-PN-beta150-ee (63144) | ING: pf2k-qwen25-cf-PN-beta150-ee (63144) | ING: pf2k-qwen25-cf-PN-beta150-ee (63144) | ING: pf2k-qwen25-cf-PN-beta150-ee (63144) | ING: pf2k-qwen25-cf-PN-beta150-ee (63144) | ING: pf2k-qwen25-cf-PN-beta150-ee (63144) | ING: pf2k-qwen25-zsre-PN-beta150-ee (63145) | ING: pf2k-qwen25-zsre-PN-beta150-ee (63145) | ING: pf2k-qwen25-zsre-PN-beta150-ee (63145) |

### GPT-J-6B

**2026-10-09 SH2 검산: CF 6종·zsRE 6종 모두 W20/2,000건 factual 완료.**
방법 순서 FT/MEMIT/AlphaEdit/BLUE/MEMIT-FE/SPHERE의 CF job은
**61650/61725/61778/61779/61780/61781**, zsRE는 **61726/61728/61730/61732/61734/61735**다.
이는 원 편집 완료 기록이다. 아래 zsRE 표는 새 공개-query 최종 평가 `61942–61947`의 상태로 대체했다.
2026-10-10 10:31 KST 후속 검산에서 SPHERE61947까지 **공개-query 최종 재평가6종 모두 완료**했다.
SPHERE는 E/G/Loc **99.67/96.29/28.00**, 요청2,000개·target token 분모5,557/5,557/9,694다.
[SH2 최신 공개-query 원자료·분모 검산](experiment-reports/servers/server2/completed-table-flucon-cap3-20261010/report-ko.md).
각 원 raw SHA·20 commit·ordered cohort·source/config를 결속하고 CPU 재집계를 대조했다.
CF 분모 R2,000/P4,000/N20,000, zsRE는 request-macro이며 Loc은 **loc_ans 정답 정확도**다.
이전 W0 prediction agreement는 보조지표로 분리했고, 저장 predicted/target token ID에서
6개 행을 독립 재집계했다. Eff/Gen 표시는 그대로다.
[zsRE Loc 정정 보고](experiment-reports/servers/server2/zsre-loc-recalculate-20261009/report-ko.md).
CF Score는 반올림 전 E/G/S의 조화평균이다. 소수 둘째 자리 표시는 decimal half-up이며,
CF 비율의 이진 부동소수점 꼬리는 고정 분모의 정수 분자로 정합화한다.
[완료 결과 보고](experiment-reports/servers/server2/qwen-migration-results-20261009/report-ko.md) ·
[각 job/source/config/raw SHA·정확한 수치](audits/servers/server2/qwen-migration-results-20261009/gptj-results.json).
이 검산은 GPU 추가 평가나 W&B 온라인 재검증이 아니다. 이전 W0 전송 오류·취소 이력은 보존한다.

W&B: [zsRE 전용 페이지](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsreindex) ·
[Llama3](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsrellama3) ·
[Qwen2.5](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsreqwen25) ·
[GPT-J](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsregptj).
새 `zsre/*`의 Specificity는 teacher-forced loc_ans 정확도다. 기존 frozen run의 W0-agreement
Specificity 기록은 수정하지 않으며, 별도 CPU 후처리 결과와 새 source의 지표를 구분한다.
페이지 설정 원격 검증은 actual run/GPU/성능 완료 검증과 별개다.

**원 CF 실행의 FLU/CON은 W0·W20 모두 `DEFERRED_CHECKPOINT_EVALUATION`이었다.**
generation 점수·count·progress를 0으로 채우지 않는다. factual 평가는 유지하고,
매 batch 완료 후 최신 checkpoint 1개와 최종 W20 checkpoint를 보존한다.
이후 별도 2k checkpoint 평가62864–62869를 실제 제출했으며, 그 consumer 완료 전에는
최종 checkpoint를 삭제하지 않는다. 아래 성능표는 미관측 수치를 채우지 않는다.

**2026-10-10 17:28:13 KST SH2 단발 snapshot:** CF 최종 checkpoint FLU/CON 평가13개
**62864–62876** 및 GPU0 collector **62877**을 held 검사 후 release했다.
GPT-J FT62864/MEMIT62865는 RUNNING, 나머지는 Dependency PENDING이다.
원 편집은 재실행하지 않으며 전체cap3/DAG폭3, 기존62531→62532→62538 및 SH1평가는 유지한다.
Llama historical3개는 원source/lock/CP와 새평가 consumer identity를 분리했다.
62864/62865의 W&B startup identity 확인·SDK접수는 최종지표/전체history readback 완료가 아니다.
[실제 등록·CP/source/config·dependency 근거](experiment-reports/servers/server2/baseline-refresh-s2-flucon-20261010/report-ko.md).

기존 source/config/dependency와 비용은
[CF 기록](experiment-reports/servers/server2/official-baselines-20261008/deferred-flucon-ready-20261009-r1.md) 및
[zsRE 기록](experiment-reports/servers/server2/zsre-wandb-20261009/report-ko.md)에 보존합니다.
새 실행의 qualification은 `NOT_RUN_USER_DISABLED`이며 GPU 검증 PASS를 뜻하지 않습니다.

| Method | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 | zsRE Eff | zsRE Gen | zsRE Loc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [W0 (편집 전)](experiment-reports/global/w0-main-table-20261009.md) | 24.44 | 17.00 | 19.30 | 82.48 | DEFERRED | DEFERRED | 27.83 | 27.15 | 27.59 |
| FT | 58.97 | 88.75 | 66.75 | 40.61 | 510.98 | 3.04 | 23.15 | 17.95 | 0.62 |
| MEMIT | 80.63 | 97.90 | 95.38 | 60.58 | 577.93 | 36.84 | 93.52 | 88.86 | 30.81 |
| AlphaEdit | 88.22 | 99.70 | 96.33 | 73.56 | PENDING: s2-flucon-gptj-alphaedit-61778 (62866) | PENDING: s2-flucon-gptj-alphaedit-61778 (62866) | 99.69 | 96.45 | 27.94 |
| AlphaEdit-BLUE | 89.23 | 99.55 | 97.53 | 75.06 | PENDING: s2-flucon-gptj-alphaedit_blue-61779 (62867) | PENDING: s2-flucon-gptj-alphaedit_blue-61779 (62867) | 99.75 | 95.71 | 28.83 |
| FE (author repo, W0-fixed z) | PENDING: official-s1-gptj-cf-fe-original (63153) | PENDING: official-s1-gptj-cf-fe-original (63153) | PENDING: official-s1-gptj-cf-fe-original (63153) | PENDING: official-s1-gptj-cf-fe-original (63153) | DEFERRED | DEFERRED | PENDING: official-s1-gptj-zsre-fe-original (63154) | PENDING: official-s1-gptj-zsre-fe-original (63154) | PENDING: official-s1-gptj-zsre-fe-original (63154) |
| AlphaEdit+SPHERE | 88.39 | 99.70 | 95.73 | 74.27 | 616.18 | 40.89 | 99.67 | 96.29 | 28.00 |
| PRICE (Ours) | 88.22¶ | 99.80¶ | 96.23¶ | 73.57¶ | PENDING: pf2k-flucon-gptj-P-beta075 (63207) | PENDING: pf2k-flucon-gptj-P-beta075 (63207) | 99.81¶ | 96.96¶ | 29.59¶ |

¶ 최종 PRICE method의 GPT-J 2K다. CF는 rent job **101707**, zsRE는 server4 job **63027**이며 둘 다 W20/2,000 edits다.
- 설정: arm `gptj-P-beta075`(resolved `d8f5ff72…`). β = c = β_max = 0.75, unit-lr ρ0.05, γ1, cap 끝점 cast, HC-PRICE, 요청별 early exit, sink EOT 규칙과 anchor guard 20.
- HC 가격은 정규화하지 않았다. 가장 싼 층 가격이 CF 1.012(B10)·zsRE 1.014(B16) 이내라 잡음 수준이라는 사용자 판단(2026-10-11)이다.
- CF R/P/N 성공률은 1,996/2,000 · 3,849/4,000 · 14,713/20,000이고, Score는 조화평균 88.2203이다. W0가 본표 W0 행과 같다.
- zsRE는 official zsRE evaluator 값이며, W0가 본표 zsRE W0 행(27.83 / 27.15 / 27.59)과 같다.
- CF Flu/Con은 W20 가중치를 baseline과 같은 생성 평가기로 devbox(server1)에서 평가한다(job **63207**).
[GPT-J 보고](experiment-reports/servers/server4/price-final-2k-20261010/gptj-report-ko.md) · [정확한 수치·SHA](audits/servers/server4/price-final-2k-20261010/gptj-results.json).

기존 연구 문서와 실험 이력은 아래에 보존합니다. 새 실험의 설정 근거는 `official/`입니다.

<details>
<summary>이전 연구 방향과 저장소 기록</summary>

**2026-09-18 최신 설계:** [Base-choice constrained L4 write](plans/global/2026-09-18-base-choice-constrained-write-v2.md).
전체 reference512의 W0 답변 선택을 제약으로 사용하고 현재 L4 편집 response를 보존하는 최소 보정을 설계한다.
[GH 실행 지시문](project/proposals/2026-09-18-base-choice-constrained-write-gh-instruction-v2.md)은 cold B100 비교 뒤 조건부 B100×10 확장을 정의한다.
설계·CPU 검증·전달문 단계이며 실제 BPCW 모델 runner 및 GPU 성능 검증은 미완료다.

**2026-09-12 연구 배경:** [Baseline 메커니즘 분석에서 출발하는 lifelong 실험 방향](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md).
기존 결과·계측 코드를 재사용해 baseline의 원인을 분석하고, 최소 개입의 결과에 따라
방법을 선택한다. Barrier/ODE는 이 진단의 실험군에 포함하지 않는다.

`ODE-Edit`는 sequential knowledge editing의 baseline 메커니즘,
edit retention과 locality를 분석하고, 그 근거에 따라 편집 방법을 개발하는 연구 저장소다.

이하 본문은 **FzCB-Edit: Fixed-z Conditional-Completion Barrier Editing**의
2026-08-31 설계와 그 이전 연구 기록이다. 당시 method pivot에 따라 output-KL/reference-fact controller를
core에서 제거하고, fixed target progress를 hard equality로 유지한 상태에서 남은 edit의
conditional completion action만 단일 barrier로 제어한다. 2026-08-30 이전 결과와 FCW
branch는 역사적 evidence로 보존하며 새 method의 성능 근거로 자동 승계하지 않는다.

## 이전 FzCB 설계 한눈에 보기

```text
stock z*/shared delta* 1회 계산
        ↓
canonical 또는 context-correct target map 구성
        ↓
MEMIT/AlphaEdit native writer basis T와 action metric H
        ↓
full-model control map C = J_Phi T
        ↓
fixed-z homotopy equality C c = b
        ↓
suffix KKT와 V_suf, spent action E
        ↓
single barrier A0 - E - V_suf >= 0
        ↓
whitened equality-null scalar rectification
        ↓
Euler predictor + nonlinear corrector + actual budget verification
        ↓
s = 1 exact target closure 뒤 atomic terminal commit
```

핵심 설계 원칙은 다음과 같다.

- 기존 editor가 산출한 direct-z는 W-write 비교에서 고정한다.
- 여러 rewrite context에 동일 absolute z를 반복하지 않고 original activation에
  shared delta를 더한 context-correct target을 사용한다.
- 단순 Euler subdivision은 negative control이며 contribution으로 보지 않는다.
- Fixed-z progress는 full-model activation equality가 담당한다.
- Barrier는 spent action과 frozen-geometry suffix action의 총 budget 하나만 사용한다.
- Output KL, locality, prior-edit prompts와 general capability는 controller가 보지 않는다.
- Closed form은 endpoint가 아니라 native low-rank writer geometry \(T,H\)로 사용한다.
- Equality KKT와 suffix KKT 뒤 whitened null-space에서 scalar rectification을 계산한다.
- Corrector의 final coefficient action까지 completion budget에 포함한다.
- Utility first-hit이 아니라 \(s=1\)의 exact target closure를 terminal event로 사용한다.
- static constrained solver가 같거나 더 좋으면 ODE claim을 폐기한다.

## Pre-reset evidence

아래 결과는 구현 자산과 historical negative evidence로만 보존한다. FzCB method를
지지하도록 설계된 scientific evidence가 아니며, 새 방법 성능으로 재해석하지 않는다.

### 확립된 기술 기반

- Llama/Qwen genuine B10에서 W64 reduced solve certificate를 검증했다.
- 동일 W64 virtual endpoint와 committed endpoint의 parameter bytes, logits와
  event identity를 검증했다.
- injected rollback과 최종 W0 restore가 exact였다.
- Native dense와 W64는 BF16 byte-exact하지 않으므로 W64를 Native의 exact
  replacement라고 부르지 않는다.

### 가장 강한 완료 결과

Warm target initialization을 사용한 no-budget `FR-A8-NEWNLL-ALLOFF`의 matched
B10에서 두 모델 모두 `Eff 10/10`, `Gen 20/20`, `Loc 80/100`을 기록했다. 이
결과는 hard H/P veto가 under-edit를 만들 수 있음을 보여주는 강한 causal reference지만,
cold fixed-E8 최종 method의 성능 증거는 아니다.

최신 완료 cold fixed-E8 R8에서 Llama Neutral은 `Eff 8/10`, `Gen 12/20`,
`Loc 89/100`이었다. Qwen Neutral trajectory는 tau=1까지 완료했지만 Soft routing의
수치 certificate failure가 post-freeze panel을 막아 paired endpoint 지표가 남지 않았다.
Common cold-coordinate 수정 후 same-seal 재실행은 당시의 다음 gate였지만,
2026-08-30 research reset 이후 scientific execution priority에서는 내려갔다.

Historical-H benefit도 당시 검증되지 않았다. 관련 B10-1 실험의 active history가
비어 있었던 한계는 pre-reset limitation으로 남기며, FzCB sequential extension을 별도로
preregister하기 전에는 해당 claim을 재개하지 않는다.

## Follow-up 읽기 순서

1. [현재 BPCW-v2 method와 GH 실행 지시문](project/proposals/2026-09-18-base-choice-constrained-write-gh-instruction-v2.md)
2. [Proposal index와 historical 문서 상태](project/proposals/README.md)
3. [2026-08-30 직전 FCW research reset proposal](project/proposals/2026-08-30-fixed-z-functional-safe-write-proposal.md)
4. [2026-08-30 F1/F2 fast falsification plan](plans/global/2026-08-30-fixed-z-fast-falsification-plan.md)
5. [이전 ODE-BF proposal](project/proposals/ODE_BF_Dynamic_Layer_Proposal.md)
6. [전체 pre-reset 실험 파이프라인](experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
7. [SH1/SH2 pre-reset 실험 종합 리뷰](experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
8. [이전 coefficient-space ODE-Alloc](project/proposals/00.ODE_Alloc_Proposal_Report.md)
9. [관련 연구와 novelty boundary](project/proposals/sections/02-related-work-and-novelty-boundary.md)
10. [운영 규칙](PROTOCOL.md)

## Repository map

- `project/proposals/`: 현재 proposal, 역사적 원문, method/related-work sections
- `project/run_scripts/ode_edit_method/`: low-rank factor, hook, transaction,
  controller/runtime와 tests
- `project/run_scripts/ode_edit_motivation/`: editor bridge, diagnostic math와
  pre-reset motivation/evaluator 자산
- `experiment-reports/global/`: raw-free 실험 결과와 causal interpretation
- `plans/global/`: preregistered experiment contract
- `scripts/`: session/resource/static gate utilities
- ignored `local/`: raw result, full log, dataset, checkpoint와 private runtime state

## Git과 실행 경계

Git은 proposal, source, lock, compact receipt와 report를 위한 control plane이다.
Model/GPU/Slurm 실행과 raw artifact는 ignored execution plane에 둔다. Active experiment
worktree의 미완성 source는 main에 섞지 않으며, 완료 checkpoint도 source/history가
정리되고 필수 gate를 통과한 뒤에만 main으로 승격한다.

현재 실험 설계의 진입점은 위 BPCW-v2와 GH 지시문이며, 2026-09-12 baseline mechanism 설계는 배경 기록이다.
2026-08-31 FzCB method pivot proposal은 이전 방법 기록으로 보존한다.
2026-08-30 FCW proposal과 pre-reset R8/R10 source/evidence는 별도 역사 계보로 보존하며,
FzCB의 hypothesis support로 자동 승계하지 않는다.

원격 서버 접속 정보, raw IP, username, port, key, token, password와 private dataset
secret은 저장소에 기록하지 않는다.

</details>

## PRICE 최종 method와 main 2K (2026-10-10 갱신)

500건 β sweep을 마친 뒤 사용자 결정으로 확정한 최종 method다. 이전 계획(c = 0.75 고정, Qwen γ1.5)은 대체됐다.

### 최종 method (세 모델 공통)

| 항목 | 값 |
| :--- | :--- |
| 편집 층 | 지정 범위 전 층: Llama·Qwen L4–L8, GPT-J L3–L8 |
| 가격 | 간섭 가격 π(batch leave-one-out) + **HC-PRICE**(writer history cache의 누적 간섭, drift·cascade 할증) |
| 예산 | 요청별 공유 weighted-L1 Σ π‖R‖/a ≤ β. **c = β**라서 층별 cap은 기본 예산에서 걸리지 않는다. 미충족 요청만 β·max π까지 확장한다 |
| step 단위 | anchor-unit lr ρ = 0.05, cap 끝점 FP32 cast 수정 |
| γ | **1** (가격을 그대로 쓴다) |
| early exit | F < τ_F(0.05)이면 요청별로 멈추고, ≥ 2τ_F이면 재개한다. 업데이트가 약 1/3로 줄어든다 |
| sink 규칙 | GPT-J·Qwen: position 0 또는 W0 sink pair에 EOT, anchor guard 20. Llama는 BOS라 불필요 |
| context·sample | 공용 패키지(Llama `cf14b`, Qwen `5c01bc1a`, GPT-J `b454d8`). baseline과 같은 eval-2K 순서·batch 경계 |
| arm | `<model>-P-beta<b>` (β = c = β_max, γ1, HC). early exit는 runner(`project/run_scripts/price_early_exit`)가 적용한다. arm과 runner는 server4 task branch commit `885a0e26`에 있고 main에는 아직 병합하지 않았다 |

- **γ1의 근거:** Qwen γ1.5는 실현된 층 배분을 바꾸지 못했다. c 0.75 cap이 요청의 약 1/3을 묶고 있었고, γ1.5는 확장 상한만 부풀렸다(β_max 최대가 base의 9배에서 27배로).
- **c = β의 근거:** 층별 cap이 기본 예산에서 걸리지 않아 knob이 β 하나로 줄고, 배분이 가격 순서를 따른다(예산이 가장 싼 L4로 더 간다). Llama β1.0 + early exit에서 c 0.75와 성능은 같은 수준이다(W5 Score 93.51 대 93.37, W0 대비 N −1.30 대 −1.32).

### β 선택 (500건 W5, 같은 cases, Score = R·P·N 조화평균)

| 모델 | 선택 | W5 R / P / N | Score | 최고 baseline W5 |
| :--- | :--- | :--- | ---: | :--- |
| Llama | β = c = 1.0 | 100 / 95.00 / 86.54 | 93.51 | MEMIT-BLUE 93.08 |
| Qwen | β = c = 1.5 | 99.8 / 96.00 / 82.76 | 92.25 | SPHERE 90.66 |
| GPT-J | β = c = 0.75 | 99.6 / 94.70 / 79.68 | 90.50 | AlphaEdit-BLUE 91.41 |

- Qwen β = c 2.0은 P 97.0 / Score 92.06이고, 2.5는 N −6.1로 Score 90.60이다.
- GPT-J는 early exit 없이 90.73이었지만, method 통일을 위해 early exit를 포함하기로 했다.

### main 2K

| 모델 | CF 2K | zsRE 2K |
| :--- | :--- | :--- |
| Llama | server4 **62604** 완료: W20 99.90 / 93.50 / 81.50, Score 90.98 (본표 반영) | server4 **62889** 완료: W20 99.62 / 94.83 / 45.31 (본표 반영) |
| Qwen | 정규화 HC(`qwen25-PN-beta150`) server4 **63144** 실행 중. 정규화 전 HC run(rent **101706**)은 ablation으로 남긴다. 비교용 β = c 2.0(server4 **62845**)은 B10 진입 때 anchor guard로 멈췄다(편집이 쌓여 생긴 sink) | 정규화 HC server4 **63145** 실행 중 |
| GPT-J | rent **101707** 완료: W20 99.80 / 96.23 / 73.57, Score 88.22 (본표 반영) | server4 **63027** 완료: W20 99.81 / 96.96 / 29.59 (본표 반영) |

- **HC 가격 정규화(2026-10-11):** HC 배수를 곱한 뒤 요청마다 가장 싼 층의 가격이 1이 되도록 다시 나눈다(method 항목 `hc_normalize`). 정규화하지 않으면 HC가 배분뿐 아니라 강도도 바꾼다. 가장 싼 층 가격 중앙값이 Qwen 2K B10에서 1.159(최대 1.386)였고, Llama 2K B20은 1.034, GPT-J 2K B10은 1.012였다. **본표는 Qwen만 정규화 HC 재실행을 쓴다.** Llama·GPT-J는 차이가 잡음 수준이라 정규화 전 HC 결과를 그대로 쓴다(사용자 판단).
- 모든 2K run은 W5·W10·W15·W20마다 all-seen 평가와 resumable checkpoint(가중치, history H, HC 통계)를 남긴다.
- zsRE는 CF와 같은 설정을 쓴다. 데이터는 official zsRE first-2K stream, 평가는 official zsRE evaluator다.
- Qwen·GPT-J zsRE는 no-BOS 모델이라 zsRE sink scan을 먼저 해야 한다.

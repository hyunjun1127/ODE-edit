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
2026-10-10 07:38 KST owner 검산 기준 **Llama 6종, GPT-J FT/MEMIT/AlphaEdit/BLUE/MEMIT-FE 5종,
Qwen FT/BLUE 2종**의 총13개 완료 행은 공개-query 최종 평가 수치다.
GPT-J SPHERE **61947**은 PENDING이다. 수정 Qwen zsRE MEMIT **62081**·MEMIT-FE **62085**는
RUNNING, AlphaEdit **62083**은 PENDING, SPHERE **62087**은 FAILED/W20 없음이다.
깨진 편집 context를 사용한 과거 MEMIT/AlphaEdit 값은 철회하고 수정 cold run 상태를 표시한다.
입력 CPU 일치가 pretrained 출력의 bitwise 논문 재현을 뜻하지는 않는다.
기존 값·raw와 CF/FLUCON 일정은 보존한다. 사용자 명시 삭제로 오류 Qwen
61956/61960/61968의 전용 checkpoint payload 3개(11,251,362,043 bytes)만 삭제했다.
로그/source/manifest는 보존하며 [삭제 영수증](audits/servers/server2/qwen-broken-checkpoint-delete-20261010/deletion.json)에 구분했다.
[공통 평가·최초 등록 기록](experiment-reports/global/zsre-2k-reeval-20261009/report-ko.md) ·
[SH1 최신 완료 검산](experiment-reports/servers/server1/baseline-completed-zsre-audit-20261010/report-ko.md) ·
[SH2 최신 완료 검산](experiment-reports/servers/server2/baseline-completed-zsre-audit-20261010/report-ko.md).

별도 사용자 승인 **MEMIT-FE + history** CF 실험은 native MEMIT-FE 행과 합치지 않는다.
SH1이 세 모델을 source `eaf78c33`으로 등록했다. Qwen 61929는 B1 FP64 history 선형계의
임시 행렬 메모리 부족으로 실패하여 메모리 수리 source `5d6dfd58`의 61975로 cold 재제출했다.
이후 61975의 context-mask 오류로 다시 cold 제출한 **62061**의 완료값을 사용한다.
오류가 있던 61975 값은 제외하며 zsRE eval-only 작업과 구분한다.

| 별도 variant | 모델 | server | 상태 / job name (ID) | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| MEMIT_FE_HISTORY | GPT-J | server1 | W20 COMPLETE: official-s1-cf-gptj-memit-fe-history (61927) | 50.18 | 50.35 | 49.90 | 50.28 | PENDING: official-s1-flucon-eval-gptj-memit_fe_history (62262) | PENDING: official-s1-flucon-eval-gptj-memit_fe_history (62262) |
| MEMIT_FE_HISTORY | Llama3 | server1 | W20 COMPLETE: official-s1-cf-llama3-memit-fe-history (61928) | 64.66 | 80.10 | 74.10 | 48.99 | PENDING: official-s1-flucon-eval-llama3-memit_fe_history (62263) | PENDING: official-s1-flucon-eval-llama3-memit_fe_history (62263) |
| MEMIT_FE_HISTORY | Qwen2.5 | server1 | W20 COMPLETE: official-s1-cf-qwen25-memit-fe-history (62061) | 49.43 | 48.45 | 48.18 | 51.83 | DEFERRED | DEFERRED |

2026-10-10 SH1 bounded 검산: 세 모델 모두 실제20 commit/2,000건과 history once·최종CP·raw를
대조한 완료값이며 native MEMIT-FE 본표 값이 아니다. 수정 Qwen62061의 생성 평가는 DEFERRED다.
GPT-J/Llama의 FLU/CON 평가 job은 위 단발 관측에서 PENDING이다.
[최신 history 및 zsRE 검산](experiment-reports/servers/server1/baseline-completed-zsre-audit-20261010/report-ko.md).
[완료 결과와 평가 등록](experiment-reports/servers/server1/flucon-paper-scale-20261010/report-ko.md).
[최신 SH1 결과](experiment-reports/servers/server1/main-table-refresh-20261009/report-ko.md) ·
[정확한 수치·source/config·raw SHA](audits/servers/server1/main-table-refresh-20261009/table-rows.json) ·
[Qwen OOM 수리 이력](experiment-reports/servers/server1/memit-fe-history-three-model-2k/oom-repair-r1.md).

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

### Llama3-8B-Instruct

**2026-10-10 SH1 검산: CF FT/SPHERE/MEMIT-FE 및 공개-query zsRE 6종 완료.**
CF FT **61771**, SPHERE **61770** 및 zsRE FT/MEMIT/AlphaEdit/BLUE/MEMIT-FE/SPHERE
**61716/61717/61718/61719/61720/61721**의 실제 2,000건·20 commit·순서·raw NLL·분모를 검산했다.
이는 원 편집 완료 기록이다. 아래 zsRE 표는 새 공개-query 최종 평가 `61932–61937`의 상태로 대체했다.
CF source `34e4d52d`, 원 zsRE 편집 source `94304dc9`; CF Flu/Con은 별도 저장 checkpoint 평가로 등록했다.
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
| FT | 56.59 | 89.25 | 73.30 | 35.50 | ING: official-s1-flucon-eval-llama3-ft (62259) | ING: official-s1-flucon-eval-llama3-ft (62259) | 14.63 | 11.82 | 25.25 |
| MEMIT | 58.88‡ | 64.75‡ | 61.70‡ | 51.83‡ | DEFERRED | DEFERRED | 44.30 | 39.95 | 22.30 |
| AlphaEdit | 84.80‡ | 99.30‡ | 93.23‡ | 68.59‡ | DEFERRED | DEFERRED | 95.18 | 91.40 | 31.10 |
| AlphaEdit-BLUE | 89.84† | 99.60† | 97.15† | 76.59† |  |  | 95.87 | 92.28 | 32.83 |
| MEMIT-FE | 64.55 | 80.45 | 73.68 | 48.85 | ING: official-s1-flucon-eval-llama3-memit_fe (62261) | ING: official-s1-flucon-eval-llama3-memit_fe (62261) | 14.81 | 13.57 | 0.56 |
| AlphaEdit+SPHERE | 86.87 | 99.50 | 94.95 | 71.68 | ING: official-s1-flucon-eval-llama3-sphere (62260) | ING: official-s1-flucon-eval-llama3-sphere (62260) | 95.13 | 91.36 | 31.38 |
| PRICE (Ours) | 90.98§ | 99.70§ | 92.78§ | 82.21§ |  |  |  |  |  |

§ 사용자 2026-10-09 지시에 따라 이전 **FREE100 / MEMIT writer**, Llama job **60103**의
**W20/2,000 edits** CF 결과를 사용한다(새 official 재실행 결과 아님).
Eff/Gen/Loc은 기존 strict NLL preference R/P/N 성공률
1,994/2,000 · 3,711/4,000 · 16,441/20,000이며, Score는 반올림 전 성공률의 조화평균이다.
소수 둘째 자리 decimal half-up으로 표시하며 미측정 Flu/Con과 zsRE는 빈칸을 유지한다.
[FREE100 출처·원자료 SHA 검산](experiment-reports/servers/server1/official-baselines-20261008/llama-free100-table-20261009.md).

† 사용자 2026-10-09 지시에 따라 표본·순서를 대조한 기존 Llama BLUE job **39283_1**의
**B20/2,000 edits** 결과를 반영했다(새 official 재실행 결과 아님).
동일 fixed10k 파일의 첫2,000 순서가 현재 official lock과 일치한다.
Eff/Gen/Loc은 기존 strict NLL preference 집계, Score는 기존 R/P/N 성공률의 조화평균이다.
Loc은 N success 15,317/20,000 = 76.585%를 소수 둘째 자리로 반올림했다.
Flu·Con은 미관측으로 기존 빈칸을 유지한다.
native source/runtime·평가기 차이와 분모 및 대조 근거는
[BLUE 2K 확인 기록](experiment-reports/servers/server1/official-baselines-20261008/llama-blue-2k-table-check.md)에 구분했다.

‡ 사용자 `USER-SIDE-GH-LLAMA-CF-MEMIT-ALPHA-2K-TABLE-20261009-R1`의 명시 예외로
기존 native MEMIT **42658**, native AlphaEdit **42657**의 **B020/2,000 edits**를 반영했다.
새 official rerun 완료 또는 최종 10K 결과가 아니다. R/P/N 분모는 2,000/4,000/20,000,
Score는 반올림 전 성공률의 조화평균이며 표시는 소수 둘째 자리 decimal half-up이다.
Flu/Con은 미평가·DEFERRED이며 zsRE는 별도 새 W20 결과다.
동일 표본·순서는 runtime·hparams·평가기 완전동등을 뜻하지 않는다.
[원 raw SHA·분자/분모·사용자 예외 및 취소 처리 근거](experiment-reports/global/llama-native-cf-historical-b020-20261009.md).

### Qwen2.5-7B-Instruct

**2026-10-10 context-mask 수정 재편집:** 편집용 cached context mask 오류가 확인되어
MEMIT/AlphaEdit/MEMIT-FE/SPHERE의 CF·zsRE 8개 chain은 **cold rerun 대상**이다.
기존 zsRE61956/61960 점수는 당시 잘못된 context로 편집한 가중치의 측정값으로 원 보고서에 보존하고,
수정된 baseline 성능으로 제시하지 않는다. 평가-only로 편집 입력 오류를 복구할 수 없다.
상태 셀은 **2026-10-10 07:38:08 KST** owner 단발 snapshot이다.
BLUE61962와 수정 MEMIT62073/MEMIT-FE62077/SPHERE62079의 CF W20/2K 수치를 반영했다.
CF AlphaEdit62075는 PENDING이다. zsRE62081/62085는 RUNNING, 62083은 PENDING,
62087은 FAILED/W20 없음이며 실패를 완료값으로 채우지 않는다.
[최신 원자료·query·상태 검산](experiment-reports/servers/server2/baseline-completed-zsre-audit-20261010/report-ko.md).
context 판정은 [실제 context 감사](experiment-reports/servers/server2/qwen-context-audit-20261010/report-ko.md)에 구분했다.
CF 네 rerun의 FLU/CON은 DEFERRED이며 최종 checkpoint를 보존한다.
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
| MEMIT | 62.23 | 68.50 | 65.83 | 54.30 | DEFERRED | DEFERRED | ING: s2-qwen25-zsre-memit-gpu (62081) | ING: s2-qwen25-zsre-memit-gpu (62081) | ING: s2-qwen25-zsre-memit-gpu (62081) |
| AlphaEdit | PENDING: s2-qwen25-cf-alphaedit-gpu (62075) | PENDING: s2-qwen25-cf-alphaedit-gpu (62075) | PENDING: s2-qwen25-cf-alphaedit-gpu (62075) | PENDING: s2-qwen25-cf-alphaedit-gpu (62075) | DEFERRED | DEFERRED | PENDING: s2-qwen25-zsre-alphaedit-gpu (62083) | PENDING: s2-qwen25-zsre-alphaedit-gpu (62083) | PENDING: s2-qwen25-zsre-alphaedit-gpu (62083) |
| AlphaEdit-BLUE | 77.98 | 97.70 | 96.75 | 55.86 | 602.99 | 37.43 | 58.62 | 53.66 | 5.73 |
| MEMIT-FE | 50.59 | 51.05 | 51.23 | 49.53 | DEFERRED | DEFERRED | ING: s2-qwen25-zsre-memit_fe-gpu (62085) | ING: s2-qwen25-zsre-memit_fe-gpu (62085) | ING: s2-qwen25-zsre-memit_fe-gpu (62085) |
| AlphaEdit+SPHERE | 83.73 | 99.40 | 97.70 | 64.37 | DEFERRED | DEFERRED | FAILED: s2-qwen25-zsre-sphere-gpu (62087) | FAILED: s2-qwen25-zsre-sphere-gpu (62087) | FAILED: s2-qwen25-zsre-sphere-gpu (62087) |
| PRICE (Ours) |  |  |  |  |  |  |  |  |  |

### GPT-J-6B

**2026-10-09 SH2 검산: CF 6종·zsRE 6종 모두 W20/2,000건 factual 완료.**
방법 순서 FT/MEMIT/AlphaEdit/BLUE/MEMIT-FE/SPHERE의 CF job은
**61650/61725/61778/61779/61780/61781**, zsRE는 **61726/61728/61730/61732/61734/61735**다.
이는 원 편집 완료 기록이다. 아래 zsRE 표는 새 공개-query 최종 평가 `61942–61947`의 상태로 대체했다.
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

**이 CF 실행의 FLU/CON은 W0·W20 모두 `DEFERRED_CHECKPOINT_EVALUATION`이다.**
generation 점수·count·progress를 0으로 채우지 않는다. factual 평가는 유지하고,
매 batch 완료 후 최신 checkpoint 1개와 최종 W20 checkpoint를 보존한다.
FLU/CON은 이후 별도 2k checkpoint 평가에서 측정하며, 그 consumer 완료 전에는
최종 checkpoint를 삭제하지 않는다. 아래 성능표는 미관측 수치를 채우지 않는다.

기존 source/config/dependency와 비용은
[CF 기록](experiment-reports/servers/server2/official-baselines-20261008/deferred-flucon-ready-20261009-r1.md) 및
[zsRE 기록](experiment-reports/servers/server2/zsre-wandb-20261009/report-ko.md)에 보존합니다.
새 실행의 qualification은 `NOT_RUN_USER_DISABLED`이며 GPU 검증 PASS를 뜻하지 않습니다.

| Method | CF Score | CF Eff | CF Gen | CF Loc | CF Flu ×100 | CF Con ×100 | zsRE Eff | zsRE Gen | zsRE Loc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [W0 (편집 전)](experiment-reports/global/w0-main-table-20261009.md) | 24.44 | 17.00 | 19.30 | 82.48 | DEFERRED | DEFERRED | 27.83 | 27.15 | 27.59 |
| FT | 58.97 | 88.75 | 66.75 | 40.61 | DEFERRED | DEFERRED | 23.15 | 17.95 | 0.62 |
| MEMIT | 80.63 | 97.90 | 95.38 | 60.58 | DEFERRED | DEFERRED | 93.52 | 88.86 | 30.81 |
| AlphaEdit | 88.22 | 99.70 | 96.33 | 73.56 | DEFERRED | DEFERRED | 99.69 | 96.45 | 27.94 |
| AlphaEdit-BLUE | 89.23 | 99.55 | 97.53 | 75.06 | DEFERRED | DEFERRED | 99.75 | 95.71 | 28.83 |
| MEMIT-FE | 73.85 | 87.15 | 85.70 | 57.22 | DEFERRED | DEFERRED | 29.20 | 27.65 | 8.45 |
| AlphaEdit+SPHERE | 88.39 | 99.70 | 95.73 | 74.27 | DEFERRED | DEFERRED | PENDING: s2-zsre-reeval-sphere (61947) | PENDING: s2-zsre-reeval-sphere (61947) | PENDING: s2-zsre-reeval-sphere (61947) |
| PRICE (Ours) |  |  |  |  |  |  |  |  |  |

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

## PRICE 3개 모델 실험 계획 (2026-10-10)

Llama3-8B-Instruct·Qwen2.5-7B-Instruct·GPT-J-6B에서 **c=0.75를 고정**하고,
모델마다 first-1K에서 **β ∈ {0.75, 1.0, 1.5, 2.0}**을 sweep한 뒤 같은 규칙으로
β를 선택하여 main 2K를 진행하는 계획이다. **γ는 Qwen만 1.5**, Llama·GPT-J는 1.0이며,
최신 사용자 결정에 따라 **HC-PRICE도 main 실험에 포함**한다. c 변경은 ablation으로 분리한다.
아래 기존 run·수치는 계획 작성 시점의 기록이며, 자원 배치와 예상 시간은 확정된 실행 상태가 아니다.

### 고정 설정

| 항목 | 값 |
| :--- | :--- |
| 편집 층 | 지정 범위 전 층: Llama·Qwen L4–L8, GPT-J L3–L8 |
| context | Llama `cf14b`, GPT-J `b454d8`, Qwen `5c01bc1a` — 공용 패키지, baseline과 동일 |
| sample | sweep: 본실험 첫 1,000건(B1–B10); main: 첫 2,000건. baseline과 같은 순서·batch 경계 |
| sink 규칙 | GPT-J·Qwen 적용: position 0 또는 W0 sink pair에 EOT, anchor guard 20. Llama는 BOS로 불필요 |
| step 단위 | 모든 모델에 unit-lr `ρ=0.05` 적용(`lr=ρ × 첫 층 anchor 중앙값`), cap 끝점 cast/반올림 수정 포함 |
| **c (층별 cap)** | **0.75 고정**. Llama·GPT-J native MEMIT clamp와 같은 값이며, 변경은 ablation에서만 수행 |
| β_max_scale | 선택한 β와 같은 값 |
| **γ** | **Qwen 1.5**, Llama·GPT-J 1.0 |
| **HC-PRICE** | **세 모델 main 실험에 포함**. HC on/off 비교는 별도 ablation으로 유지 |

Qwen의 γ=1.5는 가격에 따른 배분과 예산 충족 사이의 mismatch를 검증하기 위한 설정이다.
계획의 근거가 된 관측에서는 unit-lr로 예산이 첫 step부터 묶였지만, Qwen의 층별 가격 중앙값이
L4 1.13, L5 1.22, L6 1.52, L7 1.33, L8 1.42로 최대/최소 약 1.35배였고 Llama는 약 1.7배였다.
실제 지출은 L5가 L4보다 많았으며(0.39 대 0.31), L7·L8에도 약 25%가 배분됐다.
γ=1.5로 가격 대비를 약 1.57배로 높여 간섭이 작은 층에 먼저 배분되는지 확인한다.

구현은 `official/`을 수정하지 않고 β·c를 공통 resolver의 override로 설정한다.
γ는 기존 task-local `price_gamma_probe` 패키지를 unit-lr보다 먼저 설치하여 적용한다.

```python
resolve(model, base_arm, override={
    "beta_base": beta,
    "c": 0.75,
    "beta_max_scale": beta,
})
```

### β 선택 규칙

세 모델에 동일한 사전 선택 규칙을 적용한다.

1. first-1K의 W10 all-seen Score(R·P·N 조화평균)가 가장 높은 β를 선택한다.
2. Score 차이가 0.2 이내이면 W0 대비 N 손실이 작은 β를 선택한다.

### 모델별 sweep과 main

| 모델 | first-1K sweep: 모델마다 4 run | 기존 관련 run / 준비 사항 | main 2K 계획 |
| :--- | :--- | :--- | :--- |
| Qwen | β 0.75/1.0/1.5/2.0, c 0.75, γ 1.5 | `100342`: β0.75/c0.75/γ1의 γ ablation 점. `100424`: β2.5/lr0.1 control의 unit-lr off ablation. 계획 작성 시 두 run 진행 중 | 선택 β로 CF 2K, 같은 설정으로 zsRE 2K |
| GPT-J | β 0.75/1.0/1.5/2.0, c 0.75, γ 1.0 | 기존 held-out γ1·γ1.5 W5 관측. sweep 전에 `b454d8` context로 first-1K 패키지와 sink scan을 새로 준비 | 선택 β로 CF 2K와 zsRE 2K |
| Llama | β 0.75/1.0/1.5/2.0, c 0.75, γ 1.0 | β1/c1 unit-lr ± HC의 W10 Score 92.7–92.9는 c ablation 참고점 | 선택 β로 CF 2K와 zsRE 2K. 새 결과 완료·검산 후 기존 Llama PRICE 행(`60103`) 교체 |

zsRE는 CF에서 선택한 설정을 그대로 사용하고 별도 튜닝을 하지 않는 안이며,
적용 여부는 아래 결정 대기 항목에 둔다.

### Ablation

선택된 β를 기준으로 first-1K에서 비교하며, Qwen HC 비교는 아래의 2K 예외를 둔다.

1. **unit-lr off:** 절대 lr과의 C mismatch를 확인한다. Qwen control에서 일부 관측을 확보했다.
2. **γ:** Qwen γ1 대 γ1.5를 비교한다. 기존 β0.75·γ1 run을 참고점으로 둔다.
3. **c:** 0.5, 1.0, β를 비교한다. main의 c는 0.75로 고정한다.
4. **FLAT 가격(π=1) 대 PRICE:** 예산이 묶인 상태에서 가격의 기여를 검증한다.
5. **HC on/off:** main에는 HC-PRICE를 포함하고, Llama의 기존 비교와 Qwen 2K 추가 대조 1회를 ablation으로 둔다.
6. **context 깨짐 대 정상, sink 규칙 on/off:** 기존 run에서 확보한 관측을 활용한다.

### 자원과 예상 일정

| 서버 | 담당 계획 | run당 예상 시간 | sweep 4개 예상 시간 |
| :--- | :--- | :--- | :--- |
| server3 H200 ×2 | Qwen sweep | 약 3.5–4 h | 2 round, 약 8 h |
| server4 ×2 | Llama sweep | 약 2.5 h | 2 round, 약 5 h |
| rent A100 ×2 | GPT-J sweep: 패키지·sink scan 준비 후 | 약 4.5–5 h | 2 round, 약 10 h |

계획 작성 시 rent의 Qwen `100342`와 control `100424` 종료 예상은 각각
2026-10-10 08:30, 10:00 KST였다. 실제 종료·자원 확보 여부는 제출 시 확인한다.
두 run을 끝까지 유지하면 GPT-J sweep은 그 뒤에 rent에서 시작하는 안이다.
후속 main CF 2K는 Qwen 약 7 h, Llama 약 5 h, GPT-J 약 10 h로 예상하며,
zsRE 2K와 최종 checkpoint의 FLU/CON 평가 시간이 추가로 필요하다.

### 결정 대기 항목

| 항목 | 계획안 / 결정이 필요한 내용 |
| :--- | :--- |
| rent의 Qwen 두 run | `100342`와 `100424`를 ablation 자료 확보를 위해 끝까지 유지할지 결정 |
| Llama sweep 위치 | server4의 GPU 2장 범위에서 진행할지 결정 |
| zsRE 설정 | CF에서 선택한 설정을 그대로 사용하고 zsRE 전용 튜닝을 생략할지 결정 |

이 항목들이 결정되면 Qwen(server3)·Llama(server4) sweep 패키지를 먼저 준비하고,
그동안 GPT-J first-1K 패키지와 sink scan을 준비하는 순서다.

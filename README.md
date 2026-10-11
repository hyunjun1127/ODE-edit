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
수리 cold job **62534**는 실행 전 취소했다. B9 재개 **62538**는 B10/1,000건을 저장한 뒤
디스크 여유 32 GiB 하한으로 중단됐다(OOM 아님). server4 **63303**으로 B11–B20 재개를 등록했으나,
2026-10-11 사용자 요청으로 실행 전 취소했다. 기존 checkpoint는 보존한다.
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

**FE 현재 기준 — 2026-10-11 사용자 정정:** 기존 수정 EasyEdit FE 구현에 **원저자 hparams + 누적 history**를 적용한다.
z는 **각 배치 진입 모델에서 그 배치 100건만** 계산한다. 미래 편집 요청의 z를 W0에서 전체2K 선계산하는 저자 repo 경로는
이 sequential baseline의 공정한 비교 기준에 포함하지 않는다. 아래 FE 행은 이 기준의 결과만 사용한다.

이전 W0-fixed 전환과 완료 job(Llama63151/63152, GPT-J63153/63154, Qwen63217/63218)은 비교용 이력으로 남긴다.
그 점수를 현재 FE 행에 복사하지 않는다. 원 수치·raw SHA는 [W0-fixed 집계](audits/servers/server1/fe-original-w0-2k-20261011/table-rows.json),
과거 batch-entry 결과는 [기존 결과 이력](experiment-reports/global/fe-original-w0-reset-20261011/withdrawn-results.md)에 보존한다.
과거 checkpoint 삭제 내역도 [당시 보고](experiment-reports/global/fe-original-w0-reset-20261011/report-ko.md)의 역사적 기록이며 새 삭제 지시가 아니다.

**FE 붕괴 원인 분리와 2K 검증 완료 (2026-10-11):** server1에서 공식 FE `main.py`로
동일 CF 2,000건을 실행했다. 저자 hparams·BF16·context·writer·누적 history와 현재 모델 z를 유지했다.

- **GPT-J:** L3 공분산 캐시의 작은 음의 고유값이 역산을 폭증시켰다. 현재 모델 기준 z에서
  C0만 수치 보정하면 공통 Score **49.75 → 85.39**로 회복했다. 원본 C0는 보존했다.
- **Qwen:** 현재 편집 모델에서 반복 계산한 target이 불안정해져 공통 Score **50.18**로 붕괴했다.

편집·공통 평가(Qwen 63385/63387, GPT-J 63386/63388·63389/63390)는 모두 COMPLETED이며, 각 checkpoint 지표를 raw에서 독립 재집계했다.
history 누적과 최종 가중치 저장도 확인했다. clamp/step/lr와 sample 순서는 바꾸지 않았다.
**본표 FE 행 반영(2026-10-11 사용자 지시, 현재 모델 z):** GPT-J는 C0만 수치 보정한 **63389 → 63390** W20 (R/P/N 1,970/2,000 · 3,792/4,000 · 13,857/20,000 → **85.39 / 98.50 / 94.80 / 69.29**), Qwen은 **63385 → 63387** W20 (1,018/2,000 · 1,998/4,000 · 9,943/20,000 → **50.18 / 50.90 / 49.95 / 49.72**, 붕괴)이다. 두 행은 공식 FE `main.py`(BF16)로, Llama 행(63382)은 수정 EasyEdit FE(FP32)로 돌렸다. 설정(author hparams, 누적 history, 현재 모델 z)은 같고 구현이 다르다. 이 시점에는 두 모델의 FE Flu/Con과 zsRE를 실행하지 않았다(이후 제출은 아래 기록). [raw 재집계](audits/global/fe-collapse-repair-20261011/table-rows-current-z.json).
Native loader는 전체 JSON을 읽고 target fitting/use만 현재100건으로 제한한다.
[전체 비교표·원인·검증 결과](experiment-reports/global/fe-collapse-repair-20261011/report-ko.md) ·
[작동하는 native 구현](official/diagnostics/fe_native/README.md) ·
[결과·checkpoint SHA](audits/global/fe-collapse-repair-20261011/native-comparison.json).

**Qwen FE 데이터 변경 진단(2026-10-11 사용자 지시):** 기존 CF fixed-10K 파일의 **2,001–4,000번째 2,000건**으로
공식 FE 현재 모델 z 실행을 추가했다. 기존 63385와 case 중복 0건이며, 코드·BF16·author hparams·seed 0·history는 동일하다.
사용자 지시로 server4 **63563 / s4-qwen25-cf-fe-next2k-train → 63564 / s4-qwen25-cf-fe-next2k-evaluate**로 옮겼다.
**PENDING(편집 Resources / 평가 afterany:63563)**이며 server4 cap 2를 지킨다.
server1의 63561·63562는 GPU 실행 없이 취소했다. W0 및 500건마다 가중치·history checkpoint를 저장한다.
데이터가 다른 진단이므로 본표의 기존 CF 결과를 교체하지 않는다.
[선택 규칙·설정·진행 결과](experiment-reports/global/fe-qwen-cf-altdata-20261011/report-ko.md).

**FE author-hparams 추가 실행(2026-10-11 사용자 지시):** Llama3 **zsRE 2K 한 개**를 기존 수정 EasyEdit FE + history와
원저자 clamp **0.75**·step **35**(lr 0.1) 설정으로 새로 제출했다. z는 각 batch 진입 모델에서 계산한다.
server4 **63357 / official-s4-llama3-zsre-fe-author-hparams-2k**는 최신 사용자 정정으로 **cap 2**에 맞췄다.
**PENDING, `afterany:63284`**이며 BLUE 63284 종료 후 같은 lane에서 실행한다.
다른 lane은 MEMIT 63283 → AlphaEdit 63262다. 가능한 16개 상태를 검산하여 **동시 GPU 최대 2개**를 확인했고
임시 hold를 해제했다. 기존 job 번호·실험 설정을 유지하며 실행/완료 성능은 아직 없다.
[cap 2 실제 변경·검산](audits/servers/server4/llama-zsre-fe-author-hparams-20261011/cap2.json).
새 별도 `FE (author hparams + history)` 행의 zsRE 3개 셀에 상태를 기록한다.
**63357은 2026-10-11 12:18에 실패했다(29분). W0 평가와 B1 commit 뒤, B2 checkpoint를 server1로 스트림 저장하는 중 `torch.save`가 파이프에서 `unexpected pos` 오류를 냈다. W20 결과는 없다. 사용자 지시로 같은 과학 설정을 server4 **63545**로 다시 제출했다(아래 제출 기록).**
매 100 edit 재개 checkpoint와 500/1000/1500/2000 보존 checkpoint는 server1에 SHA 검증 후 저장한다.
[설정·제출·CPU 검증](experiment-reports/global/llama-zsre-fe-author-hparams-server4-20261011.md).

**FE Llama CF server3 재실행(2026-10-11 사용자 지시):** "Llama CF FE는 server3에서 빠르게 돌리자"에 따라
같은 기준(수정 EasyEdit FE + 원저자 clamp 0.75·step 35 + 누적 history, z는 각 batch 진입 모델)의 Llama CF 2K를
server3(H200) **63382 / official-s3-cf-llama3-memit-fe-author-history**로 **완료**했다(source `4841f720`, 3시간 23분). W20 all-seen R/P/N 1,992/2,000 · 3,815/4,000 · 15,874/20,000, Score 90.57(반올림 전 성공률의 조화평균)이며 62529의 90.58 / 99.70 / 95.55 / 79.22와 0.2 이내로 같다(H200 대 A6000). [W20 원자료 SHA](audits/servers/server3/fe-author-llama-cf-s3-20261011/w20-table-row-63382.json).
과학 설정은 완료 run **62529**의 config와 같고 서버 경로·runtime·출력·tracking 파일만 다르다.
모델 10개·C0 5개 파일은 server3에서 다시 해시해 server1 manifest와 같음을 확인했고,
C0는 strict 정책에서 5층 모두 Cholesky를 통과해 보정 없이 쓴다.
62529의 W20 checkpoint는 이전 FE 정리 때 삭제되어 Flu/Con을 잴 수 없으므로, 이번 실행은 W20을 server3에 보존해
Flu/Con을 이어서 평가한다. 첫 시도 **63377**은 server3 공용 W&B 설정의 허용되지 않은 키로 tracker 초기화에서
멈췄고(GPU 편집 0), 허용 키만 담은 run 전용 설정으로 다시 등록한 **63378**은 W0 평가 직후 첫 checkpoint 저장에서
`INCOMPLETE_CHECKPOINT_IDENTITY`로 멈췄다(편집 0). sequential FE runner(`6209019e`/`6d063860`)가 identity에
covariance·z policy·writer 필드를 더했는데 checkpoint 검사는 핵심 7개 필드만 정확히 허용했기 때문이다.
`4841f720`에서 핵심 7개 필드는 필수로 두고 추가 필드도 비어 있지 않으면 허용하도록 고쳤다(저장·재개 모두 identity 전체를 비교).
**`4841f720` 이전 source로 고정된 이 runner의 실행은 같은 이유로 첫 저장에서 멈춘다.**
[preparation·등록 기록](audits/servers/server3/fe-author-llama-cf-s3-20261011/).

**FE·CAKE CF Flu/Con과 FE zsRE 제출(2026-10-11 사용자 지시, 20:13 KST 관측):** "FE Flu/Con, CAKE Flu/Con 모두 각각 server1,2에 나눠서 제출하고 FE zsRE는 server4에"와 "gpt-j의 zsRE도 server4에서 제출해"에 따라 held 검사 후 release했다.
- **CF Flu/Con 6개**(source `3a1b6138`): 저장된 W20 편집 가중치를 다른 Flu/Con 칸과 같은 공용 평가기(FP32 eager, TF32 off)에 복원해 잰다. BF16으로 편집한 GPT-J·Qwen FE는 저장된 BF16 가중치를 FP32로 정확히 변환한다. server1: GPT-J FE **63537** → GPT-J CAKE **63538**(collector 63539). server2: Llama FE **63540** → Qwen FE **63541**, Llama CAKE **63542** → Qwen CAKE **63543**(collector 63544).
- **FE zsRE 2개**(server4, source `9a12b857`): Llama **63545**는 실패한 63357과 과학 설정이 같다(hparams·stream·순서 동일, config `6b1bda84` ← `b35dde63`). checkpoint를 server1로 스트림 저장하지 않고 server4에 두는 것만 다르다. GPT-J **63546**(편집) → **63547**(평가)는 CF 63389/63390과 같은 공식 FE `main.py`(BF16), L3 C0 수치 보정, 현재 모델 z이고 데이터만 zsRE 첫 2,000건이다. GPT-J zsRE 평가는 CF 행처럼 BF16이며 baseline zsRE 행은 FP32 평가다. server4 GPU 2개 중 1개는 downstream 63529가 쓰므로 세 job은 남은 한 자리에서 차례로 돈다. Qwen FE zsRE는 이후 사용자 지시("Qwen FE zsRE 제출해. server1")로 server1 **63555**(편집) → **63556**(평가)에 등록했다(source `bc2b084f`). CF 63385/63387과 데이터만 다르고(zsRE 첫 2,000건, GPT-J와 같은 변환 파일), 공식 FE `main.py`(BF16)·원 C0(보정 없음)·현재 모델 z·snapshot 설정은 같다.
- **CF Flu/Con 4칸 완료(2026-10-12 01:30 KST sacct, 모두 COMPLETED 0:0·FAILURE 없음):** GPT-J FE **63537**(10-11 22:28 완료) 평균 6.0455 bits · 0.3903 cosine → **604.55 / 39.03**, GPT-J CAKE **63538**(10-12 01:03) 6.1368 · 0.4103 → **613.68 / 41.03**, Llama FE **63540**(10-11 23:06) 6.3188 · 0.3470 → **631.88 / 34.70**, Llama CAKE **63542**(10-12 00:47) 6.2388 · 0.3050 → **623.88 / 30.50**. 2,000 case/20,000 prompts의 생성 원문을 다시 채점(entropy·TF-IDF cosine)해 run 요약과 같은 평균을 얻었고 ×100 후 half-up 둘째 자리만 적용했다. 복원한 W20 checkpoint SHA는 각 편집 run(63389·63298·63382·63296)이 기록한 W20과 같다(63382는 텐서별 FP32 SHA까지 일치). server2 collector 63544는 63541/63543을 기다리므로 run 출력에서 직접 검산했다. [GPT-J FE](audits/global/fe-cake-flucon-zsre-results-20261012/gptj-fe-flucon-63537.json) · [GPT-J CAKE](audits/global/fe-cake-flucon-zsre-results-20261012/gptj-cake-flucon-63538.json) · [Llama FE](audits/global/fe-cake-flucon-zsre-results-20261012/llama3-fe-flucon-63540.json) · [Llama CAKE](audits/global/fe-cake-flucon-zsre-results-20261012/llama3-cake-flucon-63542.json) 원자료 SHA·재집계.
- **FE zsRE 2칸 완료:** Llama **63545**(10-11 23:40 완료) W20 **94.90 / 90.43 / 32.72**이며, 같은 run의 W0 36.99 / 36.34 / 31.89는 표 W0와 같다. Qwen **63555 → 63556**(10-11 23:58 / 10-12 00:12 완료) W20은 **0.00 / 0.00 / 0.00**으로 붕괴했다(정답 token 0/6,691 · 0/6,691 · 0/11,476). W10 97.86 / 89.02 / 38.23이 W15 0.09 / 0.05 / 0.31로 무너졌고 batch 14부터 z error가 폭증했다. W20 가중치는 유한하지만 layer 5 down_proj 최대 |w|가 8.6×10^6이며, 원저자 `main.py` 자체 평가도 0.0이다. **Qwen FE zsRE 평가는 BF16이고 baseline zsRE 행은 FP32 평가다**(BF16 W0 36.48 / 35.29 / 38.43). 두 run 모두 저장 token ID로 정답 여부를 다시 만들어 `reduce.zsre`로 재집계했고 run 요약과 같다. 평가한 W20 checkpoint SHA도 편집 run 기록과 같다. [Llama zsRE](audits/global/fe-cake-flucon-zsre-results-20261012/llama3-fe-zsre-63545.json) · [Qwen zsRE](audits/global/fe-cake-flucon-zsre-results-20261012/qwen25-fe-zsre-63555-63556.json) 원자료 SHA·재집계.
[job·config·source 기록](audits/global/fe-flucon-zsre-submit-20261011/submission.json).

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

**CAKE 신규 2K 실행(2026-10-11):** server1에 3모델 × CF·zsRE 6개 job을 등록했다.
server1은 기존 작업 종료 후 CF·zsRE **두 lane(동시 GPU 최대 2개)**으로 실행한다.
이 중 **Qwen CF·zsRE는 server1 63300·63301을 취소하고 server4 63324·63325로 이관**했다.
사용자의 cap 예외 지시로 두 job 모두 선행 의존성 없이 **CF 07:28:48, zsRE 07:45:19 KST RUNNING**으로 시작했다.
당시 기존 PRICE 2개와 함께 **GPU 4개를 동시 사용**했다. 이후 최신 사용자 정정으로 **server4 cap은 2개**이며, 현재 lane은 위 FE 등록 설명을 따른다. 나머지 CAKE 4개는 server1 큐에 등록했다.
[CF cap 예외·실행 영수증](audits/servers/server4/cake-qwen25-cf-move-20261011/cap-exception-63324.json) ·
[zsRE cap 예외·실행 영수증](audits/servers/server4/cake-qwen25-zsre-move-20261011/running-63325.json).
[Qwen CF 이관·동일 설정 검증](experiment-reports/global/cake-qwen25-cf-server4-20261011.md) ·
[Qwen zsRE 이관·checkpoint 전송 검증](experiment-reports/global/cake-qwen25-zsre-server4-20261011.md).
**Qwen CF 63324·zsRE 63325는 2026-10-11 09:13·09:12 KST에 COMPLETED**(W20/2,000 edits)이며 아래 Qwen CAKE 행에 반영했다.
표 값은 각 run의 `factual/all-seen-20.json`을 공식 reducer로 다시 집계한 것이다. CF는 strict NLL R/P/N 1,626/2,000 · 2,804/4,000 · 11,388/20,000이고 Score는 반올림 전 성공률의 조화평균(67.9843)이다. zsRE는 공개-query 평가기(request macro, Loc = loc_ans 정확도, token 분모 6,691/6,691/11,476)다.
case 순서는 공식 eval-2K/zsRE 2K 순서와 같다. CF all-seen Score는 W5 90.70에서 W20 67.98로 계속 내려간다.
[CF 원자료 SHA·분자·분모](audits/servers/server4/cake-qwen25-cf-move-20261011/w20-table-row-63324.json) ·
[zsRE 원자료 SHA·token 분모](audits/servers/server4/cake-qwen25-zsre-move-20261011/w20-table-row-63325.json).
**Llama CF 63296·zsRE 63297은 2026-10-11 11:27·11:32 KST에 server1에서 COMPLETED**(W20/2,000 edits)이며 아래 Llama CAKE 행에 반영했다. 값은 각 run의 `factual/all-seen-20.json`을 같은 공식 reducer로 다시 집계한 것이다. CF는 strict NLL R/P/N 1,984/2,000 · 3,486/4,000 · 15,304/20,000, Score는 반올림 전 성공률의 조화평균(86.6461)이고, zsRE는 공개-query 평가기(token 분모 6,035/6,035/12,465)다.
**GPT-J CF 63298·zsRE 63299는 2026-10-11 14:19·14:00 KST에 server1에서 COMPLETED**(W20/2,000 edits)이며 아래 GPT-J CAKE 행에 반영했다. 같은 방식으로 CF는 R/P/N 1,994/2,000 · 3,856/4,000 · 14,642/20,000, Score 88.0724이고, zsRE token 분모는 5,557/5,557/9,694다.
네 run의 case 순서는 공식 eval-2K/zsRE 2K 순서와 같고, CF W0는 본표 W0 행과 같다. zsRE W0 행은 기존 exact-token-prefix 평가기 관측이라 이 run들의 공개-query W0(Llama 36.99 / 36.34 / 31.89, GPT-J 27.79 / 27.10 / 27.54)와 직접 비교하지 않는다. all-seen 값은 W5→W20에 완만하게만 내려간다(CF Score Llama 87.14→86.65, GPT-J 90.73→88.07).
[Llama CF](audits/servers/server1/cake-six-2k-20261011/w20-table-row-63296.json) · [Llama zsRE](audits/servers/server1/cake-six-2k-20261011/w20-table-row-63297.json) · [GPT-J CF](audits/servers/server1/cake-six-2k-20261011/w20-table-row-63298.json) · [GPT-J zsRE](audits/servers/server1/cake-six-2k-20261011/w20-table-row-63299.json) 원자료 SHA·분자·분모.
server4 디스크 여유를 위해 **zsRE checkpoint는 server1에 직접 전송하여 SHA 확인 후 원자적으로 보존**한다.
server4에는 포인터를 남기며, CF는 기존 저장 방식을 유지한다.
각 run은 **500·1,000·1,500·2,000 edits의 편집층 가중치·history·RNG를 모두 보존**한다.
CF FLU/CON은 W20 checkpoint 후속 평가로 미루며, 표에는 DEFERRED로 둔다.
Llama·GPT-J는 저자 CAKE hparams, Qwen2.5는 공식 AlphaEdit 편집 설정과 공개 knowns의
실측 causal score(job 63264, 정답 필터 통과 725개)를 사용한다.
[실행·checkpoint 보고](experiment-reports/servers/server1/cake-six-2k-20261011/report-ko.md) ·
[실제 job/source/config/dependency](audits/servers/server1/cake-six-2k-20261011/submission.json).

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
| [W0 (편집 전)](experiment-reports/global/w0-main-table-20261009.md) | 13.36 | 8.20 | 10.95 | 88.56 | 635.22 | 24.64 | 36.99 | 36.34 | 31.89 |
| FT | 56.59 | 89.25 | 73.30 | 35.50 | 449.50 | 2.90 | 14.63 | 11.82 | 25.25 |
| MEMIT | 58.88‡ | 64.75‡ | 61.70‡ | 51.83‡ | 507.87‡ | 5.38‡ | 44.30 | 39.95 | 22.30 |
| AlphaEdit | 84.80‡ | 99.30‡ | 93.23‡ | 68.59‡ | 619.92‡ | 32.33‡ | 95.18 | 91.40 | 31.10 |
| AlphaEdit-BLUE | 89.84† | 99.60† | 97.15† | 76.59† | 630.45† | 33.98† | 95.87 | 92.28 | 32.83 |
| FE (author hparams + history) | 90.57 | 99.60 | 95.38 | 79.37 | 631.88 | 34.70 | 94.90 | 90.43 | 32.72 |
| AlphaEdit+SPHERE | 86.87 | 99.50 | 94.95 | 71.68 | 619.20 | 33.64 | 95.13 | 91.36 | 31.38 |
| CAKE | 86.65 | 99.20 | 87.15 | 76.52 | 623.88 | 30.50 | 94.73 | 89.34 | 32.72 |
| PRICE (Ours) | 90.98§ | 99.90§ | 93.50§ | 81.50§ | 628.47§ | 33.53§ | 95.99§ | 91.37§ | 32.98§ |

§ 사용자 2026-10-10 결정에 따른 **최종 PRICE method** CF 2K이며, server4 job **62604**로 실행했다.
- 설정: arm `llama3-P-beta100`. β = c = β_max = 1.0, unit-lr ρ0.05, γ1, cap 끝점 cast, HC-PRICE, 요청별 early exit.
- 데이터: eval-2K 2,000건, B1–B20, baseline과 같은 순서와 context.
- arm과 runner는 server4 task branch commit `885a0e26`에 있고 main에는 아직 병합하지 않았다. resolved config sha256 `1208f9ce…`.
- 공식 method는 HC 가격 정규화를 포함한다. 이 run은 정규화를 도입하기 전에 돌린 것(`llama3-P-beta100`)이다. 정규화 유무의 차이(가장 싼 층 가격 B20까지 ≤ 1.034, 실질 예산 3% 이내)가 잡음 수준으로 예상되어, 재실행하지 않고 ours로 쓴다(사용자 결정 2026-10-11).
- Eff/Gen/Loc은 strict NLL preference R/P/N 성공률 1,998/2,000 · 3,740/4,000 · 16,300/20,000이다.
- Score는 반올림 전 성공률의 조화평균(90.9776)이다. 표시는 decimal half-up 둘째 자리다.
- 이전 FREE100 job 60103 값(99.70 / 92.78 / 82.21, Score 90.98)을 대체했다.
- CF Flu/Con은 W20 checkpoint를 baseline과 같은 생성 평가기(server1 = devbox, job **63125**)로 평가했다. 2,000 case/20,000 prompts의 평균 6.2847 bits · 0.3353 cosine에 ×100 후 half-up 둘째 자리만 적용했다. 평가 전후 가중치·RNG·checkpoint 불변을 확인했다. [Flu/Con 원자료 SHA](audits/servers/server1/price-final-flucon-20261011/llama3-P-beta100-W20.json)
- zsRE는 같은 설정의 job **62889**(W20/2,000 edits)를 baseline 행과 같은 **공개 query 평가기**로 다시 채점한 값이다(devbox **63526**, PRICE `price.eval.zsre_rescore` @013197e, 2026-10-11 사용자 지시). run 안 exact-token-prefix 평가기 값은 99.62 / 94.83 / 45.31이었다. zsRE W0 행도 같은 공개 query 평가기 값(36.99 / 36.34 / 31.89, CAKE 63297의 W0)으로 바꿨다(이전 exact-token-prefix W0 38.10 / 37.61 / 38.59). [재채점 SHA](audits/servers/server1/price-zsre-public-rescore-20261011.json). Loc(45.31)이 W0(38.59)보다 높은데, W0 예측 일치율은 62.68%다(원인 미확인). [zsRE 보고](experiment-reports/servers/server4/price-final-2k-20261010/llama3-zsre-report-ko.md)
- 500 edit마다 resumable checkpoint를 저장했다(가중치, history H, HC 통계).
[실행·수치·checkpoint 보고](experiment-reports/servers/server4/price-final-2k-20261010/llama3-cf-report-ko.md) ·
[정확한 수치·SHA](audits/servers/server4/price-final-2k-20261010/llama3-cf-results.json) ·
[이전 FREE100 출처](experiment-reports/servers/server1/official-baselines-20261008/llama-free100-table-20261009.md).

† 사용자 2026-10-09 지시에 따라 표본·순서를 대조한 기존 Llama BLUE job **39283_1**의
**B20/2,000 edits** 결과를 반영했다(새 official 재실행 결과 아님).
동일 fixed10k 파일의 첫2,000 순서가 현재 official lock과 일치한다.
Eff/Gen/Loc은 기존 strict NLL preference 집계, Score는 기존 R/P/N 성공률의 조화평균이다.
Loc은 N success 15,317/20,000 = 76.585%를 소수 둘째 자리로 반올림했다.
Flu·Con은 이 historical B020 checkpoint를 사용하는 별도 평가62876을 제출했다.
**62876**(server2)은 2026-10-11 15:09 KST에 완료됐다. 2,000 case/20,000 prompts의 평균 6.3045 bits · 0.3398 cosine에 ×100 후 half-up 둘째 자리만 적용했다(630.45 / 33.98).
평가 전후 가중치·RNG·checkpoint 불변이며 checkpoint SHA는 아래 BLUE 2K 확인 기록의 B020과 같다.
[62876 원자료 SHA](audits/servers/server2/baseline-refresh-s2-flucon-20261010/llama3-alphaedit_blue-39283_1-flucon.json).
native source/runtime·평가기 차이와 분모 및 대조 근거는
[BLUE 2K 확인 기록](experiment-reports/servers/server1/official-baselines-20261008/llama-blue-2k-table-check.md)에 구분했다.

‡ 사용자 `USER-SIDE-GH-LLAMA-CF-MEMIT-ALPHA-2K-TABLE-20261009-R1`의 명시 예외로
기존 native MEMIT **42658**, native AlphaEdit **42657**의 **B020/2,000 edits**를 반영했다.
새 official rerun 완료 또는 최종 10K 결과가 아니다. R/P/N 분모는 2,000/4,000/20,000,
Score는 반올림 전 성공률의 조화평균이며 표시는 소수 둘째 자리 decimal half-up이다.
Flu/Con은 이 historical B020 checkpoint의 별도 평가62874/62875를 제출했다.
MEMIT **62874**(server2)는 완료됐다. 2,000 case/20,000 prompts의 평균 5.0787 bits · 0.0538 cosine에 ×100 후 half-up 둘째 자리만 적용했다(507.87 / 5.38).
평가 전후 가중치·RNG·checkpoint 불변이며 checkpoint SHA는 아래 근거의 MEMIT B020과 같다.
AlphaEdit **62875**(server2)도 2026-10-11 11:59 KST에 완료됐다. 2,000 case/20,000 prompts의 평균 6.1992 bits · 0.3233 cosine에 ×100 후 half-up 둘째 자리만 적용했다(619.92 / 32.33).
평가 전후 가중치·RNG·checkpoint 불변이며 checkpoint SHA는 아래 근거의 AlphaEdit B020과 같다.
[62874 원자료 SHA](audits/servers/server2/baseline-refresh-s2-flucon-20261010/llama3-memit-42658-flucon.json) ·
[62875 원자료 SHA](audits/servers/server2/baseline-refresh-s2-flucon-20261010/llama3-alphaedit-42657-flucon.json).
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
MEMIT Flu/Con은 W20 checkpoint(62073) 평가 **62870**(server2)의 완료값이다(평균 5.6202 bits · 0.0624 cosine, 562.02 / 6.24). [62870 원자료 SHA](audits/servers/server2/baseline-refresh-s2-flucon-20261010/qwen25-memit-62073-flucon.json).
[새 완료 수치·raw/source/config 및 평가 준비](experiment-reports/servers/server2/baseline-refresh-s2-flucon-20261010/report-ko.md).
62087은 B10 `eigh` CUDA OOM으로 FAILED/W20 없음이며 B9 checkpoint를 보존한다.
cold **62534**는 실행 전 취소했고, **62538** `s2-qwen25-zsre-sphere-resume-b9`는
원62087의 B9를 복원해 **B10/1,000 edits checkpoint를 저장**했다. 최종 실패 원인은 CUDA OOM이 아닌
`RESOURCE_BLOCKED_STORAGE_KEEP_CHECKPOINT`(디스크 여유 32 GiB 하한)이다.
2026-10-11 사용자 지시로 server4 **63303 / s4-qwen25-zsre-sphere-resume-b10**을 held 검사 후 release했다.
이후 **07:05 KST 사용자 요청으로 63303과 CF FLU/CON 평가 62873을 모두 취소**했다.
server1–4 전체에서 실행·대기 SPHERE job 0개를 확인했다.
2026-10-11 21:3x에 두 작업을 server1(63569 Flu/Con, 63570 zsRE 재개)로 다시 등록했으나, 사용자 지시("sphere는 안올려도 된다")로 21:49에 실행 전 취소했다(GPU 사용 없음). 아래 셀은 기존 취소 상태를 유지한다. [등록·취소 기록](audits/global/qwen-sphere-s1-resubmit-20261011/submission.json).
[취소 영수증](audits/global/sphere-stop-20261011/cancellation.json). 원62087 B1–B9와
62538 B10의 출처·가중치·history·context·RNG는 보존한다. hparams와 관련 수치 코드 23개 파일은 동일하다.
server4의 별도 파일시스템에 checkpoint 두 slot을 사전 할당했고, 저장 실패 시 이전 latest를 유지한다.
CPU26 및 전체 2K query 대조 PASS. server4 GPU 재개 및 최종 W20은 아직 미관측이다.
[server4 이전·B10 검증·실제 등록](experiment-reports/servers/server4/qwen-zsre-sphere-b10-resume-20261011/report-ko.md).
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
| MEMIT | 62.23 | 68.50 | 65.83 | 54.30 | 562.02 | 6.24 | 41.49 | 39.44 | 26.39 |
| AlphaEdit | 83.03 | 99.05 | 97.55 | 63.36 | 618.99 | 33.92 | 85.05 | 78.18 | 30.79 |
| AlphaEdit-BLUE | 84.03 | 99.65 | 97.33 | 64.97 | ING: Qwen BLUE L2 95 W20 Flu/Con (devbox 63637) | ING: Qwen BLUE L2 95 W20 Flu/Con (devbox 63637) | ING: janghj-zsre-blue-120121 (rent 120121, L2 95) | ING: janghj-zsre-blue-120121 (rent 120121, L2 95) | ING: janghj-zsre-blue-120121 (rent 120121, L2 95) |
| FE (author hparams + history) | 50.18 | 50.90 | 49.95 | 49.72 | PENDING: s2-flucon-qwen25-fe_author_native-63385 (63541) | PENDING: s2-flucon-qwen25-fe_author_native-63385 (63541) | 0.00 | 0.00 | 0.00 |
| AlphaEdit+SPHERE | 83.73 | 99.40 | 97.70 | 64.37 | CANCELLED: s2-flucon-qwen25-sphere-62079 (62873) | CANCELLED: s2-flucon-qwen25-sphere-62079 (62873) | CANCELLED: s4-qwen25-zsre-sphere-resume-b10 (63303) | CANCELLED: s4-qwen25-zsre-sphere-resume-b10 (63303) | CANCELLED: s4-qwen25-zsre-sphere-resume-b10 (63303) |
| CAKE | 67.98 | 81.30 | 70.10 | 56.94 | PENDING: s2-flucon-qwen25-cake-63324 (63543) | PENDING: s2-flucon-qwen25-cake-63324 (63543) | 88.39 | 82.57 | 33.79 |
| PRICE (Ours) | 87.63※ | 99.00※ | 95.55※ | 73.17※ | 620.95※ | 31.46※ | 98.72※ | 93.97※ | 45.17※ |

※ Qwen PRICE (Ours)는 HC 가격을 정규화한 최종 method 재실행(arm `qwen25-PN-beta150`, β = c = β_max = 1.5)으로 채운다. CF는 server4 **63144**(B1–B15)를 W15에서 멈추고 **63281**(B16–B20, state join 확인)로 이어 돌려 완료했다(server4 GPU cap 2 정리, 2026-10-11 사용자 결정). W20 all-seen 성공률은 R/P/N 1,980/2,000 · 3,822/4,000 · 14,633/20,000이고, Score는 반올림 전 성공률의 조화평균(87.6310)이다. [원자료 SHA](audits/servers/server4/price-final-2k-20261010/qwen25-cf-results.json). CF Flu/Con은 W20 가중치(63281 W20과 SHA 동일)를 baseline과 같은 생성 평가기로 devbox(server1)에서 평가했다(job **63305**). 2,000 case/20,000 prompts의 평균 6.2095 bits · 0.3146 cosine에 ×100 후 half-up 둘째 자리만 적용했다. 평가 전후 가중치·RNG·checkpoint 불변을 확인했다. [Flu/Con 원자료 SHA](audits/servers/server1/price-final-flucon-20261011/qwen25-PN-beta150-W20.json). zsRE는 같은 arm의 server4 **63278**(r3, official zsRE stream·evaluator, W20/2,000 edits)이다. W0가 본표 zsRE W0 행(36.42 / 35.26 / 38.40)과 같고, 2,000건 query도 그 W0 원자료와 같다. Eff/Gen/Loc은 요청별 token 정확도의 평균이며 token 분모는 6,691/6,691/11,476이다. Loc(45.17)이 W0(38.40)보다 높은데, W0 예측 일치율은 52.31%다(Llama와 같은 양상, 원인 미확인). r1 **63145**의 W15는 r3 W15와 가중치·history·all-seen 값까지 같다. [zsRE 원자료 SHA](audits/servers/server4/price-final-2k-20261010/qwen25-zsre-results.json). HC 가격을 정규화하지 않은 rent run(**101706**, W20 98.65 / 94.25 / 72.48)은 본표에 넣지 않고 ablation으로 남긴다. 정규화하지 않으면 가장 싼 층 가격이 B10에서 1.159(최대 1.386)까지 올라, 편집 강도가 14% 이상 줄기 때문이다.

**Qwen AlphaEdit-BLUE 재선정(2026-10-11 사용자 지시, 설계 세션 janghj-7d 경유):** 기존 값은 모두 L2=1 실행이다. CF **61962** W20 77.98 / 97.70 / 96.75 / 55.86, 같은 run의 Flu/Con 602.99 / 37.43, zsRE **61964** 58.62 / 53.66 / 5.73. 같은 L2=1을 rent에서 다시 돌린 CF W20 Score는 67.76이라 재현되지 않았다. 그래서 rent L2 grid {1, 10, 95} 가운데 W20 CF Score가 가장 높은 L2(계약 규칙)로 CF를 정한다. 같은 L2로 zsRE(rent)를 돌리고, Flu/Con은 그 L2의 CF W20 checkpoint로 잰다. 값과 원자료 SHA는 설계 세션이 보내는 대로 채운다. L2=1 두 실행(server2 77.98, rent 67.76)은 grid 표와 각주의 출처로 보존한다. 61962 W20 checkpoint는 rent `checkpoints/odeedit-baseline-archive-20261011/server2/`에 있다.
**승자 L2 = 95 (2026-10-12 01:2x KST, 설계 세션 janghj-7d):** rent 선택 job **120120**(source `5f758ae8`, grid root `c78d1809`)이 계약 규칙으로 정했다(CF_BLUE_GRID_SELECTED). grid W20 CF Score는 L2=1 67.76, L2=10 79.91, **L2=95 84.03**이고, server2의 L2=1(77.98)을 넣어도 승자는 같다. 선택 증명 `qwen25-blue-selection.json` sha256 `f6acdce2…`. CF 칸은 rent **112048**(official runner `c78d1809`, L2 95) W20 값이다. 원자료 devbox 사본 `local/qwen-blue-grid-rent-20261011/qwen25-cf-alphaedit_blue-l2-95/chain-job112048/factual/W20.json` (sha256 `1144a67d1378a3e759cfda37108f87d973235ea2a0b0f05838d9a19e3dab12aa`)을 GH가 `official/evaluation/reduce.py::counterfact`로 다시 계산해 99.65 / 97.325 / 64.965, Score 84.0268을 확인했다 (2,000 요청, 순서 digest `0b912d11…` = CF stream lock, 표시는 정수 분자 half-up). all-seen Score 추이는 W5 89.84 → W10 86.74 → W15 85.55 → W20 84.03이다. zsRE는 rent **120121**(L2 95)이 01:21에 시작했고, Flu/Con은 같은 W20 checkpoint(sha256 `733c45da…`)를 devbox **63637**(01:55 시작, 공용 Flu/Con 평가기 FP32 eager·TF32 off, frozen source `3a1b6138`, config sha `99f96ea4…`)로 잰다.

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
| [W0 (편집 전)](experiment-reports/global/w0-main-table-20261009.md) | 24.44 | 17.00 | 19.30 | 82.48 | 621.71 | 30.16 | 27.79 | 27.10 | 27.54 |
| FT | 58.97 | 88.75 | 66.75 | 40.61 | 510.98 | 3.04 | 23.15 | 17.95 | 0.62 |
| MEMIT | 80.63 | 97.90 | 95.38 | 60.58 | 577.93 | 36.84 | 93.52 | 88.86 | 30.81 |
| AlphaEdit | 88.22 | 99.70 | 96.33 | 73.56 | 614.00 | 40.76 | 99.69 | 96.45 | 27.94 |
| AlphaEdit-BLUE | 89.23 | 99.55 | 97.53 | 75.06 | 621.35 | 42.00 | 99.75 | 95.71 | 28.83 |
| FE (author hparams + history) | 85.39 | 98.50 | 94.80 | 69.29 | 604.55 | 39.03 | PENDING: official-s4-gptj-zsre-fe-native-repaired (63546) | PENDING: official-s4-gptj-zsre-fe-native-repaired (63546) | PENDING: official-s4-gptj-zsre-fe-native-repaired (63546) |
| AlphaEdit+SPHERE | 88.39 | 99.70 | 95.73 | 74.27 | 616.18 | 40.89 | 99.67 | 96.29 | 28.00 |
| CAKE | 88.07 | 99.70 | 96.40 | 73.21 | 613.68 | 41.03 | 99.73 | 97.01 | 28.25 |
| PRICE (Ours) | 88.22¶ | 99.80¶ | 96.23¶ | 73.57¶ | 620.09¶ | 41.61¶ | 99.71¶ | 96.87¶ | 29.53¶ |

GPT-J W0 Flu/Con은 동일 native full2K 완료/등록을 찾지 못해 server4의 **63219**로 generation-only 평가를 등록했다.
held 검사 후 release, `afterany:63144` PENDING이며 두 생성 셀만 job 번호로 표시한다. 2026-10-11 05:00 server4 GPU cap 2·ours 우선 정리로 63219(모델 로드 2분42초 뒤)를 멈추고 같은 run.sh·config로 **63282**(`afterany:63277`, ours 뒤)로 다시 제출했다.
기존 W0 factual/zsRE 수치는 변경하지 않았다. [확인 범위·제출 근거](experiment-reports/global/gptj-w0-flucon-20261011/report-ko.md).
**63282는 2026-10-11 09:09 KST에 COMPLETED**(편집0, cold W0)이며 두 생성 셀을 채웠다. 2,000 case/20,000 prompts의 평균 6.2171 bits · 0.3016 cosine에 ×100 후 half-up 둘째 자리만 적용했다(621.71 / 30.16).
생성 프로파일·seed·reference·tokenizer는 SH2의 GPT-J W20 Flu/Con 평가와 같고, 모델 파일 SHA는 load 전에 검사했다. [63282 원자료 SHA](audits/servers/server4/gptj-w0-flucon-20261011/result-63282.json).
AlphaEdit Flu/Con은 W20 checkpoint(61778) 평가 **62866**(server2)의 완료값이다(평균 6.1400 bits · 0.4076 cosine, 614.00 / 40.76). [62866 원자료 SHA](audits/servers/server2/baseline-refresh-s2-flucon-20261010/gptj-alphaedit-61778-flucon.json).
AlphaEdit-BLUE Flu/Con은 W20 checkpoint(61779) 평가 **62867**(server2)의 완료값이다(평균 6.2135 bits · 0.4200 cosine, 621.35 / 42.00). [62867 원자료 SHA](audits/servers/server2/baseline-refresh-s2-flucon-20261010/gptj-alphaedit_blue-61779-flucon.json).

¶ 최종 PRICE method의 GPT-J 2K다. CF는 rent job **101707**, zsRE는 server4 job **63027**이며 둘 다 W20/2,000 edits다.
- 설정: arm `gptj-P-beta075`(resolved `d8f5ff72…`). β = c = β_max = 0.75, unit-lr ρ0.05, γ1, cap 끝점 cast, HC-PRICE, 요청별 early exit, sink EOT 규칙과 anchor guard 20.
- 공식 method는 HC 가격 정규화를 포함한다. 이 run은 정규화를 도입하기 전에 돌린 것(`gptj-P-beta075`)이다. 정규화 유무의 차이(가장 싼 층 가격 CF 1.012(B10)·zsRE 1.014(B16) 이내)가 잡음 수준으로 예상되어, 재실행하지 않고 ours로 쓴다(사용자 결정 2026-10-11).
- CF R/P/N 성공률은 1,996/2,000 · 3,849/4,000 · 14,713/20,000이고, Score는 조화평균 88.2203이다. W0가 본표 W0 행과 같다.
- zsRE는 job **63027**의 W20을 baseline 행과 같은 **공개 query 평가기**로 다시 채점한 값이다(devbox **63527**, 2026-10-11 사용자 지시). run 안 exact-token-prefix 평가기 값은 99.81 / 96.96 / 29.59였다. zsRE W0 행도 공개 query 값(27.79 / 27.10 / 27.54, CAKE 63299의 W0)으로 바꿨다(이전 27.83 / 27.15 / 27.59). [재채점 SHA](audits/servers/server1/price-zsre-public-rescore-20261011.json).
- CF Flu/Con은 W20 가중치를 baseline과 같은 생성 평가기로 devbox(server1)에서 평가했다(job **63207**). 2,000 case/20,000 prompts의 평균 6.2009 bits · 0.4161 cosine에 ×100 후 half-up 둘째 자리만 적용했다. 평가 전후 가중치·RNG·checkpoint 불변을 확인했다. [Flu/Con 원자료 SHA](audits/servers/server1/price-final-flucon-20261011/gptj-P-beta075-W20.json)
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
| 가격 | 간섭 가격 π(batch leave-one-out) + **HC-PRICE**(writer history cache의 누적 간섭, drift·cascade 할증). HC 배수를 곱한 뒤 요청마다 가장 싼 층이 1이 되도록 **다시 정규화**한다(`hc_normalize`) |
| 예산 | 요청별 공유 weighted-L1 Σ π‖R‖/a ≤ β. **c = β**라서 층별 cap은 기본 예산에서 걸리지 않는다. 미충족 요청만 β·max π까지 확장한다 |
| step 단위 | anchor-unit lr ρ = 0.05, cap 끝점 FP32 cast 수정 |
| γ | **1** (가격을 그대로 쓴다) |
| early exit | F < τ_F(0.05)이면 요청별로 멈추고, ≥ 2τ_F이면 재개한다. 업데이트가 약 1/3로 줄어든다 |
| sink 규칙 | GPT-J·Qwen: position 0 또는 W0 sink pair에 EOT, anchor guard 20. Llama는 BOS라 불필요 |
| context·sample | 공용 패키지(Llama `cf14b`, Qwen `5c01bc1a`, GPT-J `b454d8`). baseline과 같은 eval-2K 순서·batch 경계 |
| arm | 공식: `<model>-PN-beta<b>` (β = c = β_max, γ1, HC, HC 정규화). Llama·GPT-J 본표 run은 정규화 도입 전 `<model>-P-beta<b>`다. early exit는 runner(`project/run_scripts/price_early_exit`)가 적용한다. arm과 runner는 server4 task branch에 있고 main에는 아직 병합하지 않았다 |

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
| Llama | server4 **62604** 완료: W20 99.90 / 93.50 / 81.50, Score 90.98. Flu/Con devbox **63125** 완료: 628.47 / 33.53 (모두 본표 반영) | server4 **62889** 완료: W20 99.62 / 94.83 / 45.31 (본표 반영) |
| Qwen | 정규화 HC(`qwen25-PN-beta150`) server4 **63144**(B1–B15) → **63281**(W15에서 이어 돌림, state join 확인) 완료: W20 99.00 / 95.55 / 73.17, Score 87.63 (본표 반영). Flu/Con devbox **63305** 완료: 620.95 / 31.46 (본표 반영). 정규화 전 HC run(rent **101706**)은 ablation으로 남긴다. 비교용 β = c 2.0(server4 **62845**)은 B10 진입 때 anchor guard로 멈췄다(편집이 쌓여 생긴 sink) | 정규화 HC server4 **63278**(r3) 완료: W20 98.72 / 93.97 / 45.17 (본표 반영). r2 **63276**은 cap 2 정리로 B2에서 멈췄다(r2의 W0·B1은 r1과 상태 해시까지 같음). 첫 run **63145**는 server4 /data 여유 공간 부족(runner 저장공간 guard 16 GiB)으로 B17에서 멈췄다(B15 all-seen 98.83 / 94.66 / 44.30). zsRE는 resume 경로가 없어 같은 preparation·같은 source(e0469ca2)로 처음부터 다시 돌렸고, r3의 W15는 r1과 가중치·history·all-seen 값까지 같다. r1 원자료는 devbox에 보관했다 |
| GPT-J | rent **101707** 완료: W20 99.80 / 96.23 / 73.57, Score 88.22. Flu/Con devbox **63207** 완료: 620.09 / 41.61 (모두 본표 반영) | server4 **63027** 완료: W20 99.81 / 96.96 / 29.59 (본표 반영) |

- **HC 가격 정규화(2026-10-11, 공식 method에 포함):** HC 배수를 곱한 뒤 요청마다 가장 싼 층의 가격이 1이 되도록 다시 나눈다(method 항목 `hc_normalize`). 정규화하지 않으면 HC가 배분뿐 아니라 강도도 바꾼다. 가장 싼 층 가격 중앙값이 Qwen 2K B10에서 1.159(최대 1.386)였고, Llama 2K B20은 1.034, GPT-J 2K B10은 1.012였다. **본표(사용자 결정):** Qwen은 정규화 HC로 다시 돌린 값을 쓴다. Llama·GPT-J는 정규화 도입 전 run이지만, 차이가 잡음 수준으로 예상되어 재실행하지 않고 그대로 쓴다.
- 모든 2K run은 W5·W10·W15·W20마다 all-seen 평가와 resumable checkpoint(가중치, history H, HC 통계)를 남긴다.
- **Llama 3K 연장과 일반 능력 평가(2026-10-11 사용자 요청):** Llama ours를 W20 checkpoint에서 3,000건까지 이어 편집했다(server4 **63238**, B21–B30 = fixed10k 순서의 records[2000:3000], baseline과 같은 표본). 63238은 W25 저장 뒤 B28에서 같은 저장공간 guard로 멈췄고(W25 all-seen 2,500건: 99.92 / 91.18 / 80.67), W25부터 **63277**로 이어 돌려 **완료**했다(B26·B27의 편집 전후 상태 해시가 63238과 같음). **W30 all-seen 3,000건: 99.77 / 88.82 / 79.16**(R/P/N 2,993/3,000 · 5,329/6,000 · 23,749/30,000, Score 88.46). 앞 2,000건의 순서는 2K 본표 run(62604)과 같다. W30 가중치·history·RNG를 저장했다. [W30 원자료 SHA](audits/servers/server4/price-final-2k-20261010/llama3-3k-results.json). AlphaEdit Figure 3의 6개 과제(SST·MRPC·CoLA·RTE·MMLU·NLI, F1)를 500건마다 MEMIT·AlphaEdit(native 42658/42657)·AlphaEdit-BLUE(39283_1)·Ours로 비교한다. 평가기는 job 42706과 같다(BLUE glue_eval, rows 10–110, 0-shot, FP32). devbox **63245**의 W0가 42706 W0와 6개 과제 F1이 같다(평균 68.48, 행 차이는 확률값 1e-16 수준뿐). baseline 1,500·2,500건 checkpoint는 원 runner로 다시 만들어 원 commit 해시와 대조한다(server4 **63260/63261/63262**). 그림은 생성 기준 F1, 대안 지표도 함께 기록한다. **최종 그림 완료(2026-10-11 20:15 KST):** 4방법 × 0–3,000건(500 간격) 28개 지점을 모두 평가했다. 평가 job은 devbox **63245**(W0, MEMIT·AlphaEdit 500/1000/2000/3000, Ours 500–2000)·**63336**(Ours 2500/3000), server4 **63529**(baseline 1,500·2,500), 기존 42706(BLUE 500/1000/2000/3000)이다. 1,500·2,500건 baseline checkpoint는 replay MEMIT 63283·BLUE 63284·AlphaEdit 63262로 만들었고 원 run과 25 batch 해시가 일치한다. server4 W0는 devbox W0와 6과제 F1이 같고 600행의 생성·답·정답 여부도 같다(확률값만 최대 1.7e-5 차이). 6과제 평균 생성 F1(W0 68.48): Ours 68.54 / 69.10 / 68.03 / 65.75 / 66.01 / 65.27(500–3,000건)로 500·1,000·1,500·2,500건에서 가장 높다. 3,000건은 AlphaEdit-BLUE 66.95, Ours 65.27, AlphaEdit 64.11, MEMIT 0.00(2,000건부터 모든 출력 invalid)이다. MMLU는 1,500건부터 Ours가 세 방법 중 가장 낮다(3,000건 47.51, W0 56.24). [그림·과제별 수치·보고서](experiment-reports/global/downstream-3k-20261011/report-ko.md).
- zsRE는 CF와 같은 설정을 쓴다. 데이터는 official zsRE first-2K stream, 평가는 official zsRE evaluator다.
- Qwen·GPT-J zsRE는 no-BOS 모델이라 zsRE sink scan을 먼저 해야 한다.

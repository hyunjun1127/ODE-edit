# JLZ native 기준 재설계 및 실제 8B pilot 결과

2026-10-01. **native KL(current∥entry)을 기본 방법으로 확정하고, 실제 Llama-3-8B-Instruct에서 BS2 pilot을 실행했다.** 반환 후보의 목표 NLL은 크게 감소했지만 120회 호출 안에 설정 residual 기준을 만족하지 못했다. 가중치·H를 commit하지 않고 종료했으며 두 번째 batch는 실행하지 않았다.

## 이번에 바꾼 기준과 구현

새 실행 경로는 `project/run_scripts/jlz_pilot/`이다. KL 방향 선택 옵션 없이 native 방향만 사용한다. 첨부 `jlz_ref` 및 v1 CPU 결과는 감사 자료로 보존했다.

|항목|이번 native 기준|
|---|---|
|손실|활성 요청 NLL + .0625·KL(current∥entry)의 합 + 층·요청별 norm decay|
|decay/clamp|`.5 ||R_lr|| / ||h_lr||²`, 반경 `.75 ||h_lr||`|
|층과 공동 변수|L4–L8, 두 요청의 R을 함께 최적화|
|실현|`adj=solve(15000 C0+H+KKᵀ,K)`, `W_eff=W_entry+(R.double()@adj.T).float()`|
|native key 평균|context group 내부 평균 후 group 평균; canonical .5, 나머지 각각 .1|
|native C0|`mom2.mom2 / mom2.count`를 FP32로 계산한 뒤 FP64로 변환; count66,019,200|
|기준 분포|현재 batch 진입 모델의 KL teacher를 고정|
|commit|최종 반환 평가의 W_eff 그대로 사용. 수렴 통과 전에는 W/H 유지|

파이프라인은 **진입 teacher·anchor·key 측정 → 고정 adj 계산 → 실제 W_eff를 적용한 joint loss/gradient → block prox로 decay·clamp 처리 → 수렴 확인 → commit/history**다. key와 teacher를 최적화 중 계속 바꾸지 않는다.

입력 6개, solver 12개, 실제 Oracle를 이용한 tiny Llama 통합 3개로 **CPU 검사 21개가 통과**했다. 실제 Llama tokenizer로 pinned native prompt·target·lookup과 비교한 5개 구성도 통과했다. 이 중 multi-token target은 합성 text fixture이며 이번 실제 데이터의 네 target은 모두 single token이다.

## 실제 실행 범위와 비용

|항목|실측|
|---|---|
|job|56668, `odeedit_jlz_native_pilot_v2`|
|실행 서버|server1/devbox, A6000 1개; server3 작업 변경 없음|
|모델|Meta-Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`|
|정밀도|W/activation/R FP32, solve FP64, TF32 off|
|runtime|PyTorch2.9.1+cu128 / Transformers4.57.1|
|W0 관측|고정 첫4요청|
|실제 joint fit|첫2요청, case_id16186·3743, BS2 한 batch|
|미실행|다음 case_id1481·18140의 batch, 실제 commit·history append|
|process / Slurm 경과|218.620초 / 225초(1GPU×3분45초)|
|geometry / solver 시간|81.325초 / 122.101초|
|PyTorch GPU peak allocated|36.0507GiB|
|Slurm host MaxRSS|21,196,252KiB|
|optimizer 호출|120 = 초기1 + 수락101 + backtrack17 + 반환1|
|parity 포함 oracle|122; prompt chunk forward/backward 각각854|

모델·데이터·6context·Wikipedia C0는 기존 로컬 캐시를 사용했고 다운로드나 covariance 재계산은 하지 않았다. GPU는 Slurm으로 별도 할당받았으며 종료 후 반환됐다. 원본 모델 파일과 checkpoint는 쓰지 않았다.

## 반환 후보의 결과

아래 NLL은 native 여섯 context의 target 평균이며 **commit된 모델의 성능이 아니라 solver 반환 후보의 fit 결과**다.

|요청|진입 NLL|반환 후보 NLL|후보 KL(current∥entry)|commit|
|---|---:|---:|---:|---|
|16186: Jeff Hardy → FIFA|11.857386|0.000445231|0.052289959|아니오|
|3743: Uwe Barschel → Athens|12.210334|0.000673236|0.013350378|아니오|

초기 전체 목적은 24.067719, 반환 목적은 **0.100117208**이다. 반환점의 NLL 합은 .001118468, weighted KL 합은 .004102521, decay는 .094896220이다. 재계산한 합과 receipt 값이 일치한다. 원래 native compute_z의 요청별 조기 종료와 이번 joint solver의 stationarity 종료는 다르므로 낮은 NLL만으로 수렴을 선언하지 않는다.

|수렴·기술 항목|측정 및 판정|
|---|---|
|설정 tol|1e-4, 이 pilot의 임시 기준이며 production 교정값 아님|
|최종 normalized prox residual|**.004527743**, 기준 미달|
|gradient normalization scale|20.770494|
|최대 block KKT stationarity violation|.061276880|
|종료|**BUDGET_STOP / CALL_CAP**, commit_eligible=false|
|비영점 full/suffix loss 오차|0|
|비영점 full/suffix gradient 정규화오차|0|
|유한값·clamp|PASS; feasibility violation0|
|OOM·exception|없음|

Slurm은 프로그램의 exit2를 받아 `FAILED/2:0`으로 기록했다. 이 실행의 exit2는 사전 정의한 미수렴 중단이며 기술 exception과 구분한다. cap·tol·decay를 결과에 맞춰 바꾸거나 미수렴 후보를 기록하지 않았다.

## 층별 동작

|층|활성 block/2|R norm|진입 key에서 ΔK norm|ΔK 제곱 norm 비중|
|---|---:|---:|---:|---:|
|L4|1/2|.348835|.204099|3.205%|
|L5|2/2|.569469|.327717|8.262%|
|L6|2/2|1.399376|.910295|63.748%|
|L7|2/2|.820385|.487325|18.270%|
|L8|2/2|.512833|.291020|6.515%|

전체10 block 중9개가 nonzero다. L4의 두 번째 요청 block만 정확히0이며 다섯 층 모두 적어도 하나의 요청에 사용됐다. 최대 norm/radius는 **.318810**으로 반환점에서 clamp에 붙은 block은 없다. 위 비중은 진입 key에서의 실현 변위 크기이며, 층별 인과적 성능 기여율이나 수렴된 최적 배분을 뜻하지 않는다.

## 판정과 남은 확인

**native 목적의 입력·실현 사상·공동 gradient가 실제 8B에서 동작하고 낮은 목표 NLL의 후보를 만들었다. 그러나 120회 budget에서 수렴 완료는 확인하지 못했다.** 따라서 이번 결과는 파이프라인 전체 성공이나 실제 편집 성능 개선의 증거로 승격하지 않는다.

8B에서 최종 W_eff commit, post-write history, 다음 batch의 누적 효과는 실행되지 않았다. 이 동작들은 CPU 통합 검사에서는 확인됐지만 이번 GPU 결과로 추가 확인한 것은 아니다. Transformers4.44.2와의 runtime byte parity, 정밀도 바닥 교정, matched MEMIT-H 비교, R/P/N 전체 및 장기 retention도 미검증이다.

후속 판단은 표본 수를 늘리기 전에 **같은 작은 문제에서 solver의 수렴 비용과 FP32 정밀도 기준을 교정하는 것**이다. 이번 cap을 사후에 늘린 결과를 같은 pilot로 합치지 않는다. 현재 자료만으로 .00453을 허용하도록 기준을 완화하지 않는다.

## 근거와 재현

- [재설계 문서](/mnt/raid5/janghj/ODE-edit/plans/global/2026-10-01-jlz-native-pilot-v2/design-ko.md), [고정 contract](/mnt/raid5/janghj/ODE-edit/plans/global/2026-10-01-jlz-native-pilot-v2/contract.json)
- [집계](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/2026-10-01-jlz-native-pilot-v2/summary.json), [실행 receipt](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/2026-10-01-jlz-native-pilot-v2/receipt.json)
- [요청별 수치](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/2026-10-01-jlz-native-pilot-v2/per-request.csv), [수락된 목적 trajectory](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/2026-10-01-jlz-native-pilot-v2/accepted-objective.csv), [층별 값](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/2026-10-01-jlz-native-pilot-v2/layer-allocation.csv)
- [native tokenizer parity](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/2026-10-01-jlz-native-pilot-v2/prompts-parity.json), [source·artifact manifest](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/2026-10-01-jlz-native-pilot-v2/manifest.json)

CPU 재현 및 Slurm 실행 경로는 [README](/mnt/raid5/janghj/ODE-edit/project/run_scripts/jlz_pilot/README.md)에 있다. 원 실행 source 스냅샷과 Slurm 로그는 `local/jlz-native-pilot-v2/attempt-01/`에 보존했다. 원 실행의 Python source SHA는 현재 source와 모두 일치한다.
